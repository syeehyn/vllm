# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP query-row sharding of the DSA indexer prefill (see PrefillRowShard):
each rank scores a contiguous slice; the group exchanges int32s per row."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
import vllm.v1.attention.backends.mla.indexer as indexer
from vllm.config import AttentionConfig, CUDAGraphMode
from vllm.model_executor.layers.sparse_attn_indexer import PrefillRowShard

MIN_ROWS = AttentionConfig().indexer_tp_shard_min_rows


def _chunk(start: int, end: int) -> SimpleNamespace:
    """Chunk over rows [start, end); bounds derive from the global row."""
    n = end - start
    return SimpleNamespace(
        token_start=start,
        token_end=end,
        cu_seqlen_ks=torch.zeros(n, dtype=torch.int32),
        cu_seqlen_ke=((torch.arange(n) * 7 + start) % 65).int(),
    )


@pytest.mark.parametrize("world", [2, 3, 8])
def test_shard_narrowing_covers_every_row_once(world: int) -> None:
    """Narrowed windows tile the prefill rows exactly once, carry the causal
    bounds of those global rows, and skip chunks foreign to this rank."""
    decode, rows = 5, 997
    edges = [decode, decode + 330, decode + 337, decode + rows]
    chunks = [_chunk(lo, hi) for lo, hi in zip(edges, edges[1:])]
    sizes = [rows // world + int(r < rows % world) for r in range(world)]
    covered = []
    for rank in range(world):
        shard = PrefillRowShard(decode, sizes, rank)
        for chunk in chunks:
            window = shard.narrow(chunk)
            if window is None:
                continue
            start, end, ks, ke = window
            lo, hi = start - chunk.token_start, end - chunk.token_start
            torch.testing.assert_close(ks, chunk.cu_seqlen_ks[lo:hi])
            torch.testing.assert_close(ke, chunk.cu_seqlen_ke[lo:hi])
            covered.append((start, end))
    covered.sort()
    # Windows must be disjoint, contiguous and exactly cover the prefill rows.
    assert [s for s, _ in covered] == [decode, *[e for _, e in covered][:-1]]
    assert covered[-1][1] == decode + rows


def test_shard_exchange_reassembles_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gathered indices land only on real prefill rows: leading decode rows
    and trailing CUDA-graph padding stay untouched, and the full k-pool
    width (tail columns included) is exchanged."""
    decode, pad, topk, kpool = 5, 3, 8, 4
    width = topk + kpool - 1
    sizes = [17, 9, 21]
    buffer = torch.full((decode + sum(sizes) + pad, width), -7, dtype=torch.int32)
    gathered = torch.arange(sum(sizes) * width, dtype=torch.int32).reshape(-1, width)
    calls = []

    def fake_group():
        def all_gatherv(local, dim, sizes):
            calls.append((local, sizes))
            return gathered

        return SimpleNamespace(all_gatherv=all_gatherv)

    monkeypatch.setattr(sparse_indexer, "get_tp_group", fake_group)
    for rank in range(len(sizes)):
        PrefillRowShard(decode, sizes, rank).exchange_topk(buffer, width)
    assert calls[0][1] == sizes and calls[0][0].is_contiguous()
    torch.testing.assert_close(buffer[decode : decode + sum(sizes)], gathered)
    assert torch.all(buffer[:decode] == -7), "decode rows stay outside"
    assert torch.all(buffer[decode + sum(sizes) :] == -7), "padding untouched"


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_balanced_row_shard_partitions_and_declines(tp_size: int) -> None:
    """The split covers every row once with >= 1 row per rank and declines
    below the activation floor (or on a single rank)."""
    n = MIN_ROWS
    shapes = [
        ([4 * n], [4 * n]),  # fresh prompt: the full causal cost ramp
        # uneven requests at different context depths
        ([2 * n, 7, 4 * n, n // 3 + 5], [2 * n, 9007, 4 * n + 500, n // 3 + 60005]),
    ]
    ratio = 4
    for query_lens, seq_lens in shapes:
        sizes = indexer.balanced_prefill_row_shard(
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(query_lens, dtype=torch.int32),
            ratio,
            tp_size,
            n,
        )
        assert sizes is not None and len(sizes) == tp_size
        assert sum(sizes) == sum(query_lens) and min(sizes) >= 1
    lens = torch.tensor([n - 1], dtype=torch.int32)
    assert indexer.balanced_prefill_row_shard(lens, lens, ratio, tp_size, n) is None
    assert indexer.balanced_prefill_row_shard(lens, lens, ratio, 1, n) is None


def test_row_shard_threshold_is_total_not_per_rank() -> None:
    """The activation floor counts TOTAL rows regardless of tp_size: a
    per-rank floor would need 128K rows at TP8, i.e. an unschedulable
    --max-num-batched-tokens of 128K on 80 GB Hopper nodes."""
    lens = torch.tensor([MIN_ROWS], dtype=torch.int32)
    for tp_size in (2, 4, 8):
        sizes = indexer.balanced_prefill_row_shard(lens, lens, 4, tp_size, MIN_ROWS)
        assert sizes is not None


def test_tp2_old_per_rank_window_stays_replicated() -> None:
    """32K-64K total rows engaged on TP2 under the old per-rank floor
    (16K rows/rank); the total floor now keeps them replicated — the
    all_gatherv latency term is largest at small tp."""
    window = torch.tensor([3 * MIN_ROWS // 4], dtype=torch.int32)
    assert indexer.balanced_prefill_row_shard(window, window, 4, 2, MIN_ROWS) is None


def test_row_shard_floor_is_at_least_tp_size() -> None:
    """A floor below tp_size still leaves every rank at least one row."""
    lens = torch.tensor([8], dtype=torch.int32)
    assert indexer.balanced_prefill_row_shard(lens, lens, 4, 8, 1) is not None
    assert indexer.balanced_prefill_row_shard(lens - 1, lens - 1, 4, 8, 1) is None


def test_row_sharding_gate_rejects_unsupported_configurations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is the whole safety envelope; nothing else guards the exchange."""
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)

    def supported(**kwargs):
        mode = kwargs.pop("cudagraph_mode", CUDAGraphMode.PIECEWISE)
        cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))
        args = {"dcp_world_size": 1, "use_pcp": False, "tp_size": 4, **kwargs}
        return indexer.tp_prefill_row_sharding_supported(cfg, **args)

    assert supported()
    for bad in ({"tp_size": 1}, {"dcp_world_size": 2}, {"use_pcp": True}):
        assert not supported(**bad)
    for name in (
        "VLLM_DISABLE_PYNCCL",
        "VLLM_USE_NCCL_SYMM_MEM",
        "VLLM_BATCH_INVARIANT",
    ):
        with monkeypatch.context() as m:
            m.setattr(indexer.envs, name, True)
            assert not supported()
    # FULL captures mixed batches whole, so the exchange must stay out.
    assert not supported(cudagraph_mode=CUDAGraphMode.FULL)
    assert supported(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)


def test_profiling_run_reserves_exchange_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dummy run returns before the exchange executes, so its all_gatherv
    output plus staging copy must be reserved up front when row sharding is
    active -- and only then."""
    rows, topk = 128, 8
    buffer = torch.empty((rows, topk), dtype=torch.int32)
    peak = 2 * rows * topk * buffer.element_size()
    specs = (((1,), torch.uint8), ((1,), torch.uint8))
    numels: list[int] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        numels.append(out.numel())
        return out

    def run_dummy(tp_world: int) -> None:
        numels.clear()
        with monkeypatch.context() as m:
            m.setattr(torch, "empty", recording_empty)
            m.setattr(
                sparse_indexer,
                "get_forward_context",
                lambda: SimpleNamespace(attn_metadata=None),
            )
            m.setattr(sparse_indexer, "_gather_workspace_shapes", lambda *a: specs)
            m.setattr(
                sparse_indexer,
                "current_workspace_manager",
                lambda: SimpleNamespace(get_simultaneous=lambda *s: None),
            )
            m.setattr(
                sparse_indexer,
                "get_tp_group",
                lambda: SimpleNamespace(world_size=tp_world),
            )
            m.setattr(
                sparse_indexer,
                "get_current_vllm_config_or_none",
                lambda: SimpleNamespace(
                    compilation_config=SimpleNamespace(
                        cudagraph_mode=CUDAGraphMode.PIECEWISE
                    ),
                    parallel_config=SimpleNamespace(
                        decode_context_parallel_size=1,
                        prefill_context_parallel_size=1,
                    ),
                    scheduler_config=SimpleNamespace(
                        max_num_seqs=4, max_num_batched_tokens=512
                    ),
                    num_speculative_tokens=0,
                ),
            )
            m.setattr(sparse_indexer.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
            m.setattr(indexer.current_platform, "is_cuda", lambda: True)
            sparse_indexer.sparse_attn_indexer(
                hidden_states=torch.empty(4, 4),
                k_cache_prefix="k",
                kv_cache=torch.empty(4, 4),
                q_quant=torch.empty(4, 4),
                q_scale=None,
                k=None,
                weights=torch.empty(4),
                quant_block_size=128,
                scale_fmt=None,
                topk_tokens=topk,
                head_dim=64,
                max_model_len=1024,
                total_seq_lens=512,
                topk_indices_buffer=buffer,
                skip_k_cache_insert=True,
                use_pcp=False,
                pcp_shard_decode_requests=False,
                dense_mha_metadata_layer_name="",
            )

    run_dummy(tp_world=2)
    assert peak in numels
    run_dummy(tp_world=1)
    assert peak not in numels


# V4.1 candidate indexers (VLLM_DSV41_INDEXER_SHARD_CANDIDATE_ROWS). Two prefill
# chunks after any decode rows: a fresh 300-token prompt, then 200 rows over a
# 200-token cached prefix; rank 1 straddles the chunk boundary.
CHUNKS = ((0, 300, 0), (300, 500, 200))
NUM_ROWS = CHUNKS[-1][1]
MAX_DECODE = 3
SIZES = [170, 130, 200]
HEADS, HEAD_DIM, TOPK, CAND_TOPK, CAND_BLOCK = 4, 32, 32, 6, 8


def _prefill_chunk(start: int, end: int, prefix: int, decode: int, device: str):
    """Row i sees the chunk's K positions [0, prefix + i + 1), row-local."""
    n, keys = end - start, prefix + end - start
    return SimpleNamespace(
        token_start=decode + start,
        token_end=decode + end,
        cu_seqlen_ks=torch.zeros(n, dtype=torch.int32, device=device),
        cu_seqlen_ke=torch.arange(prefix + 1, keys + 1, dtype=torch.int32).to(device),
        local_cu_seq_lens=torch.tensor([0, keys], dtype=torch.int32, device=device),
        block_table=None,
        local_total_seq_lens=keys,
        max_local_total_seq_lens=keys,
        max_seqlen_k=keys,
        skip_kv_gather=True,
        pcp_deinterleave_idx=None,
    )


def _ref_mqa_logits(q, kv, weights, ks, ke, clean_logits=True, max_seqlen_k=None):
    scores = torch.einsum("mhd,nd->mhn", q.float(), kv[0][:max_seqlen_k].float())
    return (scores.relu() * weights[..., None]).sum(1)


def _ref_select_candidate_blocks(logits, row_ks, row_ke, topk_blocks, block_size, out):
    rows, width = logits.shape
    nblocks = -(-width // block_size)
    padded = logits.new_full((rows, nblocks * block_size), float("-inf"))
    padded[:, :width] = logits
    cols = torch.arange(padded.shape[1], device=logits.device)
    padded[cols[None] >= row_ke[:, None]] = float("-inf")
    scores = padded.view(rows, nblocks, block_size).amax(-1)
    # Pin each row's newest block, as the kernel does.
    scores[torch.arange(rows), (row_ke.long() - 1) // block_size] = float("inf")
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    out.fill_(-1)
    out[:, : top.indices.shape[1]] = torch.where(
        top.values == float("-inf"), -1, top.indices
    ).int()


def _ref_apply_candidate_mask(logits, row_ks, row_ke, candidates, block_size):
    cols = torch.arange(logits.shape[1], device=logits.device)
    keep = (cols[None, :, None] // block_size == candidates[:, None, :]).any(-1)
    logits.masked_fill_(~(keep & (cols[None] < row_ke[:, None])), float("-inf"))


def _ref_top_k_per_row_prefill(logits, starts, ends, out, num_rows, s0, s1, k):
    for r in range(num_rows):
        top = logits[r, starts[r] : ends[r]].topk(min(k, int(ends[r] - starts[r])))
        out[r].fill_(-1)
        out[r, : top.indices.numel()] = top.indices.int()


class _SequentialTP:
    """all_gatherv over per-rank runs made one after another: a first pass
    records each rank's slices and returns zeros, a second pass hands every
    rank the concatenation of the first pass's slices."""

    def __init__(self) -> None:
        self.sent: dict[int, list[torch.Tensor]] = {}

    def group(self):
        """One rank's group for one run; its calls are numbered in order."""
        call = iter(range(1 << 30))

        def all_gatherv(local, dim, sizes):
            slices = self.sent.setdefault(next(call), [])
            if len(slices) < len(sizes):
                slices.append(local.clone())
                return local.new_zeros((sum(sizes), *local.shape[1:]))
            return torch.cat(slices)

        return SimpleNamespace(all_gatherv=all_gatherv)


@pytest.fixture
def candidate_indexer(monkeypatch: pytest.MonkeyPatch, request):
    """Runs one TP rank's prefill step of a candidate indexer on fixed inputs:
    the reference kernels above on CPU, the real ones on CUDA."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    gen = torch.Generator().manual_seed(0)
    keys = max(prefix + end - start for start, end, prefix in CHUNKS)
    q = torch.randn(MAX_DECODE + NUM_ROWS, HEADS, HEAD_DIM, generator=gen)
    k = torch.randn(keys, HEAD_DIM, generator=gen)
    if device == "cuda":
        q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    q, k = q.to(device), k.to(device)
    k_scale = torch.ones(keys, dtype=torch.float32).view(torch.uint8).view(keys, 4)
    k_scale = k_scale.to(device)
    weights = torch.rand(MAX_DECODE + NUM_ROWS, HEADS, generator=gen).to(device)

    m = monkeypatch
    m.setattr(sparse_indexer, "DeepseekV32IndexerMetadata", SimpleNamespace)
    m.setattr(sparse_indexer, "is_deep_gemm_supported", lambda: False)
    m.setattr(sparse_indexer, "mqa_logits_use_mx_scales", lambda: False)
    m.setattr(
        sparse_indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(get_simultaneous=lambda *specs: (k, k_scale)),
    )
    if device == "cpu":
        m.setattr(sparse_indexer, "fp8_mqa_logits_triton", _ref_mqa_logits)
        m.setattr(
            sparse_indexer, "_select_candidate_blocks", _ref_select_candidate_blocks
        )
        m.setattr(sparse_indexer, "_apply_candidate_mask", _ref_apply_candidate_mask)
        m.setattr(
            sparse_indexer.ops, "top_k_per_row_prefill", _ref_top_k_per_row_prefill
        )

    def run(
        shard: bool,
        candidate_write: bool,
        candidates: torch.Tensor,
        rank=0,
        tp=None,
        decode=0,
    ):
        """``decode`` leading rows count as decode tokens; the decode pass,
        which runs after the prefill rows' exchange, is left out."""
        num_tokens = decode + NUM_ROWS
        metadata = SimpleNamespace(
            slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            num_decodes=0,
            num_prefills=len(CHUNKS),
            num_decode_tokens=decode,
            prefill=SimpleNamespace(
                chunks=[_prefill_chunk(*c, decode, device) for c in CHUNKS],
                row_shard_sizes=SIZES,
            ),
            decode=None,
        )
        topk_buffer = torch.full(
            (num_tokens + 4, TOPK), -7, dtype=torch.int32, device=device
        )
        with monkeypatch.context() as c:
            c.setattr(
                sparse_indexer.envs, "VLLM_DSV41_INDEXER_SHARD_CANDIDATE_ROWS", shard
            )
            c.setattr(
                sparse_indexer,
                "get_forward_context",
                lambda: SimpleNamespace(
                    attn_metadata={"k": metadata},
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ),
            )
            c.setattr(sparse_indexer, "get_tensor_model_parallel_rank", lambda: rank)
            group = tp.group() if tp else None
            c.setattr(sparse_indexer, "get_tp_group", lambda: group)
            sparse_indexer.sparse_attn_indexer(
                hidden_states=torch.empty(num_tokens, 1, device=device),
                k_cache_prefix="k",
                kv_cache=torch.empty(0, device=device),
                q_quant=q[:num_tokens],
                q_scale=None,
                k=None,
                weights=weights[:num_tokens],
                quant_block_size=128,
                scale_fmt=None,
                topk_tokens=TOPK,
                head_dim=HEAD_DIM,
                max_model_len=1024,
                total_seq_lens=keys,
                topk_indices_buffer=topk_buffer,
                skip_k_cache_insert=True,
                use_pcp=False,
                pcp_shard_decode_requests=False,
                dense_mha_metadata_layer_name="",
                candidate_blocks=candidates,
                candidate_block_size=CAND_BLOCK,
                candidate_write=candidate_write,
            )
        return topk_buffer, candidates

    def empty_candidates(decode: int) -> torch.Tensor:
        return torch.full(
            (decode + NUM_ROWS + 4, CAND_TOPK), -7, dtype=torch.int32, device=device
        )

    run.empty_candidates = empty_candidates
    return run


@pytest.mark.parametrize("candidate_indexer", ["cpu", "cuda"], indirect=True)
@pytest.mark.parametrize("candidate_write", [True, False])
@pytest.mark.parametrize("decode", [0, MAX_DECODE])
def test_sharded_candidate_rows_match_replicated(
    candidate_indexer, candidate_write: bool, decode: int
) -> None:
    """Sharding the candidate source's rows (candidate_write) or a masked
    indexer's leaves the top-k and candidate buffers exactly as replicated
    scoring does, on every rank, for every real row; the exchanges leave a
    mixed step's leading decode rows alone."""
    run = candidate_indexer
    rows = decode + NUM_ROWS
    # Masked indexers read the candidate source's replicated output.
    _, source = run(False, True, run.empty_candidates(decode), decode=decode)
    prefill_source = source[decode:rows]
    assert (prefill_source >= 0).any() and (prefill_source == -1).any()
    if candidate_write:
        candidates = run.empty_candidates(decode)
    else:
        candidates = source.clone()
    # The flag off keeps the rows replicated: no exchange (get_tp_group=None).
    ref_topk, ref_candidates = run(
        False, candidate_write, candidates.clone(), decode=decode
    )
    assert (ref_topk[decode:rows] >= 0).all(dim=1).any()

    tp = _SequentialTP()
    for _ in range(2):
        outputs = [
            run(True, candidate_write, candidates.clone(), rank, tp, decode)
            for rank in range(len(SIZES))
        ]
    # One exchange for the top-k, one more for the source's candidate blocks.
    assert len(tp.sent) == 1 + candidate_write
    # The CUDA top-k writes a row's indices in no fixed order (two replicated
    # runs differ too), so compare each row's set.
    ref_sorted = ref_topk[:rows].sort(dim=1).values
    for topk, cands in outputs:
        torch.testing.assert_close(
            topk[:rows].sort(dim=1).values, ref_sorted, rtol=0, atol=0
        )
        torch.testing.assert_close(cands[:rows], ref_candidates[:rows], rtol=0, atol=0)
        # Decode rows: the top-k clear's -1 and the candidates' sentinel.
        assert (topk[:decode] == -1).all() and (cands[:decode] == -7).all()


@pytest.mark.parametrize(
    ("supported", "min_rows", "inert"),
    [(False, 1024, True), (True, 65536, True), (True, 1024, False)],
)
def test_candidate_row_shard_flag_says_when_inert(
    monkeypatch: pytest.MonkeyPatch, supported: bool, min_rows: int, inert: bool
) -> None:
    """The flag only joins the existing row shard, so it says so when that
    cannot run or no 4096-token step reaches its threshold."""
    logged: list[str] = []
    monkeypatch.setattr(
        sparse_indexer, "tp_prefill_row_sharding_supported", lambda *a: supported
    )
    monkeypatch.setattr(
        sparse_indexer.logger,
        "info_once",
        lambda msg, *args: logged.append(msg % args),
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=8,
        ),
        attention_config=SimpleNamespace(indexer_tp_shard_min_rows=min_rows),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )
    sparse_indexer._log_candidate_row_shard_inert(vllm_config)
    assert bool(logged) == inert
