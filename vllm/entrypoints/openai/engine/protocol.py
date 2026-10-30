# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility path for callers built against vLLM 0.27 (e.g. Ray Serve LLM 2.59)."""

from vllm.entrypoints.serve.engine.protocol import *  # noqa: F401,F403
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse  # noqa: F401
