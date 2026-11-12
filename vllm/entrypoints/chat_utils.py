# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility path for callers built against vLLM 0.27 (e.g. Ray Serve LLM 2.59)."""

from vllm.renderers.chat_utils.chat_message import *  # noqa: F401,F403
from vllm.renderers.chat_utils.chat_message import parse_chat_messages_async  # noqa: F401
from vllm.renderers.chat_utils.chat_template import *  # noqa: F401,F403
from vllm.renderers.chat_utils.chat_template import load_chat_template  # noqa: F401
