# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch a warm vLLM OpenAI-compatible embeddings server (kills per-query cold-load).

The exact `vllm serve` invocation was pinned by the Objective-3a feasibility spike:
vLLM 0.20 uses `--runner pooling` (not the removed `--task embed`), and the
nemotron embed model requires `--trust-remote-code`.
"""

from __future__ import annotations

import shutil
import time
import urllib.request


def build_vllm_argv(model: str, host: str, port: int) -> list[str]:
    """argv to `vllm serve` an embedding model as an OpenAI /v1/embeddings server."""
    vllm = shutil.which("vllm") or "vllm"
    return [
        vllm, "serve", model,
        "--runner", "pooling",
        "--trust-remote-code",
        "--host", host,
        "--port", str(port),
    ]


def embeddings_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/embeddings"


def export_line(host: str, port: int) -> str:
    return f"export EMBED_INVOKE_URL={embeddings_url(host, port)}"


def usage_hint(model: str) -> str:
    """Reminder that a warm query must also pass the matching model name.

    vLLM returns 404 if the requested model name differs from the served one, so
    callers pass `--embed-model-name <model>` (there is no env fallback for it).
    """
    return f'Then query with the matching model name, e.g.:\n  retriever query "<q>" --embed-model-name {model}'


def wait_ready(host: str, port: int, timeout: float = 600.0, interval: float = 3.0) -> bool:
    """Poll the vLLM /health endpoint until 200 or timeout."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)
    return False
