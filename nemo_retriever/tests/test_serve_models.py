# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.server
import threading

from nemo_retriever.adapters.cli import serve_models as sm


def test_build_vllm_argv_uses_pooling_runner_and_trust_remote_code() -> None:
    argv = sm.build_vllm_argv("my/embed", "127.0.0.1", 8081)
    # argv[0] is the resolved vllm path; the rest is the spike-confirmed invocation.
    assert argv[1:] == [
        "serve", "my/embed",
        "--runner", "pooling",
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", "8081",
    ]


def test_urls_and_export_line() -> None:
    assert sm.embeddings_url("127.0.0.1", 8081) == "http://127.0.0.1:8081/v1/embeddings"
    assert sm.export_line("127.0.0.1", 8081) == "export EMBED_INVOKE_URL=http://127.0.0.1:8081/v1/embeddings"
    assert "--embed-model-name my/embed" in sm.usage_hint("my/embed")


def test_wait_ready_true_against_fake_health_server() -> None:
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        assert sm.wait_ready("127.0.0.1", port, timeout=5, interval=0.2) is True
    finally:
        srv.shutdown()


def test_wait_ready_false_when_nothing_listening() -> None:
    assert sm.wait_ready("127.0.0.1", 1, timeout=1.0, interval=0.2) is False


def test_terminate_group_reaps_spawned_process() -> None:
    proc = sm.spawn(["sleep", "100"])
    try:
        assert proc.poll() is None  # running
        sm.terminate_group(proc, timeout=5)
        assert proc.poll() is not None  # reaped
    finally:
        if proc.poll() is None:
            proc.kill()
