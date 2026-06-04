# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import lancedb

import nemo_retriever.adapters.cli.sdk_workflow as sdk_workflow


def _make_table(uri: str, rows: list[dict]) -> None:
    lancedb.connect(uri).create_table("t", data=rows)


def test_verify_returns_independent_text_evidence_and_term_overlap(tmp_path) -> None:
    uri = str(tmp_path / "lancedb")
    _make_table(uri, [
        {"vector": [0.0, 0.0], "text": "Premium desk fan $150 (chart caption)",
         "metadata": json.dumps({"page_number": 1, "type": "chart"}),
         "source": json.dumps({"source_id": "/x/doc.pdf", "source_name": "doc.pdf"})},
        {"vector": [0.0, 0.0], "text": "The premium desk fan is priced at $150 per the prose.",
         "metadata": json.dumps({"page_number": 1, "type": "text"}),
         "source": json.dumps({"source_id": "/x/doc.pdf", "source_name": "doc.pdf"})},
    ])

    out = sdk_workflow.verify_claim(
        "Premium desk fan costs $150", "doc", page=1, lancedb_uri=uri, table_name="t"
    )

    assert out["independent_evidence_found"] is True
    # chart chunk excluded; only the text chunk is independent evidence
    assert [e["modality"] for e in out["evidence"]] == ["text"]
    assert "150" in out["matched_terms"]
    assert "desk" in out["matched_terms"]


def test_verify_reports_no_independent_evidence_when_only_chart(tmp_path) -> None:
    uri = str(tmp_path / "lancedb")
    _make_table(uri, [
        {"vector": [0.0, 0.0], "text": "chart only",
         "metadata": json.dumps({"page_number": 1, "type": "chart"}),
         "source": json.dumps({"source_name": "doc.pdf"})},
    ])

    out = sdk_workflow.verify_claim("anything 999", "doc", page=1, lancedb_uri=uri, table_name="t")

    assert out["independent_evidence_found"] is False
    assert out["evidence"] == []
    assert out["unmatched_terms"] == ["999", "anything"]


def test_root_verify_cli_prints_json(monkeypatch) -> None:
    import importlib

    from typer.testing import CliRunner

    cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")
    captured: dict = {}

    def fake_verify(claim, source, **kwargs):
        captured["call"] = (claim, source, kwargs)
        return {
            "claim": claim, "source": source, "page": kwargs.get("page"),
            "evidence": [], "independent_evidence_found": False,
            "matched_terms": [], "unmatched_terms": [],
        }

    # verify_command calls the name imported INTO main.py, so patch it there.
    monkeypatch.setattr(cli_main, "verify_claim", fake_verify)

    result = CliRunner().invoke(
        cli_main.app,
        ["verify", "Premium desk fan costs $150", "--source", "doc", "--page", "1",
         "--lancedb-uri", "/tmp/lancedb", "--table-name", "docs"],
    )

    assert result.exit_code == 0
    out = json.loads(result.output)
    assert out["claim"] == "Premium desk fan costs $150"
    assert out["source"] == "doc"
    assert captured["call"][2]["page"] == 1
    assert captured["call"][2]["against"] == "text,table"
    assert captured["call"][2]["table_name"] == "docs"
