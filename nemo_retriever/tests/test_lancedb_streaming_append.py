# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming round-trip for ``LanceDB.append`` + ``LanceDB.build_index``.

The end-to-end ingest+retrieve integration test covers the full pipeline; this
single direct test pins down the per-method contract used by
``IngestVdbOperator``/``VdbBuildIndexOperator`` so a regression on either method
surfaces fast without a multi-minute pipeline run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb

from nemo_retriever.vdb.lancedb import LanceDB


VECTOR_DIM = 8


def _records(*texts: str) -> list[list[dict[str, Any]]]:
    inner = [
        {
            "document_type": "text",
            "metadata": {
                "embedding": [0.1 + 0.01 * i] * VECTOR_DIM,
                "content": text,
                "content_metadata": {"type": "text", "page_number": i + 1},
                "source_metadata": {"source_id": f"/tmp/doc-{i}.pdf"},
            },
        }
        for i, text in enumerate(texts)
    ]
    return [inner]


def test_streaming_append_then_build_index_round_trip(tmp_path: Path) -> None:
    """First append(overwrite=True) creates the table; subsequent
    append(overwrite=False) calls add rows; build_index() then builds the
    vector index on the populated table."""
    vdb = LanceDB(uri=str(tmp_path / "db"), table_name="stream", vector_dim=VECTOR_DIM, hybrid=False)

    vdb.append(_records("a", "b"), overwrite=True)
    vdb.append(_records("c", "d", "e"), overwrite=False)
    vdb.build_index()

    table = lancedb.connect(uri=vdb.uri).open_table(vdb.table_name)
    assert table.count_rows() == 5
    indexed_columns = {idx.columns[0] if idx.columns else "" for idx in table.list_indices()}
    assert "vector" in indexed_columns
