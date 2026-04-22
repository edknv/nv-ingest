import ast
import json
from types import SimpleNamespace

from nemo_retriever.vector_store.lancedb_store import (
    LanceDBConfig,
    _build_lancedb_rows_from_df,
    create_lancedb_index,
)
from nemo_retriever.vector_store.lancedb_utils import build_lancedb_row


def test_build_lancedb_row_persists_normalized_content_type() -> None:
    row = SimpleNamespace(
        path="/tmp/doc_a.pdf",
        page_number=7,
        metadata={"embedding": [0.1, 0.2], "source_path": "/tmp/doc_a.pdf"},
        text="table text",
        _content_type="table_caption",
    )

    row_out = build_lancedb_row(row)

    assert row_out is not None
    metadata = json.loads(row_out["metadata"])
    assert metadata["_content_type"] == "table"


def test_build_lancedb_rows_from_df_persists_normalized_content_type() -> None:
    rows = [
        {
            "path": "/tmp/doc_b.pdf",
            "page_number": 3,
            "text": "chart text",
            "_content_type": "chart_caption",
            "metadata": {"embedding": [0.3, 0.4], "source_path": "/tmp/doc_b.pdf"},
        }
    ]

    row_out = _build_lancedb_rows_from_df(rows)

    assert len(row_out) == 1
    metadata = ast.literal_eval(row_out[0]["metadata"])
    assert metadata["_content_type"] == "chart"


def test_create_lancedb_index_caps_num_partitions_for_small_tables():
    """With fewer rows than num_partitions, LanceDB's KMeans fails; cap the K."""

    class _FakeTable:
        def __init__(self, rows: int) -> None:
            self.rows = rows
            self.seen_kwargs: dict = {}
            self.indices: list = []

        def count_rows(self) -> int:
            return self.rows

        def create_index(self, **kwargs):
            self.seen_kwargs = kwargs

        def create_fts_index(self, *args, **kwargs) -> None:
            pass

        def list_indices(self) -> list:
            return []

    cfg = LanceDBConfig(uri="ignored", table_name="t", num_partitions=16, num_sub_vectors=256)

    # Small corpus: cap K down to row count.
    small = _FakeTable(rows=11)
    create_lancedb_index(small, cfg=cfg)
    assert small.seen_kwargs["num_partitions"] == 11

    # Single row: K must still be ≥ 1.
    singleton = _FakeTable(rows=1)
    create_lancedb_index(singleton, cfg=cfg)
    assert singleton.seen_kwargs["num_partitions"] == 1

    # Large corpus: honour the requested K unchanged.
    large = _FakeTable(rows=1000)
    create_lancedb_index(large, cfg=cfg)
    assert large.seen_kwargs["num_partitions"] == 16
