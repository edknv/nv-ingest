# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming-friendly accessors for graph-pipeline ingest output.

``RayDataExecutor.ingest()`` used to call ``ds.to_pandas()`` and return a
single pandas DataFrame to the driver; the driver code in
``pipeline/__main__.py`` then converted that DataFrame to a list-of-dicts.
For a 767-PDF corpus that pulled ~60 GiB of embedded rows onto the driver
even when nothing downstream needed all rows at once.

:class:`IngestResult` wraps the ingest output and exposes only accessors
that can be served by streaming/per-block reads:

- ``row_count()`` — uses ``Dataset.count()`` or ``len(df.index)``
- ``unique_source_count()`` — uses ``Dataset.unique(col)`` or ``Series.nunique``
- ``iter_records()`` — streams rows one at a time
- ``count_uploadable_vdb_records()`` — streams in batches, sums per-batch
- ``detection_summary()`` — feeds streaming rows into ``compute_detection_summary``
- ``write_parquet_dir(out_dir)`` — Ray Data writes per-block files into the dir;
  pandas-backed mode writes a single file inside the same dir for symmetry.

Three constructors map cleanly to the three ingest run modes:

- ``IngestResult.from_dataset(ds)`` — batch mode (Ray Dataset).
- ``IngestResult.from_dataframe(df)`` — inprocess mode (single pandas DataFrame).
- ``IngestResult.from_service(records)`` — service mode (list of records from SSE,
  carrying an optional ``total_pages`` attribute for the input-unit count).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd


_VDB_RECORD_SCAN_BATCH = 1024


def _streaming_iter_batches(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[pd.DataFrame]:
    buf: list[dict[str, Any]] = []
    for row in rows:
        buf.append(row)
        if len(buf) >= batch_size:
            yield pd.DataFrame(buf)
            buf = []
    if buf:
        yield pd.DataFrame(buf)


@dataclass
class IngestResult:
    """Streaming-friendly handle over an ingest result.

    Construct via the ``from_*`` classmethods rather than calling this directly.
    Exactly one of ``_dataframe``, ``_dataset``, ``_service_records`` is set.
    """

    _dataframe: Optional[pd.DataFrame] = None
    _dataset: Optional[Any] = None
    _service_records: Optional[list[dict[str, Any]]] = None
    _service_total_pages: Optional[int] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "IngestResult":
        return cls(_dataframe=df)

    @classmethod
    def from_dataset(cls, dataset: Any) -> "IngestResult":
        return cls(_dataset=dataset)

    @classmethod
    def from_service(cls, records: Iterable[dict[str, Any]]) -> "IngestResult":
        materialized = list(records)
        total_pages = getattr(records, "total_pages", None)
        return cls(_service_records=materialized, _service_total_pages=total_pages)

    # ------------------------------------------------------------------ counts

    def row_count(self) -> int:
        if self._dataset is not None:
            return int(self._dataset.count())
        if self._dataframe is not None:
            return int(len(self._dataframe.index))
        assert self._service_records is not None
        return len(self._service_records)

    def unique_source_count(self) -> int:
        """Number of distinct input units (used by run-summary reporting)."""
        if self._service_records is not None:
            # SSE results don't carry source-side columns; the gateway reports
            # ``total_pages`` directly. Fall back to len(records) when absent.
            return int(self._service_total_pages or len(self._service_records))
        if self._dataset is not None:
            for col in ("source_id", "source_path"):
                try:
                    return len(self._dataset.unique(col))
                except (KeyError, ValueError):
                    continue
            return int(self._dataset.count())
        assert self._dataframe is not None
        df = self._dataframe
        if "source_id" in df.columns:
            return int(df["source_id"].nunique())
        if "source_path" in df.columns:
            return int(df["source_path"].nunique())
        return int(len(df.index))

    # ----------------------------------------------------------------- streams

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if self._dataset is not None:
            yield from self._dataset.iter_rows()
            return
        if self._dataframe is not None:
            yield from self._dataframe.to_dict("records")
            return
        assert self._service_records is not None
        yield from self._service_records

    def _iter_record_batches(self) -> Iterator[list[dict[str, Any]]]:
        if self._dataset is not None:
            for df in self._dataset.iter_batches(batch_format="pandas", batch_size=_VDB_RECORD_SCAN_BATCH):
                yield df.to_dict("records")
            return
        if self._dataframe is not None:
            yield self._dataframe.to_dict("records")
            return
        assert self._service_records is not None
        yield self._service_records

    # ------------------------------------------------------- derived accessors

    def count_uploadable_vdb_records(self) -> int:
        """Streaming version of ``_count_uploadable_vdb_records``.

        Fast path: ``IngestVdbOperator`` emits a per-row ``_vdb_uploadable``
        boolean flag and projects away the heavy embedding/metadata payload.
        When that column is present (the batch-mode path) we just sum it,
        which keeps plasma + driver memory bounded.

        Slow path (inprocess / service mode, or any caller using a raw
        DataFrame without the flag): walk the rows and re-derive uploadable
        status via ``to_client_vdb_records``.
        """
        from nemo_retriever.vdb.records import to_client_vdb_records

        total = 0
        # Iterate as pandas batches so the ``_vdb_uploadable`` column check
        # is a single Series sum per batch.
        if self._dataset is not None:
            batches = self._dataset.iter_batches(batch_format="pandas", batch_size=_VDB_RECORD_SCAN_BATCH)
        elif self._dataframe is not None:
            batches = iter([self._dataframe])
        else:
            assert self._service_records is not None
            batches = iter([pd.DataFrame(self._service_records)])

        for batch_df in batches:
            if "_vdb_uploadable" in batch_df.columns:
                total += int(batch_df["_vdb_uploadable"].sum())
            else:
                total += sum(len(group) for group in to_client_vdb_records(batch_df.to_dict("records")))
        return total

    def detection_summary(self) -> dict[str, Any]:
        from nemo_retriever.utils.detection_summary import (
            compute_detection_summary,
            iter_dataframe_rows,
        )

        # compute_detection_summary wants (page_key, meta, row_dict) tuples.
        # For DataFrame backing reuse the existing helper (it handles JSON-
        # string metadata); for dataset/service stream the same shape from
        # individual row dicts.
        if self._dataframe is not None:
            return compute_detection_summary(iter_dataframe_rows(self._dataframe))

        def _streaming_rows() -> Iterator[tuple[Any, dict[str, Any], dict[str, Any]]]:
            for row in self.iter_records():
                path = str(row.get("path") or row.get("source_id") or "")
                try:
                    page_number = int(row.get("page_number", -1))
                except (TypeError, ValueError):
                    page_number = -1
                meta = row.get("metadata")
                if isinstance(meta, str):
                    try:
                        import json

                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                yield (path, page_number), meta, row

        return compute_detection_summary(_streaming_rows())

    def write_parquet_dir(self, out_dir: Path | str) -> Path:
        """Write the result as a directory of per-block parquet files.

        Returns the output directory. Dataset-backed results use
        ``Dataset.write_parquet``; pandas-backed results write a single
        ``part-00000.parquet`` file inside ``out_dir`` so the on-disk shape
        is the same regardless of run mode.
        """
        out_path = Path(out_dir).expanduser().resolve()
        if self._dataset is not None:
            self._dataset.write_parquet(str(out_path))
            return out_path

        out_path.mkdir(parents=True, exist_ok=True)
        df = self._dataframe if self._dataframe is not None else pd.DataFrame(self._service_records or [])
        df.to_parquet(out_path / "part-00000.parquet", index=False)
        return out_path
