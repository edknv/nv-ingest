# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset entry / run-config loaders for skill_eval."""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from nemo_retriever.harness.config import _read_yaml_mapping


class GroundTruthPage(BaseModel):
    doc_id: str
    page_number: int
    score: int = 1


class DatasetEntry(BaseModel):
    entry_id: int
    query_id: int
    taxonomy_slot_id: str
    original_query: str
    paraphrased_prompt: str
    ground_truth_pages: list[GroundTruthPage]
    ground_truth_answer: str = ""


def load_dataset(path: Path | None = None) -> list[DatasetEntry]:
    if path is None:
        path = Path(str(pkg_files("nemo_retriever.skill_eval").joinpath("data/dataset.yaml")))
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"dataset.yaml must be a list, got {type(raw).__name__}")
    return [DatasetEntry.model_validate(item) for item in raw]


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(str(pkg_files("nemo_retriever.skill_eval").joinpath("configs/run.yaml")))
    return _read_yaml_mapping(Path(path))
