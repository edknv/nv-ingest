"""Shared model identifier constants."""

from __future__ import annotations

VL_EMBED_MODEL_IDS: frozenset[str] = frozenset(
    {
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "llama-nemotron-embed-vl-1b-v2",
    }
)


def is_vl_embed_model(model_name: str) -> bool:
    """Return True if *model_name* refers to a VL (vision-language) embedding model."""
    return (model_name or "") in VL_EMBED_MODEL_IDS
