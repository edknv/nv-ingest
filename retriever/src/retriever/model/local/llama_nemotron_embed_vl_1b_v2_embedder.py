from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image


# Regex for the data-URI image format used by nv-ingest:
#   data:image/<subtype>;base64,<payload>
_DATA_URI_RE = re.compile(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)")


def _decode_image(data_uri: str) -> Image.Image:
    """Decode a ``data:image/...;base64,...`` string into a PIL Image."""
    m = _DATA_URI_RE.search(data_uri)
    if m is None:
        raise ValueError("String does not contain a valid data:image/...;base64,... URI")
    raw = base64.b64decode(m.group(1))
    return Image.open(BytesIO(raw)).convert("RGB")


@dataclass
class LlamaNemotronEmbedVL1BV2Embedder:
    """
    Minimal embedder wrapper for the multimodal
    ``nvidia/llama-nemotron-embed-vl-1b-v2`` model.

    The VL model exposes ``encode_queries()`` and ``encode_documents()``
    instead of the standard tokenizer + forward pass, and supports three
    modalities: *text*, *image*, and *image_text*.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    model_id: Optional[str] = None

    # Populated in __post_init__
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import AutoModel

        model_id = self.model_id or "nvidia/llama-nemotron-embed-vl-1b-v2"
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = self.hf_cache_dir or str(Path.home() / ".cache" / "huggingface")

        # Try flash_attention_2 first, fall back to sdpa/eager if unavailable.
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                self._model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                    cache_dir=hf_cache_dir,
                )
                break
            except (ValueError, ImportError):
                if attn_impl == "eager":
                    raise
                continue

        self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    @property
    def is_multimodal(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, 2048]``."""
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode():
            out = self._model.encode_queries(texts_list)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().float()
        return torch.as_tensor(out, dtype=torch.float32).cpu()

    def embed_documents(
        self,
        texts: Optional[Sequence[str]] = None,
        images: Optional[Sequence[Image.Image]] = None,
        *,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Embed documents (text-only, image-only, or text+image).

        Returns CPU tensor ``[N, 2048]``.
        """
        kwargs: Dict[str, Any] = {}
        n = 0
        if texts is not None:
            kwargs["texts"] = list(texts)
            n = len(texts)
        if images is not None:
            kwargs["images"] = list(images)
            n = max(n, len(images))
        if not kwargs:
            return torch.empty((0, 2048), dtype=torch.float32)

        with torch.inference_mode():
            out = self._model.encode_documents(**kwargs)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().float()
        return torch.as_tensor(out, dtype=torch.float32).cpu()

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Backward-compatible text-only embedding (document side).

        Returns CPU tensor ``[N, 2048]``.
        """
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        return self.embed_documents(texts=texts_list, batch_size=batch_size)

    def parse_and_embed_mixed(
        self, input_strings: Sequence[str], *, batch_size: int = 16
    ) -> Sequence[Sequence[float]]:
        """Embed a mixed list of text and/or image inputs.

        Each element of *input_strings* is classified as:

        - **image-only** if it matches ``data:image/...;base64,...`` with no
          surrounding text (after stripping).
        - **image+text** if it contains a ``data:image/...;base64,...`` marker
          *and* additional non-whitespace text.
        - **text-only** otherwise.

        Groups are embedded with the appropriate ``encode_documents`` call and
        then reassembled in the original order.

        Returns a list-of-lists of floats (same length as *input_strings*).
        """
        # Classify each input.
        TEXT = 0
        IMAGE = 1
        IMAGE_TEXT = 2

        groups: List[Tuple[int, int, str, Optional[Image.Image]]] = []  # (original_idx, modality, text, image)

        for idx, raw in enumerate(input_strings):
            s = str(raw)
            m = _DATA_URI_RE.search(s)
            if m is None:
                # Pure text
                groups.append((idx, TEXT, s, None))
            else:
                img = _decode_image(s)
                # Strip the data-URI from the string to get any surrounding text.
                text_part = _DATA_URI_RE.sub("", s).strip()
                if text_part:
                    groups.append((idx, IMAGE_TEXT, text_part, img))
                else:
                    groups.append((idx, IMAGE, "", img))

        # Prepare result array.
        results: List[Optional[List[float]]] = [None] * len(input_strings)

        # --- text-only group ---
        text_items = [(orig_idx, txt) for orig_idx, mod, txt, _ in groups if mod == TEXT]
        if text_items:
            idxs, txts = zip(*text_items)
            vecs = self.embed_documents(texts=list(txts), batch_size=batch_size)
            for i, orig_idx in enumerate(idxs):
                results[orig_idx] = vecs[i].tolist()

        # --- image-only group ---
        img_items = [(orig_idx, img) for orig_idx, mod, _, img in groups if mod == IMAGE and img is not None]
        if img_items:
            idxs, imgs = zip(*img_items)
            vecs = self.embed_documents(images=list(imgs), batch_size=batch_size)
            for i, orig_idx in enumerate(idxs):
                results[orig_idx] = vecs[i].tolist()

        # --- image+text group ---
        it_items = [
            (orig_idx, txt, img)
            for orig_idx, mod, txt, img in groups
            if mod == IMAGE_TEXT and img is not None
        ]
        if it_items:
            idxs_it, txts_it, imgs_it = zip(*it_items)
            vecs = self.embed_documents(texts=list(txts_it), images=list(imgs_it), batch_size=batch_size)
            for i, orig_idx in enumerate(idxs_it):
                results[orig_idx] = vecs[i].tolist()

        # Fill any remaining Nones (shouldn't happen, but safety).
        for i in range(len(results)):
            if results[i] is None:
                results[i] = [0.0] * 2048

        return results  # type: ignore[return-value]
