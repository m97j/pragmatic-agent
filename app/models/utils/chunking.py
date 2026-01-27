# app/models/utils/chunking.py
from typing import Callable

import torch

from app.modules.common.utils import split_content


def chunk_text_with_offsets(
    *,
    text: str,
    tokenize_fn: Callable,
    max_tokens: int = 8000,
) -> list[torch.Tensor]:
    """
    Split text into token chunks not exceeding max_tokens,
    respecting sentence boundaries when possible.

    Args:
        text: raw input text
        tokenize_fn: callable(text, return_offsets=True) -> BatchEncoding
        max_tokens: maximum tokens per chunk

    Returns:
        List[torch.Tensor]: list of token ID tensors (1D, CPU)
    """
    max_tokens = min(14000, max_tokens)

    encodings = tokenize_fn(text, return_offsets=True)
    tokens = encodings["input_ids"][0]          # (N,)
    offsets = encodings["offset_mapping"][0]    # (N, 2)

    # sentence boundary positions (character indices)
    sentence_boundaries = set(split_content(text, return_boundaries=True))

    chunks: list[torch.Tensor] = []
    start = 0

    for i, (_, end) in enumerate(offsets):
        if (i - start + 1) >= max_tokens:
            boundary_candidates = [b for b in sentence_boundaries if b <= end]

            if boundary_candidates:
                boundary_index = max(boundary_candidates)
                cutoff = max(
                    j for j, (_, e) in enumerate(offsets[: i + 1])
                    if e <= boundary_index
                )
                chunks.append(tokens[start : cutoff + 1])
                start = cutoff + 1
            else:
                chunks.append(tokens[start : i + 1])
                start = i + 1

    if start < len(tokens):
        chunks.append(tokens[start:])

    return chunks
