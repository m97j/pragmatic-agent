# app/models/runtime/generation_state.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class GenerationState:
    """
    GenerationState defines the canonical mutable state of an
    autoregressive language model during inference.

    This state is shared across:
    - runtime (model execution)
    - decoding policies (token selection)
    - reasoning/planning policies (state transformation)
    """
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    past_key_values: Optional[Tuple]

    step: int = 0
    finished: bool = False

    # optional / advanced
    scores: Optional[float | torch.Tensor] = None
    metadata: dict | None = None

    def clone(self):
        return GenerationState(
            input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            past_key_values=self.past_key_values,
            step=self.step,
            finished=self.finished,
            scores=self.scores,
            metadata=dict(self.metadata) if self.metadata else None,
        )

