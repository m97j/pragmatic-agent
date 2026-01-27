# app/models/decoding/modifiers/contrastive.py
import torch

from app.models.decoding.modifiers.base import LogitsModifier
from app.models.runtime.generation_state import GenerationState


class ContrastiveModifier(LogitsModifier):
    """
    Contrastive Decoding:
        logits = strong_logits - alpha * weak_logits
    """

    def __init__(
        self,
        alpha: float = 0.5,
    ):
        self.alpha = alpha

    def apply(
        self,
        logits: torch.Tensor,
        weak_logits: torch.Tensor,
        state: GenerationState,
    ) -> torch.Tensor:
        """
        Note:
        - weak model does NOT use KV cache
        - weak model always runs full forward on current input
        """
        return logits - self.alpha * weak_logits
