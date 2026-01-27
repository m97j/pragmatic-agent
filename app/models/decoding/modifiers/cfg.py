# app/models/decoding/modifiers/cfg.py
import torch

from app.models.decoding.modifiers.base import LogitsModifier
from app.models.runtime.generation_state import GenerationState


class CFGModifier(LogitsModifier):
    """
    Classifier-Free Guidance for LLMs.

    logits = logits_cond + scale * (logits_cond - logits_uncond)
    """

    def __init__(
        self,
        scale: float = 1.5,
    ):
        self.cfg_scale = scale

    def apply(
        self,
        logits: torch.Tensor,
        uncond_logits: torch.Tensor,
        state: GenerationState,
    ) -> torch.Tensor:
        """
        uncond_runtime should receive:
        - same input_ids
        - but without conditioning prompt
        (handled outside or via state)
        """
        guided = uncond_logits + self.cfg_scale * (logits - uncond_logits)
        return guided
