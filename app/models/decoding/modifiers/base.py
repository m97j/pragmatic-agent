# app/models/decoding/modifiers/base.py
from abc import ABC, abstractmethod

import torch

from app.models.runtime.generation_state import GenerationState


class LogitsModifier(ABC):
    """
    Interface for logits-level transformations.

    Design principles:
    - MUST be side-effect free
    - MUST NOT mutate GenerationState
    - Operates only on logits
    """
    temperature: float | None = None
    cfg_scale: float | None = None
    alpha: float | None = None

    def apply_overrides(self, **kwargs):
        """
        Apply any modifier-specific overrides before execution.
        """
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "cfg_scale" in kwargs:
            self.cfg_scale = kwargs["cfg_scale"]
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]

    @abstractmethod
    def apply(
        self,
        logits: torch.Tensor,
        state: GenerationState,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, vocab]
            state: current GenerationState (read-only)

        Returns:
            modified logits
        """
        raise NotImplementedError
