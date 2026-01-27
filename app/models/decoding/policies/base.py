# app/models/decoding/policies/base.py
from abc import ABC, abstractmethod

import torch

from app.models.runtime.decoding_results import DecodingResults
from app.models.runtime.generation_state import GenerationState


class DecodingPolicy(ABC):
    """
    Token-level decoding policy.

    Responsibilities:
    - Select the next token from logits
    - Decide whether generation should stop

    This policy MUST be stateless or derive state only
    from GenerationState.
    """

    top_p: float | None = None
    max_tokens: int | None = None

    def apply_overrides(self, **kwargs):
        """
        Apply any policy-specific overrides before execution.
        """
        if "top_p" in kwargs:
            self.top_p = kwargs["top_p"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]

    @abstractmethod
    def select(
        self,
        logits: torch.Tensor,
        state: GenerationState,
    ) -> DecodingResults:
        """
        Select the next token id from model logits.

        Args:
            logits: [batch, vocab] tensor
            state: current GenerationState

        Returns:
            DecodingResults
        """
        raise NotImplementedError
