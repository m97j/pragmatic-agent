# app/models/decoding/policies/greedy.py
import torch

from app.models.decoding.policies.base import DecodingPolicy
from app.models.runtime.decoding_results import DecodingResults
from app.models.runtime.generation_state import GenerationState


class GreedyPolicy(DecodingPolicy):
    """
    Greedy decoding (argmax).
    """

    def __init__(
        self,
        max_tokens: int = 128,
        stop_id: int | None = None,
    ):
        self.max_tokens = max_tokens
        self.stop_id = stop_id

    def select(
        self,
        logits: torch.Tensor,
        state: GenerationState,
    ) -> DecodingResults:
        token = int(torch.argmax(logits, dim=-1).item())
        return DecodingResults(token_id=token)

