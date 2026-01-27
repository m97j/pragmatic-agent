# app/models/decoding/policies/sampling.py
import torch

from app.models.decoding.policies.base import DecodingPolicy
from app.models.runtime.decoding_results import DecodingResults
from app.models.runtime.generation_state import GenerationState


class SamplingPolicy(DecodingPolicy):
    """
    Nucleus (top-p) sampling policy.
    """

    def __init__(
        self,
        top_p: float = 0.9,
        max_tokens: int = 128,
        stop_id: int | None = None,
    ):
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_id = stop_id

    def select(
        self,
        logits: torch.Tensor,
        state: GenerationState,
    ) -> DecodingResults:
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative <= self.top_p
        mask[..., 0] = True  # always keep at least one token

        filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        idx = torch.multinomial(filtered_probs, num_samples=1)
        token = sorted_indices.gather(-1, idx).squeeze(-1)

        return DecodingResults(token_id=token)

