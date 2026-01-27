# app/models/orchestrator/workflows/self_consistency.py
# not used yet
from collections import Counter
from typing import List

import torch

from app.models.controller.base import BaseWorkflow
from app.models.decoding.engine import DecodingEngine
from app.models.runtime.generation_state import GenerationState


class SelfConsistencyWorkflow(BaseWorkflow):
    """
    Self-consistency decoding controller.

    Runs the same pipeline multiple times and selects
    the most frequent output.
    """

    def __init__(
        self,
        pipeline: DecodingEngine,
        num_samples: int = 5,
    ):
        super().__init__(pipeline)
        self.num_samples = num_samples

    def run(
        self,
        prompt_ids: torch.Tensor,
        **kwargs,
    ) -> GenerationState:
        states: List[GenerationState] = []

        for _ in range(self.num_samples):
            state = self.pipelines[0].generate(prompt_ids)
            states.append(state)

        # voting by decoded token sequence
        token_sequences = [
            tuple(state.input_ids.squeeze().tolist())
            for state in states
        ]

        most_common_tokens, _ = Counter(token_sequences).most_common(1)[0]

        for state in states:
            if tuple(state.input_ids.squeeze().tolist()) == most_common_tokens:
                return state

        return states[0]  # fallback
