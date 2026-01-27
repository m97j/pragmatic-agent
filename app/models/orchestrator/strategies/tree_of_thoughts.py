# app/models/orchestrator/strategies/tree_of_thoughts.py
# not in use yet
from typing import List

import torch

from app.models.orchestrator.strategies.base import ReasoningStrategy
from app.models.runtime.generation_state import GenerationState


class TreeOfThoughts(ReasoningStrategy):
    """
    Tree-of-Thoughts reasoning.

    Generates multiple reasoning branches and selects the best one.
    """

    def __init__(
        self,
        controller,
        num_branches: int = 3,
        max_depth: int = 2,
    ):
        super().__init__(controller)
        self.num_branches = num_branches
        self.max_depth = max_depth

    def run(
        self,
        prompt_ids: torch.Tensor,
        **kwargs,
    ) -> GenerationState:
        states: List[GenerationState] = []

        for _ in range(self.num_branches):
            state = self.controller.run(prompt_ids)
            states.append(state)

        # naive evaluation: longest completion
        best_state = max(
            states,
            key=lambda s: s.input_ids.size(1),
        )

        return best_state
