# app/models/orchestrator/strategies/base.py
from abc import ABC, abstractmethod
from typing import List

from app.models.runtime.generation_state import GenerationState


class ReasoningStrategy(ABC):
    """
    Reasoning algorithms operating on completed generations.

    Examples:
    - tree of thoughts
    - self-consistency
    - verifier scoring
    - debate
    """

    @abstractmethod
    def select(
        self,
        candidates: List[GenerationState],
        **kwargs,
    ) -> GenerationState:
        """
        Choose or refine among multiple generation results.
        """
        pass

    def should_rerun(self, candidates: List[GenerationState]) -> bool:
        """
        Optional hook: decide whether more generations are needed.
        """
        return False
