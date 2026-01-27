# app/models/controller/constrained.py
from typing import Iterable, Iterator

from app.models.controller.base import GenerationController
from app.models.runtime.generation_state import GenerationState


class ConstrainedGenerationController(GenerationController):
    """
    Controller that enforces constraints during generation.
    """

    def __init__(self, constraints: Iterable[str]):
        self.constraints = set(constraints)

    def control(
        self,
        states: Iterable[GenerationState],
    ) -> Iterator[GenerationState]:
        for state in states:
            # Simple constraint enforcement logic (placeholder)
            generated_text = state.get_generated_text()
            if any(constraint in generated_text for constraint in self.constraints):
                # If any constraint is violated, stop generation
                state.stop_generation = True
            yield state

    def run():
        """
        Will be implemented in future.
        """
        pass