# app/models/outputs/base.py
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List

from app.models.runtime.generation_state import GenerationState
from app.models.runtime.llm_runtime import LLMRuntime


class OutputTransform(ABC):
    """
    Post-processing layer for generation results.

    Supports:
    - detokenization
    - streaming guards
    - format validation
    - candidate filtering
    """

    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime

    # -------- core --------

    def detokenize(self, state: GenerationState) -> str:
        return self.runtime.detokenize(state.input_ids)

    # -------- streaming --------

    @abstractmethod
    def stream(
        self,
        states: Iterable[GenerationState],
    ) -> Iterator[str]:
        pass

    # -------- finalization --------

    @abstractmethod
    def finalize(
        self,
        state: GenerationState,
    ) -> str:
        pass

    # -------- validation (optional) --------

    def validate(self, text: str) -> bool:
        """
        Override for JSON/markdown/schema validation.
        """
        return True

    def filter_valid(
        self,
        states: List[GenerationState],
    ) -> List[GenerationState]:
        """
        Keep only valid formatted outputs.
        """
        valid = []

        for s in states:
            text = self.detokenize(s)
            if self.validate(text):
                valid.append(s)

        return valid
