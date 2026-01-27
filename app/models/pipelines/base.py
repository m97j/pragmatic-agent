# app/models/pipelines/base.py
from abc import ABC, abstractmethod
from typing import Iterable, Union

from app.models.controller.base import GenerationController
from app.models.decoding.engine import DecodingEngine
from app.models.runtime.generation_state import GenerationState
from app.models.runtime.llm_runtime import LLMRuntime


class BasePipeline(ABC):
    """
    Base class for all LLM pipelines.

    A pipeline represents a preset composition of:
    - reasoning
    - decoding
    - controller
    executed via runtime.
    """

    def __init__(
        self,
        name: str,
        runtime: LLMRuntime,
        decoding: DecodingEngine,
        controller: GenerationController = None,
    ):
        self.name = name if name else self.__class__.name
        self.runtime = runtime
        self.decoding = decoding
        self.controller = controller

    @abstractmethod
    def run(
        self,
        prompt: str,
        *,
        mode: str | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[GenerationState, Iterable[GenerationState]]:
        """
        Execute pipeline.

        Returns:
            - str            : non-streaming output
            - Iterable[str]  : streaming output
        """
        raise NotImplementedError
    
    def _apply_overrides(self, **kwargs):
        """
        Apply any runtime/decoding/controller overrides before execution.
        """
        if self.decoding:
            self.decoding.apply_overrides(**kwargs)

        if self.controller:
            self.controller.apply_overrides(**kwargs)