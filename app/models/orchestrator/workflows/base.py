# app/models/orchestrator/workflows/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List

from app.models.orchestrator.strategies.base import ReasoningStrategy
from app.models.outputs.base import OutputTransform
from app.models.pipelines.registry import PipelineRegistry
from app.models.runtime.generation_state import GenerationState
from app.models.runtime.llm_runtime import LLMRuntime


class BaseWorkflow(ABC):
    """
    Orchestrates one or many generation runs.
    
    Owns:
    - generation execution
    - reasoning loop
    - output transform
    """

    def __init__(
        self,
        *,
        runtime: LLMRuntime,
        output_transform: OutputTransform,
        reasoning: ReasoningStrategy | None = None,
        pipeline_registry: PipelineRegistry,
    ):
        self.runtime = runtime
        self.output_transform = output_transform
        self.reasoning = reasoning
        self.registry = pipeline_registry

    @abstractmethod
    def generate_once(self, prompt: str, **kwargs) -> GenerationState:
        """
        Execute a single generation run.
        """
        pass

    def run(self, prompt: str, **kwargs):
        """
        Default orchestration logic (single or multi generation).
        """

        candidates: List[GenerationState] = []

        # basic single or multi-sample loop
        num_samples = kwargs.get("num_samples", 1)

        for _ in range(num_samples):
            state = self.generate_once(prompt, **kwargs)
            candidates.append(state)

        # reasoning phase
        if self.reasoning:
            while self.reasoning.should_rerun(candidates):
                candidates.append(self.generate_once(prompt, **kwargs))

            final_state = self.reasoning.select(candidates)
        else:
            final_state = candidates[0]

        return final_state

    # --------------------
    # Output adapters
    # --------------------

    def stream(self, states: Iterable[GenerationState]) -> Iterable[str]:
        return self.output_transform.stream(states)

    def finalize(self, state: GenerationState) -> str:
        return self.output_transform.finalize(state)
