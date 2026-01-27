# app/models/orchestrator/workflows/think.py
from typing import Iterable, Union

from app.models.decoding.engine import DecodingEngine
from app.models.decoding.modifiers.temperature import TemperatureModifier
from app.models.decoding.policies.sampling import SamplingPolicy
from app.models.orchestrator.strategies.chain_of_thought import ChainOfThought
from app.models.orchestrator.workflows.base import BaseWorkflow
from app.models.outputs.tool_call import ToolCallOutput
from app.models.pipelines.factory import build_pipeline_registry
from app.models.pipelines.generate import GeneratePipeline
from app.models.pipelines.registry import PipelineRegistry
from app.models.runtime.llm_runtime import LLMRuntime


class ThinkWorkflow(BaseWorkflow):
    """
    Thinking / decision-making pipeline.

    Characteristics:
    - reasoning enabled
    - low temperature
    - non-streaming
    """

    name = "think"

    def __init__(
        self, 
        *,
        runtime: LLMRuntime, 
        output_transform=None,
        reasoning=None,
        registry: PipelineRegistry =None
    ):
        policy = SamplingPolicy(
            top_p=0.9,
            max_tokens=256,
        )

        modifiers = [
            TemperatureModifier(
                temperature=0.2
            )
        ]

        decoding = DecodingEngine(
            runtime=runtime,
            decoding_policy=policy,
            modifiers=modifiers,
        )

        pipeline = GeneratePipeline(
            runtime=runtime,
            decoding=decoding,
            controller=None,
        )
        if registry is None:
            registry = build_pipeline_registry(runtime)

        registry.register(name="think.generate", pipeline=pipeline)

        reasoning = ChainOfThought(
            expose_reasoning=False   # 내부 추론만 사용
        )

        output_transform = ToolCallOutput(runtime)

        super().__init__(
            runtime=runtime,
            output_transform=output_transform,
            reasoning=reasoning,
            pipeline_registry=registry,
        )

    def generate_once(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ):
        pipeline = self.registry.get("think.generate")

        state = pipeline.run(
             prompt,
             mode="think",
             max_tokens=max_tokens,
             stream=stream,
             **kwargs,
        )
        return state

    def run(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Iterable[str]]:

        state = self.generate_once(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        return self.output_transform.finalize(state)
