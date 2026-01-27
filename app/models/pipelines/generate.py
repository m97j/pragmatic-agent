# app/models/pipelines/generate.py
from typing import Iterable, Union

from app.models.decoding.engine import DecodingEngine
from app.models.decoding.modifiers.temperature import TemperatureModifier
from app.models.decoding.policies.sampling import SamplingPolicy
from app.models.pipelines.base import BasePipeline
from app.models.prompts.mode import apply_mode_prefix
from app.models.runtime.generation_state import GenerationState
from app.models.runtime.llm_runtime import LLMRuntime


class GeneratePipeline(BasePipeline):
    """
    Generic text generation pipeline.

    Characteristics:
    - prompt engineering only (no reasoning/controller)
    - sampling-based decoding
    - supports streaming
    """

    name = "generate"

    def __init__(self, runtime: LLMRuntime, decoding: DecodingEngine = None, controller=None):

        if decoding is None:
            policy = SamplingPolicy(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )

            modifiers = [
                TemperatureModifier(temperature=1.0)
            ]

        decoding = decoding or DecodingEngine(
            runtime=runtime,
            decoding_policy=policy,
            modifiers=modifiers,
        )

        super().__init__(
            name=self.name,
            runtime=runtime,
            decoding=decoding,
            controller=controller,
        )

    # -------------------------
    # Public API
    # -------------------------
    def run(
        self,
        prompt: str,
        *,
        mode: str | None = "instruct",
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[GenerationState, Iterable[GenerationState]]:
        """
        Execute generation.

        Args:
            prompt: raw text prompt
            mode: prompt mode ("instruct", "think", None)
            max_tokens: optional override
            stream: whether to stream output
        """
        self._apply_overrides(**kwargs)
        
        # 1️ apply mode prefix (string level)
        prompt = apply_mode_prefix(prompt, mode)

        # 2️ tokenize
        encoding = self.runtime.tokenize(prompt)
        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask")

        # 3️ override max_tokens if provided
        if max_tokens is not None:
            self.decoding.policy.max_tokens = max_tokens

        # 4️ run decoding
        result = self.decoding.run(
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            max_tokens=self.decoding.policy.max_tokens,
            stream=stream,
        )

        # 5 return result
        return result

