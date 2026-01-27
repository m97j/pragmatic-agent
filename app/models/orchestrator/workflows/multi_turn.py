# app/models/orchestrator/workflows/multi_turn.py
import torch

from app.models.orchestrator.workflows.base import BaseWorkflow
from app.models.outputs.sentence import SentenceOutput
from app.models.pipelines.factory import build_pipeline_registry


class MultiTurnWorkflow(BaseWorkflow):
    """
    Multi-turn generation workflow.

    Each turn:
    - takes previous output
    - appends to prompt
    - runs decoding pipeline again
    """

    def __init__(
        self,
        *,
        runtime,
        output_transform=None,
        registry=None,
        max_turns: int = 3,
        separator: str = "\n",
    ):
        output_transform = SentenceOutput(runtime)
        super().__init__(
            runtime=runtime,
            output_transform=output_transform,
            reasoning=None,
            pipeline_registry=registry or build_pipeline_registry(runtime),
        )
        self.max_turns = max_turns
        self.separator = separator

    def generate_once(self, prompt, **kwargs):
        pipeline = self.registry.get("summarize")
        state = pipeline.run(
             prompt, 
             mode=kwargs.get("mode"), 
             max_tokens=kwargs.get("max_tokens"), 
             stream=kwargs.get("stream", False), 
             **kwargs,
             )
        return state

    def run(
        self,
        prompt_ids: torch.Tensor,
        **kwargs,
    ):
        current_prompt = prompt_ids

        for turn in range(self.max_turns):
            state = self.generate_once(current_prompt, **kwargs)
            
            if turn < self.max_turns - 1:
                # prepare for next turn
                current_prompt = torch.cat(
                    [
                        current_prompt,
                        torch.tensor(self.separator).unsqueeze(0),
                        state.input_ids,
                    ],
                    dim=-1,
                )

            if state.finished:
                break

        if self.stream:
            return self.output_transform.stream([state])
        else:
            return self.output_transform.finalize(state)
