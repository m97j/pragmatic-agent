# app/models/orchestrator/workflows/refine.py
import torch

from app.models.orchestrator.workflows.base import BaseWorkflow
from app.models.outputs.text import PlainTextOutput
from app.models.pipelines.factory import build_pipeline_registry
from app.models.runtime.generation_state import GenerationState
from app.models.utils.chunking import chunk_text_with_offsets


class RefineWorkflow(BaseWorkflow):
    """
    Refine generation workflow.

    Initial generation followed by refinement steps.
    """
    def __init__(
        self,
        *,
        runtime,
        output_transform=None,
        reasoning=None,
        registry=None,
    ):
        output_transform = PlainTextOutput(runtime)
        super().__init__(
            runtime=runtime,
            output_transform=output_transform,
            reasoning=reasoning,
            pipeline_registry=registry or build_pipeline_registry(runtime),
        )

    def generate_once(self, prompt: str | torch.Tensor, stage: str, **kwargs) -> GenerationState:
        pipeline = self.registry.get(stage)
        state = pipeline.run(
             prompt, 
             mode=kwargs.get("mode"), 
             max_tokens=kwargs.get("max_tokens"), 
             stream=kwargs.get("stream", False), 
             **kwargs,
             )
        return state
    
    def run(self, prompt: str, **kwargs):
        """
        Orchestration logic for refine workflow.
        """
        summaries = None
        chunks = chunk_text_with_offsets(text=prompt, tokenize_fn=self.runtime.tokenize)

        # Chunk summary generation
        for chunk in chunks:
            mid_state = self.generate_once(chunk, stage="summarize", **kwargs)
            summaries = torch.cat(mid_state.input_ids)

        # Refinement phase
        if summaries is not None:
            state = self.generate_once(summaries, stage="generate", **kwargs)
        else:
            state = self.generate_once(prompt, stage="generate", **kwargs)

        return self.output_transform.finalize(state)