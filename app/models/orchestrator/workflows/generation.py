# app/models/orchestrator/workflows/generation.py
from app.models.orchestrator.workflows.base import BaseWorkflow
from app.models.pipelines.factory import build_pipeline_registry
from app.models.runtime.generation_state import GenerationState


class GenerationWorkflow(BaseWorkflow):
    """
    Basic generation workflow (single or multi-sample).
    """
    def __init__(
        self,
        *,
        runtime,
        output_transform,
        reasoning=None,
        registry=None,
    ):
        super().__init__(
            runtime=runtime,
            output_transform=output_transform,
            reasoning=reasoning,
            pipeline_registry=registry or build_pipeline_registry(runtime),
        )

    def generate_once(self, prompt: str, **kwargs) -> GenerationState:
        pipeline = self.registry.get("generate")
        state = pipeline.run(
             prompt, 
             mode=kwargs.get("mode"), 
             max_tokens=kwargs.get("max_tokens"), 
             stream=kwargs.get("stream", False), 
             **kwargs,
             )
        return state
    
    def run(self, prompt: str, streaming: bool, **kwargs):
        """
        Default orchestration logic (single or multi generation).
        """
        generated = self.generate_once(prompt, **kwargs)
        if streaming:
            return self.output_transform.stream(generated)
        else:
            return self.output_transform.finalize(generated)
