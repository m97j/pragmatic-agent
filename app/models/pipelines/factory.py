# app/models/pipelines/factory.py
from app.models.pipelines.answer import AnswerPipeline
from app.models.pipelines.classify import ClassifyPipeline
from app.models.pipelines.draft import DraftPipeline
from app.models.pipelines.extract import ExtractPipeline
from app.models.pipelines.generate import GeneratePipeline
from app.models.pipelines.registry import PipelineRegistry
from app.models.pipelines.summarize import SummarizePipeline
from app.models.runtime.llm_runtime import LLMRuntime


def build_pipeline_registry(runtime: LLMRuntime) -> PipelineRegistry:
    """
    Build default pipeline presets.

    Each pipeline represents a role-specific preset composed of:
    - decoding strategy
    - optional reasoning
    - optional controller
    """
    registry = PipelineRegistry(runtime)

    registry.register("default.generate", GeneratePipeline(runtime))
    registry.register("default.answer", AnswerPipeline(runtime))
    registry.register("default.summarize", SummarizePipeline(runtime))
    registry.register("classify", ClassifyPipeline(runtime))
    registry.register("extract", ExtractPipeline(runtime))
    registry.register("fast", DraftPipeline(runtime))

    return registry
