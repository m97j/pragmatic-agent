# app/models/pipelines/registry.py
from typing import Dict

from app.models.pipelines.base import BasePipeline
from app.models.runtime.llm_runtime import LLMRuntime


class PipelineRegistry:
    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime
        self._pipelines: Dict[str, BasePipeline] = {}

    def register(self, name: str, pipeline: BasePipeline):
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' is already registered")
        self._pipelines[name] = pipeline

    def get(self, name: str) -> BasePipeline:
        try:
            return self._pipelines[name]
        except KeyError:
            raise ValueError(f"Pipeline '{name}' is not registered")
        
    def list(self):
        return list(self._pipelines.keys())
