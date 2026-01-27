# app/models/orchestrator/workflows/registry.py
from typing import Dict

from app.models.orchestrator.workflows.base import BaseWorkflow
from app.models.runtime.llm_runtime import LLMRuntime


class WorkflowRegistry:
    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime
        self._workflows: Dict[str, BaseWorkflow] = {}

    def register(self, name: str, workflow: BaseWorkflow):
        if name in self._workflows:
            raise ValueError(f"Workflow '{name}' is already registered")
        self._workflows[name] = workflow

    def get(self, name: str) -> BaseWorkflow:
        try:
            return self._workflows[name]
        except KeyError:
            raise ValueError(f"Workflow '{name}' is not registered")
        
    def list(self):
        return list(self._workflows.keys())
