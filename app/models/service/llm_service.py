# app/models/service/llm_service.py
from typing import Iterable, Optional, Union

from app.models.orchestrator.workflows.factory import build_workflow_registry
from app.models.orchestrator.workflows.registry import WorkflowRegistry
from app.models.pipelines.base import BaseWorkflow
from app.models.runtime.llm_runtime import LLMRuntime

LLMResult = Union[str, Iterable[str]]

class LLMService:
    """
    External-facing Facade for LLM inference.

    Responsibilities:
    - expose stable semantic APIs (generate / summarize / plan / refine / answer)
    - route calls to appropriate workflows
    - remain agnostic to decoding, reasoning, and streaming mechanics
    """

    def __init__(
        self,
        *,
        workflows: Optional[WorkflowRegistry] = None,
        runtime: Optional[LLMRuntime] = None,
    ):
        self.runtime = runtime or LLMRuntime()

        # default workflows (factory-based)
        self.registry = workflows or build_workflow_registry(
            runtime=self.runtime
        )

    # -------------------------
    # Core helpers
    # -------------------------
    def _select_workflow(self, name: str, **kwargs) -> BaseWorkflow:
        if name not in self.registry:
            raise ValueError(f"Workflow '{name}' not registered")

        return self.registry.get(name)

    def _run_workflow(
        self,
        workflow: BaseWorkflow,
        *args,
        stream: bool = False,
        **kwargs,
    ):
        if workflow not in self.registry:
            raise ValueError(f"Workflow '{workflow.name}' not registered")

        result = workflow.run(*args, stream=stream, **kwargs)

        return result

    # -------------------------
    # External APIs
    # -------------------------

    def generate(
        self,
        prompt: str,
        *,
        mode: str = "instruct",
        stream: bool = False,
        **kwargs,
    ) -> LLMResult:
        """
        Basic text generation.
        """
        pipeline = self._select_workflow("generate")
        result = self._run_workflow(
            pipeline,
            prompt,
            mode=mode,
            stream=stream,
            **kwargs,
        )

        return result

    def summarize(
        self,
        query: str,
        text: str,
        *,
        stream: bool = False,
        **kwargs,
    ) -> LLMResult:
        result = self._run_workflow(
            "summarize",
            query,
            text,
            stream=stream,
            **kwargs,
        )

        return result

    def plan(
        self,
        prompt: str,
        *,
        stream: bool = False,
        **kwargs,
    ) -> LLMResult:
        return self._run_workflow(
            "think",
            prompt,
            stream=stream,
            **kwargs,
        )

    def refine(
        self,
        query: str,
        text: str,
        *,
        stream: bool = False,
        **kwargs,
    ) -> LLMResult:
        return self._run_workflow(
            "refine",
            query,
            text,
            stream=stream,
            **kwargs,
        )

    def answer(
        self,
        question: str,
        *,
        context: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResult:
        """
        RAG-style answering (future).
        """
        return self._run_workflow(
            "answer",
            question,
            context=context,
            stream=stream,
            **kwargs,
        )
