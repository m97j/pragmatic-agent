# app/models/orchestrator/workflows/factory.py
from app.models.orchestrator.workflows.generation import GenerationWorkflow
from app.models.orchestrator.workflows.multi_turn import MultiTurnWorkflow
from app.models.orchestrator.workflows.refine import RefineWorkflow
from app.models.orchestrator.workflows.registry import WorkflowRegistry
from app.models.orchestrator.workflows.think import ThinkWorkflow
from app.models.runtime.llm_runtime import LLMRuntime


def build_workflow_registry(runtime: LLMRuntime) -> WorkflowRegistry:
    """
    Build default workflow presets.

    Each pipeline represents a role-specific preset composed of:
    - decoding strategy
    - optional reasoning
    - optional controller
    """
    registry = WorkflowRegistry(runtime)
    registry.register("default.generate", GenerationWorkflow(runtime))
    registry.register("default.refine", RefineWorkflow(runtime))
    registry.register("default.multi_turn", MultiTurnWorkflow(runtime))
    registry.register("default.think", ThinkWorkflow(runtime))

    return registry
