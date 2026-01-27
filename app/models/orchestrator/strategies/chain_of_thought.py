# app/models/orchestrator/strategies/chain_of_thought.py
# not in use yet
import torch

from app.models.orchestrator.strategies.base import ReasoningStrategy
from app.models.runtime.generation_state import GenerationState


class ChainOfThought(ReasoningStrategy):
    """
    Chain-of-Thought reasoning.

    Adds an explicit reasoning instruction before generation.
    """

    def __init__(
        self,
        controller,
        cot_prompt: str = "Let's think step by step.",
    ):
        super().__init__(controller)
        self.cot_prompt = cot_prompt

    def run(
        self,
        prompt_ids: torch.Tensor,
        tokenizer,
        **kwargs,
    ) -> GenerationState:
        cot_ids = tokenizer(
            self.cot_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(prompt_ids.device)

        augmented_prompt = torch.cat([prompt_ids, cot_ids], dim=-1)

        return self.controller.run(augmented_prompt)
