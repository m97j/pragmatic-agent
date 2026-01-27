# app/models/orchestrator/strategies/reflexion.py
# not in use yet
import torch

from app.models.orchestrator.strategies.base import ReasoningStrategy
from app.models.runtime.generation_state import GenerationState


class Reflexion(ReasoningStrategy):
    """
    Reflexion-based reasoning.

    Generates an answer, critiques it, then retries.
    """

    def __init__(
        self,
        controller,
        critique_prompt: str = "The previous answer was incorrect. Let's try again.",
    ):
        super().__init__(controller)
        self.critique_prompt = critique_prompt

    def run(
        self,
        prompt_ids: torch.Tensor,
        tokenizer,
        **kwargs,
    ) -> GenerationState:
        first_state = self.controller.run(prompt_ids)

        critique_ids = tokenizer(
            self.critique_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(prompt_ids.device)

        new_prompt = torch.cat(
            [prompt_ids, first_state.input_ids, critique_ids],
            dim=-1,
        )

        return self.controller.run(new_prompt)
