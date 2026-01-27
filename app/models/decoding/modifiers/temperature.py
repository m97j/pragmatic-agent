# app/models/decoding/modifiers/temperature.py
import torch

from app.models.decoding.modifiers.base import LogitsModifier
from app.models.runtime.generation_state import GenerationState


class TemperatureModifier(LogitsModifier):
    def __init__(self, temperature: float):
        assert temperature > 0
        self.temperature = temperature

    def apply(
        self,
        logits: torch.Tensor,
        state: GenerationState,
    ) -> torch.Tensor:
        return logits / max(self.temperature, 1e-6)
