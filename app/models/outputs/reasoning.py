# app/models/outputs/reasoning.py
import re
from typing import Iterable, Iterator

from app.models.outputs.base import OutputTransform
from app.models.runtime.generation_state import GenerationState

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class HideReasoningOutput(OutputTransform):
    """
    Removes chain-of-thought from output.
    """

    def _strip_reasoning(self, text: str) -> str:
        return _THINK_BLOCK.sub("", text).strip()

    def stream(
        self,
        states: Iterable[GenerationState],
    ) -> Iterator[str]:
        prev_len = 0
        last_emitted = ""

        for state in states:
            text = self._strip_reasoning(
                self.runtime.detokenize(state.input_ids)
            )

            chunk = text[len(last_emitted):]
            last_emitted = text

            if chunk:
                yield chunk

    def finalize(self, state: GenerationState) -> str:
        text = self.runtime.detokenize(state.input_ids)
        return self._strip_reasoning(text)
