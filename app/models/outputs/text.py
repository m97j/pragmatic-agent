# app/models/outputs/text.py
from typing import Iterable, Iterator

from app.models.outputs.base import OutputTransform
from app.models.runtime.generation_state import GenerationState


class PlainTextOutput(OutputTransform):
    """
    Default text renderer.

    - streaming: diff-based (append-only)
    - non-stream: full detokenize
    """

    def stream(
        self,
        states: Iterable[GenerationState],
    ) -> Iterator[str]:
        prev_len = 0

        for state in states:
            text = self.detokenize(state)
            chunk = text[prev_len:]
            prev_len = len(text)

            if chunk:
                yield chunk

    def finalize(self, state: GenerationState) -> str:
        return self.detokenize(state)
