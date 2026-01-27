# app/models/outputs/markdown.py
from typing import Iterable, Iterator

from app.models.outputs.base import OutputTransform
from app.models.runtime.generation_state import GenerationState


class MarkdownOutput(OutputTransform):
    """
    Flush markdown blocks safely (especially ``` code blocks).
    """

    def stream(
        self,
        states: Iterable[GenerationState],
    ) -> Iterator[str]:
        buffer = ""
        prev_len = 0
        open_block = False

        for state in states:
            text = self.runtime.detokenize(state.input_ids)
            delta = text[prev_len:]
            prev_len = len(text)

            buffer += delta

            while "```" in buffer:
                idx = buffer.index("```")
                block = buffer[: idx + 3]
                buffer = buffer[idx + 3 :]
                open_block = not open_block
                yield block

            if not open_block and buffer.strip():
                yield buffer
                buffer = ""

        if buffer.strip():
            yield buffer

    def finalize(self, state: GenerationState) -> str:
        return self.runtime.detokenize(state.input_ids)
