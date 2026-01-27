# app/models/outputs/sentence.py
import re
from typing import Iterable, Iterator

from app.models.outputs.base import OutputTransform
from app.models.runtime.generation_state import GenerationState

_SENTENCE_END = re.compile(r"([.!?。！？])")


class SentenceOutput(OutputTransform):
    """
    Stream output sentence-by-sentence.
    """

    def stream(
        self,
        states: Iterable[GenerationState],
    ) -> Iterator[str]:
        buffer = ""
        prev_len = 0

        for state in states:
            text = self.runtime.detokenize(state.input_ids)
            delta = text[prev_len:]
            prev_len = len(text)

            buffer += delta

            while True:
                match = _SENTENCE_END.search(buffer)
                if not match:
                    break

                end = match.end()
                sentence = buffer[:end]
                buffer = buffer[end:]

                yield sentence

        if buffer.strip():
            yield buffer

    def finalize(self, state: GenerationState) -> str:
        return self.runtime.detokenize(state.input_ids)
