# app/models/outputs/tool_call.py
import json
from typing import Iterable, Iterator

from app.models.outputs.base import OutputTransform
from app.models.runtime.generation_state import GenerationState


class ToolCallOutput(OutputTransform):
    """
    Streams partial JSON and emits only valid JSON objects.
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

            try:
                parsed = json.loads(buffer)
                yield json.dumps(parsed)
                buffer = ""
            except json.JSONDecodeError:
                continue

    def finalize(self, state: GenerationState) -> str:
        text = self.runtime.detokenize(state.input_ids)
        try:
            return json.dumps(json.loads(text))
        except Exception:
            return text
