# app/models/decoding/engine.py
from typing import Generator, List, Optional

import torch

from app.models.decoding.modifiers.base import LogitsModifier
from app.models.decoding.policies.base import DecodingPolicy
from app.models.runtime.decoding_results import DecodingResults
from app.models.runtime.generation_state import GenerationState
from app.models.runtime.llm_runtime import LLMRuntime


class DecodingEngine:
    """
    Executes a single decoding run using:
    - one decoding policy
    - optional logits modifiers
    """

    def __init__(
        self,
        runtime: LLMRuntime,
        decoding_policy: DecodingPolicy,
        modifiers: Optional[List[LogitsModifier]] = None,
    ):
        self.runtime = runtime
        self.policy = decoding_policy
        self.modifiers = modifiers or []

    # ------------------------------
    # Core primitive: model forward
    # ------------------------------

    def forward(self, state: GenerationState):
        logits, new_past = self.runtime.decode_step(
            state.input_ids,
            state.past_key_values,
            state.attention_mask,
        )

        for modifier in self.modifiers:
            logits = modifier.apply(logits, state)

        return logits, new_past

    # ---------------------------------
    # Token selection only (no mutation)
    # ---------------------------------

    def select(self, logits, state: GenerationState) -> DecodingResults:
        return self.policy.select(logits, state)

    # ------------------------------
    # State mutation (explicit)
    # ------------------------------

    def update_state(
        self,
        state: GenerationState,
        result: DecodingResults,
        past_key_values,
        *,
        metadata=None,
        score=None,
    ) -> GenerationState:

        token = result.token_id.view(1, 1).to(state.input_ids.device)

        new_input_ids = torch.cat([state.input_ids, token], dim=-1)

        new_attention_mask = (
            torch.cat([state.attention_mask, torch.ones_like(token)], dim=-1)
            if state.attention_mask is not None
            else None
        )

        new_state = GenerationState(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values,
            step=state.step + 1,
            scores=score,
            metadata=metadata,
        )

        new_state.finished = result.stop

        return new_state

    # ---------------------------------
    # High-level convenience step
    # (optional for simple decoding)
    # ---------------------------------

    def step(self, state: GenerationState) -> GenerationState:
        logits, past = self.forward(state)

        result = self.select(logits, state)

        return self.update_state(
            state,
            result,
            past,
            metadata=result.metadata,
            score=result.score,
        )
    
    # default decoding loop
    def run(
        self,
        *,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_tokens: int,
        stream: bool = False,
        **kwargs,
    ) -> GenerationState | Generator[GenerationState, None, None]:

        def _decode_loop():
            nonlocal state

            while not state.finished and state.step < max_tokens:
                state = self.step(state)

                yield state

        # ---- init state ----
        state = GenerationState(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        logits, past = self.prefill(state)

        for modifier in self.modifiers:
            logits = modifier.apply(logits, state)

        result = self.policy.select(logits, state)
        state = self.update_state(
            state,
            result,
            past,
        )

        if stream:
            return _decode_loop()

        # non-stream
        for state in _decode_loop():
            pass
        return state

    # ---------------------------------
    # Branching utilities (beam/spec)
    # ---------------------------------

    def clone_state(self, state: GenerationState) -> GenerationState:
        return state.clone()

    def fork_state(
        self,
        state: GenerationState,
        *,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
    ) -> GenerationState:
        new = state.clone()

        if input_ids is not None:
            new.input_ids = input_ids
        if attention_mask is not None:
            new.attention_mask = attention_mask
        if past_key_values is not None:
            new.past_key_values = past_key_values

        return new

    # ---------------------------------
    # Prefill primitive (controller use)
    # ---------------------------------

    def prefill(self, state: GenerationState):
        logits, past = self.runtime.prefill(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
        )
        return logits, past


    def apply_overrides(self, **kwargs):
        """
        Apply any runtime/decoding/modifier overrides before execution.
        """
        self.policy.apply_overrides(**kwargs)

        for modifier in self.modifiers:
            modifier.apply_overrides(**kwargs)
