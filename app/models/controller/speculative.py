# models/controller/speculative.py
from typing import List, Tuple

import torch

from app.models.controller.base import GenerationController
from app.models.decoding.engine import DecodingEngine
from app.models.runtime.generation_state import GenerationState
from app.models.runtime.kv_cache.manager import KVCacheManager


class SpeculativeController(GenerationController):
    """
    Production-grade speculative decoding controller
    (KV-safe + extensible + controller-level stopping logic)

    Supports:
    - probabilistic acceptance
    - rollback-safe KV handling
    - early stop hooks
    - verifier / entropy / threshold extensions
    """

    def __init__(
        self,
        draft_pipeline: DecodingEngine,
        target_pipeline: DecodingEngine,
        kv_manager: KVCacheManager,
        max_draft_tokens: int = 5,
        eos_token_id: int | None = None,
    ):
        super().__init__([draft_pipeline, target_pipeline])

        self.draft = draft_pipeline
        self.target = target_pipeline
        self.kv = kv_manager

        self.max_draft_tokens = max_draft_tokens
        self.eos_token_id = eos_token_id

    # ---------------------------------------------------------
    # Core public API
    # ---------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        prompt_ids: torch.Tensor,
        attention_mask=None,
        max_tokens: int = 128,
    ) -> GenerationState:

        # Prefill target model
        state = GenerationState(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            step=0,
            finished=False,
        )

        state, past_kvs = self.target.prefill(state)
        kv_key = self.kv.register(past_kvs)

        generated = 0

        while not state.finished and generated < max_tokens:

            base_step = state.step
            base_kv = self.kv.get(kv_key)

            # ---- 1. Draft propose ----
            draft_tokens, draft_probs, draft_kv = self._draft_propose(
                state,
                base_kv,
            )

            # ---- 2. Target verify ----
            accepted, state, generated = self._verify_and_commit(
                state,
                base_step,
                draft_tokens,
                draft_probs,
                generated,
            )

            # ---- 3. Cleanup ----
            self.kv.drop(draft_kv)

        return state

    # ---------------------------------------------------------
    # Draft stage
    # ---------------------------------------------------------

    def _draft_propose(
        self,
        state: GenerationState,
        base_kv,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], object]:

        draft_kv = self.kv.fork(base_kv)
        draft_state = state.copy(past_key_values=draft_kv)

        tokens = []
        probs = []

        for _ in range(self.max_draft_tokens):

            logits, past = self.draft.forward(draft_state)

            result = self.draft.select(logits, draft_state)

            token = result.token_id
            prob_dist = torch.softmax(logits[:, -1], dim=-1)
            prob = prob_dist.gather(-1, token)

            tokens.append(token)
            probs.append(prob)

            draft_state = self.draft.update_state(
                draft_state,
                result,
                past,
            )

            if draft_state.finished:
                break

        return tokens, probs, draft_kv

    # ---------------------------------------------------------
    # Verification stage
    # ---------------------------------------------------------

    def _verify_and_commit(
        self,
        state: GenerationState,
        base_step: int,
        tokens: List[torch.Tensor],
        draft_probs: List[torch.Tensor],
        generated: int,
    ) -> Tuple[int, GenerationState, int]:

        accepted = 0

        for token, p_draft in zip(tokens, draft_probs):

            candidate_ids = torch.cat([state.input_ids, token], dim=-1)

            candidate_state = state.copy(
                input_ids=candidate_ids,
                past_key_values=state.past_key_values,
            )

            logits, past = self.target.forward(candidate_state)
            result = self.target.select(logits, candidate_state)

            probs_t = torch.softmax(logits[:, -1], dim=-1)
            p_target = probs_t.gather(-1, token)

            accept_prob = torch.minimum(
                torch.ones_like(p_target),
                p_target / p_draft,
            )

            if self._accept(accept_prob):
                state = self.target.update_state(state, result, past)
                state.step += 1
                generated += 1
                accepted += 1

                if self._should_stop_token(token, state):
                    state.finished = True
                    break
            else:
                # ---- rollback + greedy step ----
                rolled_kv = self.kv.rollback(
                    state.past_key_values,
                    base_step,
                )

                state.past_key_values = rolled_kv

                state = self.target.step(state)
                state.step += 1
                generated += 1
                break

        return accepted, state, generated

    # ---------------------------------------------------------
    # Acceptance policy (extensible)
    # ---------------------------------------------------------

    def _accept(self, accept_prob: torch.Tensor) -> bool:
        return torch.rand(1, device=accept_prob.device) < accept_prob

    # ---------------------------------------------------------
    # Unified stopping hook
    # ---------------------------------------------------------

    def _should_stop_token(self, token, out: GenerationState) -> bool:

        if out.finished:
            return True

        if self.eos_token_id is not None:
            if token.item() == self.eos_token_id:
                return True

        return False
