# app/models/runtime/llm_runtime.py
import threading
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.tokenization_utils_base import BatchEncoding

from app.models.initializer import get_models

PastKeyValues = tuple

class LLMRuntime:
    """
    Low-level LLM inference engine: 
    - Thread-safe single-model runtime.
    - tokenize()/detokenize() for external use
    - token-based inference function.
    - KV-cache is managed outside (GenerationState)
    - This lock serializes forward passes 
    """

    # class lock for thread-safe model inference
    _lock = threading.RLock()

    def __init__(self):
        models = get_models()
        self.model = models["llm"]
        self.tokenizer = models["llm_tokenizer"]
        self.device = next(self.model.parameters()).device
        self.model.eval()

    # ---------------------------
    # Internal helper
    # ---------------------------
    def _ensure_model_on_device(self):
        if torch.cuda.is_available():
            current = next(self.model.parameters()).device
            if current.type != "cuda":
                self.model.to("cuda")
                self.device = torch.device("cuda")

    def _ensure_tensor_ids(self, input_ids: BatchEncoding | torch.Tensor | list) -> torch.Tensor:
        """
        Normalize various input types to a torch.Tensor of IDs on the correct device.
        
        :param self: CustomLLMEngine
        :param input_ids: Accepts BatchEncoding, dict-like, lisk, or tensor.
        :type input_ids: BatchEncoding | torch.Tensor | list
        :return: Input IDs type of tensor
        :rtype: Tensor
        """
        # BatchEncoding / dict-like
        if hasattr(input_ids, "keys"):
            if "input_ids" not in input_ids:
                raise ValueError("There's no 'input_ids' key in BatchEncoding/dict input")
            input_ids = input_ids["input_ids"]
        # list/ndarray to tensor
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids.to(self.device)
    
    def _ensure_inputs(
            self,
            inputs: BatchEncoding | torch.Tensor | list | dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Normalize inputs and optionally return attention_mask if present.
        """
        attention_mask = None
        if hasattr(inputs, "keys"):     # BatchEncoding/dict-like
            if "input_ids" not in inputs:
                raise  ValueError("There's no 'input_ids' key in BatchEncoding/dict-like input")
            input_ids = inputs["input_ids"].to(self.device)
            if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                attention_mask = inputs["attention_mask"].to(self.device)
        else:
            input_ids = self._ensure_tensor_ids(inputs)
        return input_ids, attention_mask
    
    # ---------------------------
    # Tokenize API
    # ---------------------------
    def tokenize(self, text: str, return_offsets: bool = False) -> BatchEncoding:
        """
        External API: convert text to tokens. Returns BatchEncoding on device.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=return_offsets,
            add_special_tokens=True
        )
        return enc.to(self.device)

    def detokenize(self, token_ids, skip_special_tokens: bool = True) -> str:
        """
        External API: convert tokens back to text (decode).
        Safely handles 2D tensors by decoding the first item.
        """
        if torch.is_tensor(token_ids):
            if token_ids.dim() == 2:
                token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ---------------------------
    # Token-based inference API
    # ---------------------------
    def prefill(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Encode full prompt once.
        """
        with LLMRuntime._lock, torch.no_grad():
            self._ensure_model_on_device()
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        return outputs.logits[:, -1, :], outputs.past_key_values

    def decode_step(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PastKeyValues]:
        """
        Single decoding forward step.

        Returns
        -------
        logits : torch.Tensor
            shape [batch, vocab]
        new_past_key_values : tuple
        """
        last_token = input_ids[:, -1:]

        with LLMRuntime._lock, torch.no_grad():
            self._ensure_model_on_device()
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=last_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

    def prefill_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PastKeyValues]:
        """
        Prefill forward for a batch of independent sequences.

        Each sequence has its own past_key_values.

        Inputs
        -------
        input_ids : [B, T]
        attention_mask : [B, T]

        Returns
        -------
        logits : torch.Tensor
            shape [B, vocab]
        new_past_key_values_batch : tuple (layer-wise batched KV)
        """
        with LLMRuntime._lock, torch.no_grad():
            self._ensure_model_on_device()
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

    def decode_batch_step(
        self,
        input_ids: torch.Tensor,
        past_key_values_batch: list,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PastKeyValues]:
        """
        Single decoding forward step for a batch of independent sequences.

        Each sequence has its own past_key_values.

        Inputs
        -------
        input_ids : [B, T]
        past_key_values : batched KV cache
        attention_mask : [B, T]

        Returns
        -------
        logits : torch.Tensor
            shape [B, vocab]
        past_key_values_batch : list
        """
        last_token = input_ids[:, -1:]

        with LLMRuntime._lock, torch.no_grad():
            self._ensure_model_on_device()
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=last_token,
                past_key_values=past_key_values_batch,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

        