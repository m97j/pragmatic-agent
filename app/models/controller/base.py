# app/models/controller/base.py
from abc import ABC, abstractmethod
from typing import List

import torch

from app.models.decoding.engine import DecodingEngine
from app.models.runtime.generation_state import GenerationState


class GenerationController(ABC):
    """
    High-level generation controller.

    Responsibilities
    ----------------
    - Orchestrate one or more DecodingEngines
    - Manage execution strategy (retry, branch, validate, select)
    - Aggregate or select GenerationState outputs

    Notes
    -----
    - Does NOT perform token streaming itself
    - Delegates actual generation to runtime via DecodingPipeline
    """

    speculative_k: int = None
    retry: int = None

    def __init__(self, pipelines: DecodingEngine | list[DecodingEngine]):
        self.pipelines = (
            pipelines if isinstance(pipelines, list) else [pipelines]
        )

    def apply_overrides(self, **kwargs):
        """
        Apply any controller-specific overrides before execution.
        """
        if kwargs.get("speculative_k") is not None:
            self.speculative_k = kwargs["speculative_k"]
        if kwargs.get("retry") is not None:
            self.retry = kwargs["retry"]

    @abstractmethod
    def run(
        self,
        *,
        prompt_ids: torch.Tensor,
        stream: bool = False,
        **kwargs,
    ) -> List[GenerationState]:
        """
        Execute controlled generation.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            Tokenized prompt input
        stream : bool
            Whether downstream runtime should stream tokens

        Returns
        -------
        List[GenerationState]
            One or more completed generation states
        """
        raise NotImplementedError
