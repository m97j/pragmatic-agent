# app/models/runtime/decoding_results.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class DecodingResults:
    token_id: torch.Tensor

    # logprob or score for beam / ranking
    score: Optional[float] = None

    # auxiliary algorithm signals
    metadata: Dict[str, Any] = field(default_factory=dict)

    # optional early stop signal
    stop: bool = False
