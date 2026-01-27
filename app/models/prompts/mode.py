# app/models/prompts/mode.py
from typing import Optional, Union

import torch

from app.models.initializer import get_prefixes


def apply_mode_prefix(
    prompt: Union[str, torch.Tensor],
    mode: Optional[str],
) -> Union[str, torch.Tensor]:
    """
    Stateless mode adapter.
    Applies predefined prefixes based on mode.
    """

    if not mode:
        return prompt

    mode = mode.lower()

    # ---------- string prompt ----------
    if isinstance(prompt, str):
        if mode == "instruct":
            return f"/no_think\n{prompt}"
        elif mode == "think":
            return f"/think\n{prompt}"
        return prompt

    # ---------- token prompt ----------
    if isinstance(prompt, torch.Tensor):
        prefixes = get_prefixes()

        if mode == "instruct":
            return torch.cat([prefixes["instruct"], prompt], dim=-1)
        elif mode == "think":
            return torch.cat([prefixes["think"], prompt], dim=-1)

        return prompt

    raise TypeError(f"Unsupported prompt type: {type(prompt)}")
