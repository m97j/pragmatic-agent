# app/models/prompts/task.py
import torch

from app.models.initializer import get_prefixes


def apply_task_prefix(
    *,
    task: str,
    query_tokens: torch.Tensor,
    content_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Build task-specific prompt tokens.

    Args:
        task: task name (e.g., 'summarize', 'refine')
        query_tokens: tokenized user query
        content_tokens: tokenized document or summaries

    Returns:
        torch.Tensor: concatenated input_ids
    """
    prefixes = get_prefixes()

    if task == "summarize":
        return torch.cat(
            [
                prefixes["summarize"],
                prefixes["newline"],
                prefixes["query"], query_tokens,
                prefixes["document"], content_tokens,
                prefixes["newline"],
                prefixes["summarize_reminder"],
            ],
            dim=-1,
        )

    if task == "refine":
        return torch.cat(
            [
                prefixes["refine"],
                prefixes["newline"],
                prefixes["query"], query_tokens,
                prefixes["summaries"], content_tokens,
                prefixes["newline"],
                prefixes["refine_reminder"],
            ],
            dim=-1,
        )

    raise ValueError(f"Unsupported task: {task}")
