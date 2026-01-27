# app/modules/common/utils.py
import re


def count_tokens(text):
    return len(text.split())

def split_content(text: str, role: str = None, return_boundaries: bool = False):
    """
    Split text into snippets or boundaries.
    - user: sentence-level split
    - assistant: markdown-aware split
    - plain text: sentence-level split
    usage modules: conversation/history_manager.py, data/page_crawler.py, models/llm_model.py -> refine -> _chunk_tokens_with_offsets_safe()
    """
    snippets = []
    boundaries = []

    if role == "assistant":
        # markdown-aware split
        blocks = re.split(r'(```.*?```|\|.*?\|.*?\|.*?\|)', text, flags=re.S)
        for b in blocks:
            if not b.strip():
                continue
            if b.startswith("```") or b.startswith("|"):
                snippets.append(b.strip())
                boundaries.append(len(text))  # treat block as one unit
            else:
                sentences = re.split(r'(?<=[.!?])\s+|\n+', b)
                for s in sentences:
                    if s.strip():
                        snippets.append(s.strip())
                        boundaries.append(text.find(s) + len(s))
    else:
        # plain text or user role
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        for s in sentences:
            if s.strip():
                snippets.append(s.strip())
                boundaries.append(text.find(s) + len(s))

    return boundaries if return_boundaries else snippets
