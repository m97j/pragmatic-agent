# app/modules/processing/query_processor.py
import re
from typing import List


def normalize_query(query: str) -> str:
    """
    Normalize a query string.
    - Trim whitespace
    - (Optional) Add rules for punctuation or casing if needed
    """
    return query.strip()


def deduplicate_queries(queries: List[str]) -> List[str]:
    """
    Remove exact duplicate queries.
    """
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            unique.append(q)
            seen.add(q)
    return unique


def select_top_k_queries(queries: List[str], reranker, k: int = 5) -> List[str]:
    """
    Select top-k queries using a reranker.
    - Keep broad coverage while reducing redundant API calls
    - Reranker should be provided externally (function/class)
    """
    if not queries:
        return []

    scored = reranker(queries)  # expected: List of (query, score)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [q for q, _ in scored[:k]]


def process_queries(raw_output: str, max_queries: int = 5) -> List[str]:
    """
    Entry point: process raw LLM output into final queries.
    - Split by line
    - Remove leading numbers or list markers
    - Normalize each query
    - Deduplicate identical queries
    - Limit to max_queries
    """
    queries = []
    for line in raw_output.splitlines():
        q = line.strip()
        q = re.sub(r"^[\d\.\-\)\s]+", "", q)  # remove leading numbering
        q = normalize_query(q)

        if q:
            queries.append(q)

        if len(queries) >= max_queries:
            break

    queries = deduplicate_queries(queries)
    return queries

