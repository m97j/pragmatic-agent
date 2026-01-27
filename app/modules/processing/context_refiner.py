# app/modules/processing/context_refiner.py
from app.config import REFINE_THRESHOLD as Threshold
from app.logs.logger import get_logger
from app.models.reranker_model import rerank_results
from app.modules.processing.fallback_handler import (handle_query_fallback,
                                                     handle_snippet_fallback)

logger = get_logger(__name__)

def refine_results(search_results, input_message):
    refined_all = []

    for entry in search_results:
        q = entry["query"]

        if not entry["results"]:  # query 단위 fallback
            fallback_entry = handle_query_fallback(q, input_message)
            entry["results"] = fallback_entry["results"]

        # rerank + refine
        reranked = rerank_results(entry["results"], q)
        refined_entry = {"query": q, "results": []}

        for r in reranked:
            score = r.get("score", 0)
            snippet = r.get("snippet", "") if score >= Threshold else ""
            refined_entry["results"].append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": snippet,
                "score": score
            })

        # fallback: if all snippets empty → refine top1
        if all(not r["snippet"] for r in refined_entry["results"]):
            top1 = reranked[0] if reranked else None
            if top1:
                snippet = handle_snippet_fallback(top1, input_message)
                refined_entry["results"][0]["snippet"] = snippet

        refined_all.append(refined_entry)

    return refined_all
