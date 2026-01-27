# app/models/reranker_model.py
import numpy as np

from logs.logger import get_logger
from models.initializer import get_models

logger = get_logger(__name__)

def rerank_results(results, query=None, top_k=None):
    """
    ONNX Runtime-based reranker for Qwen/Qwen3-Reranker-0.6B.
    Args:
        results: list of dicts [{"title":..., "url":..., "snippet":...}, ...]
        query: user question (multilingual string)
        top_k: number of results to return
    Returns:
        list of dicts with added "score" field:
        [{"title":..., "url":..., "snippet":..., "score":...}, ...]
    Notes:
        - Hugging Face tokenizer does NOT support return_tensors="np". However, ONNX Runtime expects np as input
          Therefore, use return_tensors="pt" and convert it to numpy via .cpu().numpy() to match the input format.
        - scores shape may vary (batch_size,) or (batch_size,1), so reshape(-1) applied.
        - fallback ensures "score" key exists with None if inference fails.
    """
    if not results:
        return []

    if not query:
        # No query → return top_k without reranking
        return [dict(r, score=None) for r in results[:top_k]]
    
    models = get_models()

    # Defensive: ensure keys exist
    reranker = models.get("reranker")
    tokenizer = models.get("reranker_tokenizer")
    if reranker is None or tokenizer is None:
        raise RuntimeError("Reranker or tokenizer not found in models cache")
    
    # Combine title + snippet for richer context
    docs = [
        (r.get("title", "") + " " + r.get("snippet", "")).strip()
        for r in results
    ]

    # Tokenize (query, document) pairs
    encodings = tokenizer(
        [query] * len(docs),
        docs,
        padding=True,
        truncation=True,
        return_tensors="pt"  # Hugging Face tokenizer supports "pt"
    )

    # Convert PyTorch tensors to numpy for ONNX Runtime
    expected_inputs = [inp.name for inp in reranker.get_inputs()]
    inputs = {}
    for k in expected_inputs:
        if k in encodings:
            inputs[k] = encodings[k].cpu().numpy()

    try:
        raw_scores = reranker.run(None, inputs)[0]
        if raw_scores is None or raw_scores.size == 0:
            raise ValueError("Reranker returned empty scores")
        scores = np.array(raw_scores).reshape(-1)


        reranked = []
        for i, r in enumerate(results):
            item = r.copy()
            if i < len(scores):
                val = scores[i]
                if hasattr(val, "item"):
                    val = val.item()
                item["score"] = float(val)
            else:
                item["score"] = 0.0
            reranked.append(item)
        
        if top_k is not None:
            reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)[:top_k]
        return reranked
    
    except Exception as e:
        logger.error(f"Reranker ONNX inference failed: {e}")
        logger.debug("Traceback: ", exc_info=True)
        # fallback: return top_k with score=None
        return [dict(r, score=None) for r in results[:top_k]]
