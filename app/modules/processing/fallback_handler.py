# app/modules/processing/fallback_handler.py
from app.logs.logger import get_logger
from app.models.service.llm_service import LLMService
from app.modules.data.page_crawler import extract_main_markdown, fetch_page
from app.modules.data.search_crawler import search_and_crawl

logger = get_logger(__name__)

def _get_snippet(query, text, max_tokens=128):
    llm = LLMService()
    # refinement step (slightly exploratory, non-streaming)
    summary = llm.refine(
        query,
        text,
        mode="instruct",
        strategy="sampling",
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.9,
    )
    return summary

def _get_raw_context(url):
    html = fetch_page(url)
    if html:
        text = extract_main_markdown(html)
        return text
    return ""

def _generate_snippet(query: str, text: str, max_tokens: int = 128) -> str:
    llm = LLMService()
    try:
        return llm.refine(
            query,
            text,
            mode="instruct",
            strategy="sampling",
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
        )
    except Exception as e:
        logger.error(f"[fallback] LLM snippet generation failed: {e}")
        return ""

def handle_query_fallback(query: str, input_message: str) -> dict:
    """
    Fallback for a single query when results are empty.
    First try DuckDuckGo crawling, then fallback to LLM direct generation.
    """
    logger.warning(f"[fallback] No search results for query='{query}', triggering fallback.")
    texts = search_and_crawl(query, num=1)
    if texts:
        snippet = _generate_snippet(input_message, texts[0])
    else:
        snippet = _generate_snippet(input_message, query)
    return {
        "query": query,
        "results": [{
            "title": "Fallback Generated",
            "url": "",
            "snippet": snippet,
            "score": 0.0
        }]
    }

def handle_global_fallback(input_message: str) -> dict:
    """
    Fallback when all queries failed (global fallback).
    """
    logger.critical("[fallback] All queries failed. Triggering global fallback.")
    texts = search_and_crawl(input_message, num=1)
    if texts:
        snippet = _generate_snippet(input_message, texts[0])
    else:
        snippet = _generate_snippet(input_message, input_message)
    return {
        "query": input_message,
        "results": [{
            "title": "Global Fallback",
            "url": "",
            "snippet": snippet,
            "score": 0.0
        }]
    }


def handle_no_results(input_message: str, queries: list) -> list:
    """
    Handle the case when all queries returned no results.
    Apply global fallback.
    """
    logger.critical("[fallback] No results for all queries. Applying global fallback.")
    fallback_entry = handle_global_fallback(input_message)
    return [fallback_entry]

def handle_snippet_fallback(top1: list, input_message: str) -> list:
    """
    For results with empty snippets, generate snippets using LLM.
    """
    raw_context = _get_raw_context(top1.get("url"))
    snippet = _get_snippet(input_message, raw_context)
    
    return snippet
