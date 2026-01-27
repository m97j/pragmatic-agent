# app/modules/clients/search_client.py
import requests

from app.config import (GOOGLE_API_KEY, GOOGLE_CSE_ID, SERPER_API_KEY,
                        TAVILY_SEARCH_API_KEY)
from app.logs.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------
# Tavily Search API (Input_message Main)
# -----------------------------------------
def tavily_search(query, num=5):
    try:
        logger.info(f"Searching tavily with query='{query}' and num={num}")
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {TAVILY_SEARCH_API_KEY}"}
        payload = {"query": query, "max_results": num}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        # If there is an answer field, add it as a snippet candidate (leave the url blank)
        if "answer" in data and data["answer"]:
            results.append({
                "title": "Tavily Answer",          
                "url": "",            
                "snippet": data["answer"]
            })

        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })
        return results
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        logger.debug(f"Traceback: ", exc_info=True)        
        return None


# --------------------------
# Serper.dev (Query main)
# --------------------------
def serper_search(query, num=5):
    try:
        logger.info(f"Searching serper with query='{query}' and num={num}")
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": num}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return results
    except Exception as e:
        logger.error(f"Serper.dev search failed: {e}")
        logger.debug(f"Traceback: ", exc_info=True)
        return None


# ------------------------------------------
# Google Custom Search API (1st fallback)
# ------------------------------------------
def google_search(query, num=5):
    try:
        logger.info(f"Searching google with query='{query}' and num={num}")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": num,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return results
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        logger.debug(f"Traceback: ", exc_info=True)
        return []
    

# ---------------------------
# Integrated search function
# ---------------------------
def search_with_api(input_message, queries, num=5):
    """
    Integrated search function with role separation:
    - Tavily: called once with the full input_message (AI overview style answer).
    - Serper.dev -> Google: called per query (fallback chain).
    - Results are stored in a unified structure: {"query": q, "results": [...]}
    """

    all_results = []

    # 1. Tavily for input_message (AI overview style)
    tavily_results = tavily_search(input_message, num=num) or []
    logger.info(f"Tavily produced {len(tavily_results)} results for input_message='{input_message}'")

    all_results.append({
        "query": input_message,
        "results": tavily_results
    })

    # # 2. Query-based search (Serper.dev -> Google fallback)
    # for q in queries:
    #     results = (
    #         serper_search(q, num=num)
    #         or google_search(q, num=num)
    #     )
    #     results = results or []
    #     logger.info(f"Query='{q}' produced {len(results)} results (aggregated)")

    #     all_results.append({
    #         "query": q,
    #         "results": results
    #     })

    return all_results
