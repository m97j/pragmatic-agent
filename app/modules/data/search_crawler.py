# app/modules/data/search_crawler.py
import random
import time

import requests
from bs4 import BeautifulSoup

from app.logs.logger import get_logger
from app.modules.data.page_crawler import extract_main_markdown, fetch_page

logger = get_logger(__name__)

DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"

# Simple rate limiter (minimum delay between requests)
_last_request_time = 0
_min_delay = 2.0  # Seconds (keep intervals of 2 seconds or more)

def _rate_limit():
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _min_delay:
        sleep_time = _min_delay - elapsed + random.uniform(0.1, 0.5)
        time.sleep(sleep_time)
    _last_request_time = time.time()

def _search_duckduckgo(query: str, num: int = 2, method: str = "POST") -> list[str]:
    """
    DuckDuckGo HTML search request (GET/POST support)
    - method="GET": Passed as a URL parameter
    - method="POST": Passed in the request body
    """
    _rate_limit()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://duckduckgo.com/",
    }

    try:
        if method.upper() == "GET":
            resp = requests.get(DUCKDUCKGO_HTML_URL, params={"q": query}, headers=headers, timeout=10)
        else:  # Default POST
            resp = requests.post(DUCKDUCKGO_HTML_URL, data={"q": query}, headers=headers, timeout=10)

        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if href and href.startswith("http"):
                links.append(href)
            if len(links) >= num:
                break

        logger.info(f"[search_crawler] Found {len(links)} links for query='{query}'")
        return links
    except Exception as e:
        logger.error(f"[search_crawler] DuckDuckGo search failed for query='{query}' with error: {e}")
        logger.debug("Traceback: ", exc_info=True)
        return []

def search_and_crawl(query: str, num: int = 2, method: str = "POST") -> list[str]:
    """
    Query → DuckDuckGo HTML search → Crawl top URLs → Extract body text
    """
    urls = _search_duckduckgo(query, num=num, method=method)
    texts = []
    for url in urls:
        html = fetch_page(url)
        if html:
            text = extract_main_markdown(html)
            if text:
                texts.append(text)
    return texts
