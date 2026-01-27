# app/modules/data/page_crawler.py
import markdownify
import requests
from bs4 import BeautifulSoup

from app.logs.logger import get_logger

logger = get_logger(__name__)

def fetch_page(url: str, timeout: int = 10) -> str | None:
    """
    Retrieves an HTML page from a given URL
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        logger.info(f"[crawler] Successfully fetched page: {url}, length={len(resp.text)}")
        return resp.text
    except Exception as e:
        logger.error(f"[crawler] fetch_page failed for URL: {url} with error: {e}")
        logger.debug("Traceback: ", exc_info=True)
        return None


def extract_main_markdown(html: str) -> str:
    """
    Extract body text from HTML and convert to Markdown
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        body = soup.body or soup

        # Convert to markdown
        md_text = markdownify.markdownify(str(body), strip=['a'])

        return md_text
    except Exception as e:
        logger.error(f"[crawler] extract_main_text failed: {e}")
        logger.debug("Traceback: ", exc_info=True)
        return ""
