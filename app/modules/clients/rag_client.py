# app/modules/clients/rag_client.py
import requests

from app.config import RAG_API_URL
from app.logs.logger import get_logger

logger = get_logger(__name__)

def rag_search(query):
    try:
        resp = requests.post(RAG_API_URL, json={"query": query})
        return resp.json().get("results", [])
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        logger.debug(f"Traceback: ", exc_info=True)
        return []
