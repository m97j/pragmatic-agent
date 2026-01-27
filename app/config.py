# app/config.py
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL_HUB = os.environ.get("HF_MODEL_HUB", "m97j/pragmatic-search")

FALLBACK_DATASET_ID = os.environ.get("FALLBACK_DATASET_ID", "m97j/pls-datasets")
SESSIONS_DATASET_ID = os.environ.get("HF_SESSIONS_DATASET_ID", "pls-assistant-sessions")

HF_LLM_FILENAME = os.environ.get("HF_LLM_FILENAME", "llm/low-model_int8.pt")
HF_CONFIG_FILENAME = os.environ.get("HF_CONFIG_FILENAME", "llm/config.json")
HF_RERANKER_FILENAME = os.environ.get("HF_RERANKER_FILENAME", "reranker/model_quantized.onnx")

HF_LLM_REPO = os.environ.get("HF_LLM_REPO", "Qwen/Qwen3-4B")
HF_RERANKER_REPO = os.environ.get("HF_RERANKER_REPO", "BAAI/bge-reranker-v2-m3")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
TAVILY_SEARCH_API_KEY = os.environ.get("TAVILY_SEARCH_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
RAG_API_URL = os.environ.get("RAG_API_URL", "fallback")

REFINE_THRESHOLD = float(os.environ.get("REFINE_THRESHOLD", "0.7"))