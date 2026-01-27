# app/modules/processing/postprocess.py
import json
import threading
from datetime import datetime

from app.logs.logger import get_logger

log = get_logger(__name__)
_file_lock = threading.Lock()

TUNING_FILE = "interactions.jsonl"

def finalize_answer(user_message, context_docs, model_answer=None, hf_token=None):
    """
    Post-processing the final answer:
    - Push the input/output pairs to the Hugging Face Datasets Hub
    - The HF_TOKEN environment variable is required when running Space
    """
    # 1. Create record
    record = {
        "timestamp": datetime.now().isoformat(),
        "input": user_message,
        "context": [doc.get("snippet", "") for doc in (context_docs or [])],
        "output": model_answer or "",
    }

    # 2. Append to local jsonl file to further model training and save context datas as cache to avoid api cost
    try:
        with _file_lock:
            with open(TUNING_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error("Failed to write to local log file:", e)

def get_file_lock():
    return _file_lock

