# app/runtime/scheduler.py
import os
import shutil
from datetime import datetime

from huggingface_hub import upload_file

from app.config import FALLBACK_DATASET_ID, HF_TOKEN
from app.logs.logger import get_logger
from app.modules.processing.postprocess import get_file_lock
from app.runtime.request_limit import reset_counters
from app.runtime.session_store import finalize_session, get_expired_sessions

log = get_logger(__name__)
tuning_file_lock = get_file_lock()

TUNING_FILE = "interactions.jsonl"
LOG_DIR = "app/logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def upload_and_rotate_logs():
    # Create file names based on today's date
    date_str = datetime.now().strftime("%Y%m%d")
    upload_filename = f"app-{date_str}.json"
    upload_path = os.path.join(LOG_DIR, upload_filename)

    # 1. Rename existing log file (create snapshot)
    if os.path.exists(LOG_FILE):
        shutil.move(LOG_FILE, upload_path)
        # 2. Create a new empty log file (so the logger can continue appending)
        open(LOG_FILE, "w").close()

        # 3. Upload to Hugging Face Hub
        upload_file(
            path_or_fileobj=upload_path,
            path_in_repo=f"logs/{upload_filename}",
            repo_id=FALLBACK_DATASET_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )

        # 4. Delete files inside the container after uploading
        os.remove(upload_path)
        log.info(f"Uploaded and cleared logs: {upload_filename}")
    else:
        log.info("No log file found to upload.")

def push_and_clear_interactions():
    """
    Push local interactions.jsonl data to HF Datasets Hub and clear local file safely.
    """
    hf_token = HF_TOKEN
    dataset_id = FALLBACK_DATASET_ID

    # Step 1: Lock and rename file
    with tuning_file_lock:
        if not os.path.exists(TUNING_FILE):
            log.info("No local TUNING file to push.")
            return
        
        date_str = datetime.now().strftime("%Y%m%d")
        replace_postfix = f"_upload{date_str}.jsonl"
        temp_file = TUNING_FILE.replace(".jsonl", replace_postfix)
        shutil.move(TUNING_FILE, temp_file)

        open(TUNING_FILE, "w").close()  # create a new empty file
    
    # Step 2: Push to HF Datasets Hub
    try:
        upload_file(
            path_or_fileobj=temp_file,
            repo_id=dataset_id,
            repo_type="dataset",
            path_in_repo=f"interactions/{temp_file}",
            token=hf_token
        )
    except Exception as e:
        log.error("Failed to read/clear local TUNING file:", e)
        return
    
    # Step 3: Remove temp file after successful upload
    try:
        os.remove(temp_file)
        log.info(f"Pushed TUNING data to {dataset_id} and cleared local file.")
    except Exception as e:
        log.error("Failed to remove temporary TUNING upload file:", e)

def cleanup_expired_sessions():
    """Store all user sessions in HF Dataset (recalled periodically)"""
    for hf_user, expired_sessions in get_expired_sessions():
        for session_id in expired_sessions:
            finalize_session(hf_user, session_id)

def reset_counters_per_day():
    """Daily Counter Reset Task Scheduler"""
    reset_counters()