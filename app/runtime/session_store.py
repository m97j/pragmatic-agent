# app/runtime/session_store.py
import logging
import threading
from datetime import datetime, timedelta
from typing import TypedDict
from uuid import uuid4

import gradio as gr
from huggingface_hub import whoami

from app.config import HF_TOKEN
from app.infrastructure.hf_dataset_client import SessionManager
from app.modules.conversation.history_controller import \
    SessionHistoryController
from app.runtime.request_limit import get_req_count
from app.ui.i18n.translations import translations

# ---------------------------------------------------------------------------
# Global in-memory store of active sessions, keyed by Hugging Face user ID.
# Each user has:
#   - "session_manager": SessionManager instance (handles persistence to HF Hub)
#   - "session_history_controllers": dict mapping session_id -> SessionHistoryController
#
# Example structure:
# {
#   "hf_user123": {
#       "session_manager": SessionManager(...),
#       "session_history_controllers": {
#           "uuid-session-1": SessionHistoryController(...),
#           "uuid-session-2": SessionHistoryController(...)
#       }
#   },
#   "hf_user456": {
#       "session_manager": SessionManager(...),
#       "session_history_controllers": { ... }
#   }
# }
#
# Concurrency:
# ​​- Since the Spaces environment can handle multiple requests (multi-threading), 
#   locks are applied to prevent potential race conditions when accessing shared data structures (SESSION_STORES).
# - A global lock (RLock) and per-user locks are used in parallel to enable fine-grained critical section management.
# ---------------------------------------------------------------------------

class StoreDict(TypedDict):
    session_manager: SessionManager
    session_history_controllers: dict[str, SessionHistoryController]

SESSION_STORES: dict[str, StoreDict] = {}

# Concurrency locks
SESSION_LOCK = threading.RLock()
USER_LOCKS: dict[str, threading.RLock] = {}

def _get_user_lock(hf_user: str) -> threading.RLock:
    # Lazy-init per-user lock under global lock to avoid races on lock creation.
    with SESSION_LOCK:
        lock = USER_LOCKS.get(hf_user)
        if lock is None:
            lock = threading.RLock()
            USER_LOCKS[hf_user] = lock
        return lock


def init_user_store(hf_user: str, hf_token: str = HF_TOKEN):
    """Initialize user-specific store upon login.
    Creates a SessionManager and an empty dict of SessionHistoryControllers."""
    user_lock = _get_user_lock(hf_user)
    with user_lock:
        if hf_user not in SESSION_STORES:
            SESSION_STORES[hf_user] = {
                "session_manager": SessionManager(hf_token),
                "session_history_controllers": {}
            }
        return SESSION_STORES[hf_user]


def fetch_user_and_sessions(hf_token: gr.OAuthToken | None, lang="en"):
    """Resolve user identity, ensure store exists, hydrate sessions dropdown, and prepare initial status."""
    if hf_token is None:
        # No token → guest-like blank state on UI
        return gr.update(choices=[], value=None), gr.update(), gr.update(), gr.update()

    user_info = whoami(hf_token)
    hf_user = user_info["name"] if user_info else "_guest"

    store = get_account_store(hf_user)
    if store is None:
        store = init_user_store(hf_user, hf_token)
    else:
        store["session_manager"].hf_token = hf_token

    mgr = store["session_manager"]
    # List sessions from HF Hub (may return [] on first login)
    sessions = mgr.get_sessions() or mgr.list_sessions(hf_user)

    # Create a fresh session id (no sidebar update yet; title will be generated on first message)
    session_update = create_session(hf_user)
    # get_req_count uses hf_user to compute remaining quota (guest or logged-in)
    req_remains = get_req_count(hf_user)

    # status message configuration
    status_msg = translations[lang]["status"].format(login_status=hf_user, remains=req_remains)

    # Return order must match your UI bindings: (dropdown choices, status label, hidden_user, hidden_session_id)
    return (
        gr.update(choices=sessions, value=None),
        gr.update(value=status_msg),
        gr.update(value=hf_user),
        session_update  # already gr.update(value=session_id)
    )


def generate_session_id(history_controllers: dict[str, SessionHistoryController]) -> str:
    """Generate unique session ID."""
    while True:
        session_id = str(uuid4())
        if session_id not in history_controllers:
            return session_id


def create_session(hf_user: str):
    """Create a new session and return a Gradio update object with the new session_id.
    In the UI, when you click the new session button, the chat screen is only initialized, 
    and the title is created/registered when processing the first message."""
    if not hf_user:
        return gr.update(value="")

    user_lock = _get_user_lock(hf_user)
    with user_lock:
        store = get_account_store(hf_user)
        if store is None:
            # In case it was not initialized (defensive)
            store = init_user_store(hf_user)

        history_controllers = store["session_history_controllers"]
        session_id = generate_session_id(history_controllers)

        history_ctr = SessionHistoryController()
        store["session_history_controllers"][session_id] = history_ctr

        return gr.update(value=session_id)


def load_session(hf_user: str, session_id: str):
    """Session loading: Create a SessionHistoryController and inject data only when needed."""
    if not hf_user or not session_id:
        return gr.update(value=[]), gr.update()

    user_lock = _get_user_lock(hf_user)
    with user_lock:
        store = get_account_store(hf_user)
        if store is None:
            return gr.update(value=[]), gr.update()

        controller = get_session_controller(hf_user, session_id)
        if controller is None:
            records = store["session_manager"].download_session(hf_user, session_id)
            controller = SessionHistoryController(history=records)
            store["session_history_controllers"][session_id] = controller

        return gr.update(value=controller.get_full_history()), gr.update(value=session_id)


def get_account_store(hf_user: str):
    """View session store by user."""
    # Reading shared dict: protect with global lock to be consistent.
    with SESSION_LOCK:
        return SESSION_STORES.get(hf_user)


def get_session_controller(hf_user: str, session_id: str):
    """Session object lookup."""
    user_lock = _get_user_lock(hf_user)
    with user_lock:
        store = SESSION_STORES.get(hf_user)
        if not store:
            return None
        return store["session_history_controllers"].get(session_id)


def get_session_manager(hf_user: str):
    """Session manager query."""
    user_lock = _get_user_lock(hf_user)
    with user_lock:
        store = SESSION_STORES.get(hf_user)
        if not store:
            return None
        return store["session_manager"]


def list_sessions(hf_user: str):
    """View user session list."""
    user_lock = _get_user_lock(hf_user)
    with user_lock:
        store = SESSION_STORES.get(hf_user)
        if not store:
            return []
        return store["session_manager"].list_sessions(hf_user)


def finalize_session(hf_user: str, session_id: str):
    """Session End: Persist to HF Dataset and delete from memory.
    Concurrency safety: Protects session controller access/deletion with per-user locks."""
    user_lock = _get_user_lock(hf_user)
    with user_lock:
        store = SESSION_STORES.get(hf_user)
        if not store:
            return

        history_ctr = store["session_history_controllers"].get(session_id)
        if not history_ctr:
            return

        title_raw = history_ctr.get_session_title()
        timestamp_dt = history_ctr.get_last_request_time()  # Always returns a valid datetime (init at creation)
        timestamp = timestamp_dt.strftime("%Y%m%d%H%M%S")

        try:
            store["session_manager"].push_session(
                hf_user,
                session_id,
                history_ctr.get_full_history(),
                title_raw,
                timestamp,
            )
            # always upload a backup as well
            store["session_manager"].push_session(
                hf_user,
                session_id,
                history_ctr.get_full_history(),
                title_raw,
                timestamp,
                backup=True,
            )
        except Exception as e:
            logging.error(f"Error saving session {session_id} for user {hf_user}: {e}")

        # Protected deletion
        del store["session_history_controllers"][session_id]

        # If no more sessions for this user, clean up the store entry too
        if not store["session_history_controllers"]:
            # Remove user-specific lock together to avoid stale locks
            with SESSION_LOCK:
                del SESSION_STORES[hf_user]
                # best-effort: remove the user lock
                USER_LOCKS.pop(hf_user, None)


def get_expired_sessions():
    """Retrieve expired sessions (inactive for >= 2 hours).
    Returns a list of tuples: (hf_user, [expired_session_ids])."""
    now = datetime.now()
    results = []

    # Iterate users safely under global lock to snapshot keys
    with SESSION_LOCK:
        users = list(SESSION_STORES.keys())

    for hf_user in users:
        user_lock = _get_user_lock(hf_user)
        with user_lock:
            store = SESSION_STORES.get(hf_user)
            if not store:
                continue

            expired_sessions = [
                session_id
                for session_id, history_ctr in store["session_history_controllers"].items()
                # Note: history_ctr.get_last_request_time() is always set at init
                if now - history_ctr.get_last_request_time() >= timedelta(hours=2)
            ]
            if expired_sessions:
                results.append((hf_user, expired_sessions))

    return results
