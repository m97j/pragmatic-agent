# app/runtime/request_limit.py
from datetime import date

import gradio as gr

from app.ui.i18n.translations import translations

# --- Request count management ---
anon_request_count = 0
user_request_counts = {}
last_reset_date = date.today()

ANON_DAILY_LIMIT = 20
USER_DAILY_LIMIT = 15


def reset_counters():
    global anon_request_count, user_request_counts, last_reset_date
    today = date.today()
    anon_request_count = 0
    user_request_counts = {}
    last_reset_date = today

def check_limit(hf_token: gr.OAuthToken | None, hf_user, lang="en") -> tuple[bool, str, str]:
    global anon_request_count, user_request_counts
    today = date.today()
    if today != last_reset_date:
        reset_counters()

    if hf_token is None or not hf_token.token:
        if anon_request_count >= ANON_DAILY_LIMIT:
            msg = translations[lang]["status"].format(login_status="guest", remains=0)
            return False, msg, "guest"
        anon_request_count += 1
        remaining = ANON_DAILY_LIMIT - anon_request_count
        msg = translations[lang]['status'].format(login_status="guest", remains=remaining)
        return True, msg, "guest"
    else:
        user_id = hf_user if hf_user else (hf_token.user if hasattr(hf_token, "user") else hf_token.token[:8])
        count = user_request_counts.get(user_id, 0)
        if count >= USER_DAILY_LIMIT:
            msg = translations[lang]["status"].format(login_status=user_id, remains=0)
            return False, msg, user_id
        user_request_counts[user_id] = count + 1
        remaining = USER_DAILY_LIMIT - user_request_counts[user_id]
        msg = translations[lang]['status'].format(login_status=user_id, remains=remaining)
        return True, msg, user_id
    
def get_guest_req_count():
    return ANON_DAILY_LIMIT - anon_request_count

def get_req_count(hf_user):
    if hf_user == "_guest":
        return ANON_DAILY_LIMIT - anon_request_count
    else:
        return USER_DAILY_LIMIT - user_request_counts.get(hf_user)