# app/app.py
import threading
import time

import gradio as gr
import schedule

from app.models.initializer import initialize_models, initialize_prefixes
from app.runtime.scheduler import (cleanup_expired_sessions,
                                   push_and_clear_interactions,
                                   reset_counters_per_day,
                                   upload_and_rotate_logs)
from app.ui.ui import render_ui


def run_scheduled_tasks():
    schedule.every().day.at("00:00").do(reset_counters_per_day)
    schedule.every().day.at("01:00").do(upload_and_rotate_logs)
    schedule.every().day.at("02:00").do(push_and_clear_interactions)
    schedule.every(1).hour.do(cleanup_expired_sessions)
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def build_ui():
    with gr.Blocks() as demo:
        # Hidden state
        hidden_user = gr.Textbox(value="_guest", visible=False)
        hidden_session_id = gr.Textbox(visible=False)
        language = gr.Textbox(value="en", visible=False, interactive=False, elem_id="lang-detect")

        # JS: detect browser language (navigator.languages[0] → Pass to hidden textbox)
        gr.HTML("""
        <script>
        const lang = navigator.languages ? navigator.languages[0] : navigator.language;
        const primary = lang.startsWith("ko") ? "ko" : "en";
        const el = document.getElementById("lang-detect");
        if (el) {
            el.value = primary;
            el.dispatchEvent(new Event("input"));
        }
        </script>
        """)
        # Initial rendering: default lang is "en"
        render_ui(hidden_user, hidden_session_id, language)

    return demo

if __name__ == "__main__":
    initialize_models() 
    initialize_prefixes()

    # Start scheduled tasks in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    scheduler_thread.start()

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
