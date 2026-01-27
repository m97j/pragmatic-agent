# app/service/request_service.py
import gradio as gr

from app.config import HF_TOKEN
from app.runtime.request_limit import check_limit
from app.runtime.session_store import (get_session_controller,
                                       get_session_manager)
from app.service.main_pipeline import run_pipeline


def generate_response(
    message: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    hf_token: str,
    status_msg: str,
    hf_user: str = None,
    session_id: str = None,
):
    """
    generate response and yield partial results for streaming
    1. run main pipeline
    2. update history in session controller if applicable
    3. yield updated chatbot and status message
    Args:
        message (str): user message
        history (list[dict]): conversation history (UI-only during streaming)
        max_tokens (int): max tokens for generation
        temperature (float): temperature for generation
        top_p (float): top-p for generation
        hf_token (str): Hugging Face token
        status_msg (str): status message to display
        hf_user (str, optional): Hugging Face user identifier. Defaults to None.
        session_id (str, optional): session identifier. Defaults to None.
    Yields:
        tuple: (updated chatbot, status message, optional dropdown update)
    """
    response = ""
    his_ctr = get_session_controller(hf_user, session_id)

    # Pass "history_ctr" sentinel if logged-in to let downstream user relevant history only
    history_for_prompt = "history_ctr" if his_ctr else history

    for partial in run_pipeline(
        message=message,
        history=history_for_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        hf_token=hf_token,
        hf_user=hf_user,
        session_id=session_id,
    ):
        response += partial
        # UI-only streaming update (do NOT write to his_ctr here)
        if history and history[-1]["role"] == "assistant":
            # Update the last assistant message content
            history[-1]["content"] = response
        else:
            # First token: create assistant message
            history.append({"role": "assistant", "content": response})

        yield gr.update(value=history), gr.update(value=status_msg), gr.update()

    # No persistence here; run_pipeline already saved final user/assistant.


def process_request(
    message,
    history,
    max_tokens,
    temperature,
    top_p,
    lang,
    hf_token=None,
    hf_user=None,
    session_id=None,
):
    allowed, status_msg, hf_user = check_limit(hf_token, hf_user, lang)
    if not allowed:
        yield gr.update(value=history + [{"role": "system", "content": status_msg}]), gr.update(value=status_msg), gr.update()
        return

    session_mgr = get_session_manager(hf_user)
    history_ctr = get_session_controller(hf_user, session_id) if session_id else None

    if history_ctr:
        ui_history = history_ctr.get_full_history() + [{"role": "user","content": message}]
        if history_ctr.get_session_title() is None:
            session_title = history_ctr.generate_session_title("user", message)
            session_mgr.add_session(session_title, session_id)
            sessions = session_mgr.get_sessions()
            yield gr.update(value=ui_history), gr.update(value=status_msg), gr.update(choices = sessions, value=session_id)
        else:
            yield gr.update(value=ui_history), gr.update(value=status_msg), gr.update()
    else:
        history.append({"role": "user", "content": message})
        ui_history = history
        yield gr.update(value=ui_history), gr.update(value=status_msg), gr.update()

    yield from generate_response(
        message=message,
        history=ui_history,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        hf_token=hf_token.token if hf_token else HF_TOKEN,
        status_msg=status_msg,
        hf_user=hf_user,
        session_id=session_id,
    )