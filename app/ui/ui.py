# app/ui/ui.py
import gradio as gr

from app.runtime.request_limit import get_guest_req_count, get_req_count
from app.runtime.session_store import (create_session, fetch_user_and_sessions,
                                       load_session)
from app.service.request_service import process_request
from app.ui.i18n.translations import translations


# --- response handler ---
def on_login(hf_token: gr.OAuthToken ,lang):
    return fetch_user_and_sessions(hf_token, lang)

def on_new_session(hf_user):
    return create_session(hf_user)

def on_session_select(hf_user, session_id):
    return load_session(hf_user, session_id)

def on_message_submit(message, history, max_tokens, temperature, top_p,
                      lang, hf_token: gr.OAuthToken = None, hf_user=None, session_id=None):
    yield from process_request(message, history, max_tokens,temperature, 
                           top_p, lang, hf_token, hf_user, session_id)

def on_language_change(selected_lang, hf_user):
    remaining = get_req_count(hf_user)
    return (
        gr.update(placeholder=translations[selected_lang]["txtbox_placeholder"]),
        gr.update(value=translations[selected_lang]["status"].format(login_status=hf_user, remains=remaining)),
        gr.update(value=translations[selected_lang]["new_session"]), 
        gr.update(label=translations[selected_lang]["previous_sessions"]),
        gr.update(value=f"### {translations[selected_lang]['title']}\n"
                        f"{translations[selected_lang]['anon_limit']}\n"
                        f"{translations[selected_lang]['user_limit']}")
    )

# --- Create UI components ---
def render_sidebar(lang: str) -> tuple[gr.LoginButton, gr.Button, gr.Dropdown]:
    with gr.Sidebar():
        login_btn = gr.LoginButton()
        new_session_btn = gr.Button(translations[lang]["new_session"])
        session_selector = gr.Dropdown(label=translations[lang]["previous_sessions"], choices=[], interactive=True)
    return login_btn, new_session_btn, session_selector

def render_chat(lang):
    guest_req_remains = get_guest_req_count()

    chatbot = gr.Chatbot()
    
    status_display = gr.Markdown(value=translations[lang]["status"].format(login_status="guest", remains=guest_req_remains))

    msg = gr.Textbox(placeholder=translations[lang]["txtbox_placeholder"],
                     submit_btn=True, stop_btn=True)

    with gr.Accordion(label="Additional Inputs", open=False):
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")


    return chatbot, msg, max_tokens, temperature, top_p, status_display

def render_header(lang):
    with gr.Row():
        description = gr.Markdown(
            f"### {translations[lang]['title']}\n"
            f"{translations[lang]['anon_limit']}\n"
            f"{translations[lang]['user_limit']}"
        )

        lang_dropdown = gr.Dropdown(
            choices=[("🇰🇷 한국어", "ko"), ("🇺🇸 English", "en"),],
            value=None,
            label="🌐",
        )
            

    return description, lang_dropdown

# --- event connection ---
def bind_events(
    hidden_user: gr.Textbox, 
    hidden_session_id: gr.Textbox,
    language: gr.Textbox,
    description: gr.Markdown,
    lang_dropdown: gr.Dropdown, 
    chatbot: gr.Chatbot, 
    msg: gr.Textbox, 
    max_tokens: gr.Slider, 
    temperature: gr.Slider, 
    top_p: gr.Slider, 
    status_display: gr.Label,  
    login_btn: gr.LoginButton,  
    new_session_btn: gr.Button, 
    session_selector: gr.Dropdown, 
):

    login_btn.click(fn=on_login,
              inputs=[language],
              outputs=[session_selector, status_display, hidden_user, hidden_session_id])
    
    new_session_btn.click(fn=on_new_session,
                          inputs=[hidden_user],
                          outputs=[hidden_session_id])
    
    msg.submit(fn=on_message_submit,
               inputs=[msg, chatbot, max_tokens, temperature, top_p,
                       language, hidden_user, hidden_session_id],
               outputs=[chatbot, status_display, session_selector])

    session_selector.change(fn=on_session_select,
                            inputs=[hidden_user, session_selector],
                            outputs=[chatbot, hidden_session_id])
    
    lang_dropdown.change(fn=on_language_change,
                         inputs=[lang_dropdown, hidden_user],
                         outputs=[msg, status_display, new_session_btn, session_selector, description])
    
# --- Full UI configuration ---
def render_ui(hidden_user: gr.Textbox, hidden_session_id: gr.Textbox, language: gr.Textbox):
    lang_value = language.value or "en"
    
    description, lang_dropdown = render_header(lang_value)
    chatbot, msg, max_tokens, temperature, top_p, status_display = render_chat(lang_value)
    login_btn, new_session_btn, session_selector = render_sidebar(lang_value)

    bind_events(hidden_user, hidden_session_id, language, description, lang_dropdown,
                chatbot, msg, max_tokens, temperature, top_p, status_display,
                login_btn, new_session_btn, session_selector)
