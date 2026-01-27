# app/modules/processing/prompt_builder.py
from app.runtime.session_store import get_session_controller


def build_planning_prompt(message, rag_context=""):
    """
    Planner prompt: Always include the RAG context.
    - If rag_context is empty, an empty string is used.
    - 1st: Unconditionally generate 1 to 5 queries.
    - 2nd: Dynamically control 0 to 5 queries after FT.
    """

    system_message = (
        "You are a planning assistant. Your role is to generate search queries "
        "based on the user message and RAG context."
    )

    return (
        f"{system_message}\n"
        f"User question (multilingual): {message}\n\n"
        f"Context from RAG DB (may be empty):\n{rag_context}\n\n"
        "Task:\n"
        "- Generate 1 to 5 concise search queries.\n"
        "- If the RAG context already fully answers the question, you may generate 0 queries.\n"
        "- Prefer official documentation, standards, academic or authoritative sources.\n"
        "- Prefer English queries for global/academic sources.\n"
        "- If the user question is strongly local or language-specific, also generate queries in the same language.\n"
        "- Avoid redundant queries; use broader queries if they cover multiple facets.\n"
        "- Output only the queries, one per line."
    )


def build_generation_prompt(message, docs, history=None, hf_user=None, session_id=None, top_k=5):
    """
    Answer generation prompt with context from retrieved documents.
    - Include up to top_k document snippets as context.
    - Include recent conversation history if available.
    - System message is fixed internally for answer generation role.
    """

    system_message = (
        "You are an answer generation assistant. Your role is to synthesize "
        "information from context and produce a clear, concise, Markdown-formatted answer "
        "with citations in (title)[url] format."
    )

    # Sort by score if available, then take Top-K
    sorted_docs = sorted(docs or [], key=lambda d: d.get("score", 0), reverse=True)
    top_docs = sorted_docs[:top_k]

    # Build structured context string
    context_entries = []
    for d in top_docs:
        q = d.get("query", "")
        title = d.get("title", "")
        url = d.get("url", "")
        snippet = d.get("snippet") or d.get("summary") or d.get("content") or ""
        score = d.get("score", 0)
        entry = (
            f"Query: {q}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}\n"
            f"Score: {score}\n"
        )
        context_entries.append(entry.strip())

    context = "\n---\n".join(context_entries)

    # Conversation history (optional)
    history_text = ""
    if history:
        if history == "history_ctr":
            history_controller = get_session_controller(hf_user, session_id)
            try:
                ctx = history_controller.build_prompt_history(message, max_tokens=1024, top_k=10)
                history_text = "\nConversation history (relevant with user message):\n" + ctx
            except Exception:
                history_text = ""
        else:
            try:
                last_items = history[-4:] if isinstance(history, list) else [str(history)]
                history_text = "\nConversation history (recent):\n" + "\n".join(f"{r}: {c}" for r, c in last_items)
            except Exception:
                history_text = ""

    # Final prompt assembly
    return (
        f"{system_message}\n"
        f"{history_text}\n"
        "Context (structured search results):\n"
        f"{context}\n\n"
        "Instructions:\n"
        "- Provide the answer to the user message below using the context above.\n"
        "- Answer in the same language as the user message.\n"
        "- Follow Markdown format strictly.\n"
        "- Include citations using (title)[url] format whenever referencing sources.\n"
        "- Use the score field to prioritize higher relevance results first.\n"
        "- Cross-check and summarize multiple sources when possible.\n"
        "- If context is insufficient, state limitations clearly.\n\n"
        f"User message: {message}\n"
        "Answer:"
    )
