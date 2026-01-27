# app/modules/conversation/history_controller.py
from datetime import datetime

from app.models.reranker_model import rerank_results
from app.models.service.llm_service import LLMService
from app.modules.common.utils import split_content


class SessionHistoryController:
    def __init__(self, history=None, max_items=100, time=None):
        self.max_items = max_items
        self.title = None
        self.timestamp = time or datetime.now()
        if history:
            self.history = history[-max_items:]
        else:
            self.history = []
        self.llm = LLMService()

    def append_message(self, role, content):
        """
        Append a new message to history.
        role: "user" or "assistant"
        content: string
        """
        self.history.append({"role": role, "content": content})
        # Keep only the last max_items
        if len(self.history) > self.max_items:
            self.history = self.history[-self.max_items:] 

        self.timestamp = datetime.now()

    def get_recent(self, n=5):
        """
        Get the last n messages.
        """
        return self.history[-n:]

    def summarize_history(self, context, max_tokens=512):
        """
        Optional: Summarize long history for compact context.
        Could call LLM summarizer here.
        """
        # Placeholder: join contents
        prompt = f"Summarize the following conversation history for context:\n{context}"
        summary = self.llm.generate(
            prompt=prompt,
            mode="instruct",
            strategy="sampling",
            max_tokens=max_tokens//2,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        # return " ".join([h["content"] for h in self.history])
        return summary
    
    def clear_history(self):
        """ Clear the conversation history. """
        self.history = []
    
    def get_full_history(self):
        """ Get the full conversation history. """
        return self.history
    
    def build_prompt_history(self, user_message, max_tokens=1024, top_k=10):
        """
        Build the history context for generation prompt:
        - Split into sentences
        - Rerank by relevance to current user_message
        - Select top_k
        - If length exceeds max_tokens, summarize with LLM
        """
        # 1. Flatten history into snippets
        snippets = []
        for h in self.history:
            snippets.extend(split_content(h["content"], h["role"], return_boundaries=False))

        # 2. Rerank snippets
        reranked = rerank_results(
            [{"snippet": s} for s in snippets],
            query=user_message
        )
        top_snippets = [r["snippet"] for r in reranked[:top_k]]

        # 3. Join into context string
        context = "\n".join(top_snippets)

        # 4. Length check
        if len(context.split()) > max_tokens:
            return self.summarize_history(context, max_tokens=max_tokens)
        return context
    
    def get_session_title(self):
        """ Get the session title. """
        return self.title

    def get_last_request_time(self):
        """ Get the timestamp of the last message in history. """
        return self.timestamp

    def generate_session_title(self, role, content):
        """ Generate a session title based on the first user message. """
        if role == "user" and not self.title:
            self.title = content[:30] + "..." if len(content) > 30 else content  
        return self.title