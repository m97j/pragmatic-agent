# app/service/main_pipeline.py
import spaces

from app.models.service.llm_service import LLMService
from app.modules.clients.rag_client import rag_search
from app.modules.clients.search_client import search_with_api
from app.modules.processing.context_refiner import refine_results
from app.modules.processing.postprocess import finalize_answer
from app.modules.processing.prompt_builder import (build_generation_prompt,
                                                   build_planning_prompt)
from app.modules.processing.query_processor import process_queries
from app.runtime.session_store import get_session_controller


@spaces.GPU
def run_pipeline(message, history, max_tokens, temperature, top_p, hf_token, hf_user=None, session_id=None):

    llm = LLMService()

    # 1. execute RAG search first
    rag_results = rag_search(message)

    # Configure the RAG context string (empty string if none)
    rag_context = "\n".join([r.get("snippet", "") for r in rag_results]) if rag_results else ""


    # 2. build planner prompt (with RAG context)
    planning_prompt = build_planning_prompt(message, rag_context=rag_context)


    # 3. Call LLM planner → Generate search query candidates
    raw_queries = llm.generate(
        prompt=planning_prompt,
        mode="instruct",
        strategy="sampling",
        max_tokens=100,
        beam_width=3,
        stream=False,
    )

    # 3.1 Parse queries
    queries = process_queries(raw_queries)


    # 4. External search (only when there is a query)
    if queries and any(q.strip() for q in queries):
        search_results = search_with_api(message, queries)
    else:
        # Skip search
        search_results = []

    # 4.1 Search Results Refinement
    # refined_results = refine_results(search_results, message)
    refined_results = search_results

    # 5. Results consolidation + reranking
    combined = (refined_results or []) + (rag_results or [])
    # reranked = rerank_results(combined, query=message)


    # 6. Create a final answer prompt (history is only used for answer prompts)
    generation_prompt = build_generation_prompt(message, combined, history=history, hf_user=hf_user, session_id=session_id)


    # 7. final generation (user-facing, streaming)
    generated_answer = ""

    for token in llm.generate(
        prompt=generation_prompt,
        mode="think",
        strategy="sampling",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        stream_chunk="delta",
    ):
        # save tokens 
        generated_answer += token
        yield token

    # 7.1 Save to conversation history if applicable
    controller = get_session_controller(hf_user, session_id)

    if controller is not None:
        controller.append_message("user", message)
        controller.append_message("assistant", generated_answer)


    # 8. post processing
    finalize_answer(message, combined, generated_answer, hf_token)
