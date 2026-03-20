# 📘 pragmatic-agent


## Project Overview
This is an AI search and summary chatbot that provides **real-time search (RAG) + summary + translation** functions based on open-source LLMs for 10B to 20B levels.

- Users ask questions in natural language → the model translates them into search queries → the search results are synthesized to generate answers.
- Up-to-date information is supplemented with the **Web Search API (Google Custom Search)**.
- During the inference phase, the model uses a **prompt system** to separate the roles of the **planner (search decision/query rewrite)** and the **generator (final answer synthesis)**.

---

## Technology Stack
- **Model**
  - LLM: GPT-OSS 20B
  - Embedder: bge-small / e5-small
  - Reranker: monoT5-small / e5-reranker
- **Framework**: PyTorch, Transformers, TRL
- **Search**: FAISS + BM25, Google Custom Search API
- **Serving/Distribution**: Docker, **HF Space**, gradio

---

## Goals
1. Aim for large-scale model-level search accuracy with a medium-sized LLM.
2. Generate reliable/evidence-rich answers with RAG.
3. Ensure up-to-dateness with a real-time search API.
4. Optimize the prompt system, decoding policy, and RAG context design.
5. Demonstrate SaaS-based deployment with Hugging Face Space.
6. Explore scalability through research extensions (architecture level).

---

## Key Features
1. **Rate limit application/cache reuse**  
2. **LLM (planner prompt)**  
   - Analyzes question intent, evaluates the quality of local database search results, and generates 1-5 search queries if necessary.  
   - Recommends search for up-to-date/uncertain results.  
3. **Local search + web search API**  
   - Vector DB (BM25+Dense) → Top-k, calls Google Custom Search API if necessary.  
4. **Reranking**  
   - Top-50 → Top-5~10 with Cross Encoder  
5. **Context Composition**  
   - Normalization within budget for key phrases/titles/URLs/tokens  
6. **LLM (Generator Prompt)**  
   - Generates answers with summaries/key points/sources based on context  
   - Marks "Insufficient Information" when evidence is lacking  
7. **Multilingual Service**  
8. **Response Streaming/Meta Display**  
   - Remaining Search Counts/Source Links/Timestamps  
9. **Cache Storage/Log Collection**  

---

## Demo
👉[huggingface space](https://huggingface.co/spaces/m97j/pragmatic-agent)


