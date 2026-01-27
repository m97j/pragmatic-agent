# 📘 pragmatic-llm-search - mvp

[![HF Space](https://img.shields.io/badge/HF%20Spaces(pragmatic--search)-Live-blue)](https://huggingface.co/spaces/m97j/pragmatic-llm-search)
[![HF Model](https://img.shields.io/badge/HF%20Model-pragmatic--search-ff69b4)](https://huggingface.co/m97j/pragmatic-search)
[![Colab](https://img.shields.io/badge/Colab-Notebook-yellow)](https://colab.research.google.com/drive/1FT-PxsCi1FDLMmCELw3S0_gHHNrwNYN0?usp=sharing)

## Project Overview
This is an AI search and summary chatbot that provides **real-time search (RAG) + summary + translation** functions based on open-source LLMs for 7B to 14B levels.

- Users ask questions in natural language → the model translates them into search queries → the search results are synthesized to generate answers.
- Up-to-date information is supplemented with the **Web Search API (Google Custom Search)**.
- During the inference phase, the model uses a **prompt system** to separate the roles of the **planner (search decision/query rewrite)** and the **generator (final answer synthesis)**.

---

## Goals
1. Aim for large-scale model-level search accuracy with a medium-sized LLM (7B to 14B).
2. Generate reliable/evidence-rich answers with RAG + RLHF (DPO).
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

## Technology Stack
- **Model**
  - LLM: Qwen 3 8B (Main LLM, QLoRA SFT + DPO)
  - Embedder: bge-small / e5-small
  - Reranker: monoT5-small / e5-reranker
- **Framework**: PyTorch, Transformers, PEFT, TRL
- **Search**: FAISS/Weaviate + BM25, Google Custom Search API...
- **Serving/Distribution**: Docker, **HF Space**, gratio

---

## Demo
👉[huggingface space](https://huggingface.co/spaces/m97j/pragmatic-llm-search)
