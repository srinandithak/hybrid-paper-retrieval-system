# Hybrid Paper Retrieval System

A research paper search and comparison tool that combines semantic search and keyword search to find relevant papers, then uses an LLM to generate citation-grounded comparisons across them.

**Live demo:** https://hybrid-paper-retrieval-system.vercel.app
**Backend API:** https://hybrid-paper-retrieval-system.onrender.com/docs

## What it does

- Ingests papers from arXiv and stores them in Supabase with vector embeddings
- Combines semantic search (OpenAI embeddings + pgvector) and keyword search (BM25) using Reciprocal Rank Fusion for better retrieval than either method alone
- `/compare` route retrieves relevant papers for a query and asks an LLM to generate a comparison, citing a specific paper for every claim
- Citations are validated in code against the actual retrieved papers — if the model cites something not in the retrieved set, the response is flagged (or retried once with a stricter prompt)

## Tech stack

- **Backend:** FastAPI, Python
- **Retrieval:** OpenAI embeddings, pgvector, `rank_bm25`, custom RRF fusion
- **LLM:** OpenAI (`gpt-4o-mini`) for comparison generation
- **Database:** Supabase (Postgres + pgvector)
- **Frontend:** React (Vite)
- **Deployment:** Render (backend), Vercel (frontend)

## Current scope

The paper corpus is currently focused on computer vision topics (object detection, vision transformers, CNNs). Queries far outside this scope will return an answer noting the papers don't cover it, since citations are always grounded in what's actually retrieved rather than the model's general knowledge.

## Known limitations / next steps

- BM25 index and paper corpus are rebuilt on every request — currently implementing Redis caching to avoid this
- No automated retrieval evaluation yet (planned: a small labeled query set to measure precision@k)
- Corpus currently covers a few hundred papers; more topic coverage planned
