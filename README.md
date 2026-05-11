# Agentic RAG for arXiv cs.AI

Production-grade agentic RAG assistant for recent arXiv AI papers using LangGraph, FAISS, local embeddings, and OpenRouter-hosted Nemotron reasoning models.

## Setup

1. Clone and enter the project:
   - `git clone <repo-url>`
   - `cd agentic-rag`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment:
   - `cp .env.example .env`
   - Set `OPENROUTER_API_KEY` in `.env`
4. Ingest corpus:
   - `python ingest.py`
5. Run app:
   - `streamlit run app.py`

## Architecture

```text
User Query
   |
   v
[classify_intent] -----> [ask_clarification] -> END
   |          |    |      [refuse] -----------> END
   |          |    +----> [call_arxiv_tool] --+
   |          |                               |
   v          +----> [answer_from_memory] --> [generate_answer]
[rewrite_query]                                      |
   |                                                 v
   v                                          [update_memory] -> END
[check_context] --> [generate_answer]
   |
   +--> [ask_clarification] -> END
   +--> [call_arxiv_tool] ---> [generate_answer]
```

## Decisions Log

**LLM choice (Nemotron via OpenRouter)**  
The primary model is configured through OpenRouter (default: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`) for straightforward model routing and key management via a single API. The active model can be changed with `OPENROUTER_MODEL` in `.env` without code changes.

**Embedding choice (all-MiniLM-L6-v2)**  
`all-MiniLM-L6-v2` is local, fast, and cost-free at inference time, which keeps retrieval reproducible and assignment-friendly. It balances semantic quality and speed well for medium-length technical chunks.

**Vector store choice (FAISS)**  
FAISS `IndexFlatL2` was selected for deterministic similarity search, simple disk persistence, and straightforward integration with local embeddings. For this assignment scale, flat search avoids ANN complexity while maintaining predictable behavior.

**Framework choice (LangGraph over AgentExecutor)**  
LangGraph is used for an explicit state machine with auditable branching (`retrieve`, `tool`, `clarify`, `refuse`, `answer_from_memory`). This improves control, observability, and failure handling compared with implicit planner loops.

**Retrieval technique (query rewriting)**  
Query rewriting is performed before embedding and search to map informal user language into paper-native technical terminology. This is more defensible than hybrid lexical retrieval for a homogeneous research corpus where semantic alignment is the primary bottleneck.

**Memory design (buffer + summary, why not semantic)**  
Memory architecture: We implement conversation memory (recent turns buffer) and episodic memory (summary compression). We deliberately omit semantic memory (long-term user/world facts) because the system is stateless between sessions and the corpus is the authoritative knowledge source — user-specific facts would not improve retrieval quality.

**Chunking strategy (800/100, why)**  
Chunks use 800 characters with 100 overlap to preserve enough argument continuity for technical text while still allowing precise retrieval granularity. Overlap protects against boundary loss for equations, definitions, and method details split across pages.

**Rate limit optimization (free-tier LLM calls)**  
The free-tier OpenRouter models allow ~10-20 requests/minute shared across all users. The original design made 4 LLM calls per turn (classify → rewrite → check_context → generate), which hit limits after 2-3 questions. Optimizations applied: (1) check_context replaced with keyword-overlap heuristic, saving 1 LLM call per turn; (2) rewrite step skipped for already-technical queries; (3) doc context truncated from 6×500 to 4×300 chars; (4) prompt templates shortened by ~40%. Net result: 2-3 LLM calls per turn on technical queries, 3 on vague ones.

## Failure Modes Observed During Testing

- Retrieval may be sparse when PDFs have poor extractable text quality.
- Intent classification can occasionally over-route broad queries to `retrieve` instead of `clarify`.
- Tool outputs are concise previews and can miss important details beyond the first abstract characters.
- arXiv tool calls occasionally return HTTP 500 errors from the arXiv 
API. The system handles this by falling back to corpus retrieval, 
so the user receives a degraded but valid answer rather than an error.
- Full-text PDF extraction success rate varies by paper. When PDFs 
fail to parse (scanned documents, download errors), the system falls 
back to title + abstract chunks. This limits answer depth for synthesis 
questions but preserves basic retrievability for all indexed papers. 
Run `python ingest.py` output shows the PDF success rate for your corpus.

## What I Would Do With Another Week

**1. Citation-grounded answer verification**  
The current generate_answer node produces answers that cite paper titles 
inline but doesn't verify the citations are accurate — the LLM can 
hallucinate title-content mismatches. I'd add a post-generation 
verification step that re-retrieves the cited chunk and checks semantic 
overlap between the claim and the source using cosine similarity. Answers 
below a threshold would be flagged with a confidence warning.

**2. Reranking layer on top of query rewriting**  
Query rewriting improves recall but not necessarily precision. Adding a 
cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) on 
the top-6 retrieved chunks before generation would improve answer quality 
measurably — I have the ablation framework already in place to demonstrate 
the improvement with eval scores.

**3. Persistent session memory across app restarts**  
Currently all conversation state lives in `st.session_state` and is lost 
on page refresh. I'd add SQLite-backed session persistence so users can 
resume conversations and the agent can reference prior research sessions. 
This also enables longitudinal eval — tracking how answer quality changes 
as the user builds context.

**4. Retrieval diagnostics panel**  
The trace viewer in the sidebar shows raw JSON. I'd replace this with a 
structured diagnostics view: query rewrite diff side-by-side, top-k chunk 
distances as a bar chart, and a hit distribution across papers showing 
which papers are being retrieved most often. This would surface corpus 
coverage gaps immediately.

**5. Contradiction detection as a first-class node**  
The current check_context node routes "contradictory" retrievals to 
generate_answer with a warning flag. I'd make contradiction detection 
its own LangGraph node with a dedicated prompt that identifies which 
specific claims conflict and presents them explicitly to the user, rather 
than deferring to the generation model to handle it inline.

## Known Limitations

- Corpus is limited to ingested cs.AI papers from the last 90 days and is only as current as the last ingestion run.
- FAISS flat index is memory-heavy as corpus grows.
- No multi-session persistence beyond trace logs and stored index artifacts.
- No authentication or multi-user isolation by design (assignment constraint).
