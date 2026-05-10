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
[classify_intent] ---> [ask_clarification] -> END
   |                     [refuse] ----------> END
   |                     [call_arxiv_tool] -> [generate_answer]
   v
[rewrite_query] -> [retrieve] -> [check_context] -> [generate_answer]
                                   |                     |
                                   +--> clarify/tool ----+
                                                         v
                                                   [update_memory] -> END
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

## Failure Modes Observed During Testing

- Retrieval may be sparse when PDFs have poor extractable text quality.
- Intent classification can occasionally over-route broad queries to `retrieve` instead of `clarify`.
- Tool outputs are concise previews and can miss important details beyond the first abstract characters.
- Streaming fallback degrades to non-LLM token simulation if API streaming fails.

## What I Would Do With Another Week

- Add citation-grounded answer verification and contradiction detection with confidence scoring.
- Add retrieval diagnostics dashboard (query rewrite diff, distance histograms, hit distribution).
- Improve PDF parsing robustness with layout-aware extraction and OCR fallback.
- Add regression eval suites with behavior drift alerts over new ingestions.
- Implement stronger prompt+policy testing around refusal and clarification boundaries.

## Known Limitations

- Corpus is limited to ingested cs.AI papers from the last 90 days and is only as current as the last ingestion run.
- FAISS flat index is memory-heavy as corpus grows.
- No multi-session persistence beyond trace logs and stored index artifacts.
- No authentication or multi-user isolation by design (assignment constraint).
