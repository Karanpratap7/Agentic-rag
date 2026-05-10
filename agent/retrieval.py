"""Retrieval stack with query rewriting and FAISS lookup."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from agent.llm import build_chat_model

INDEX_PATH = Path("data/index.faiss")
METADATA_PATH = Path("data/metadata.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6
REWRITE_PROMPT = (
    "You are a retrieval optimization assistant. Rewrite the following user query "
    "to maximize recall from a corpus of arXiv AI research papers. Use precise technical "
    "terminology, expand acronyms, add relevant synonyms. Return ONLY the rewritten query, "
    "no explanation.\nOriginal query: {query}"
)


def _build_model():
    """Build OpenRouter model client for query rewriting."""
    return build_chat_model(temperature=0, streaming=False)


def _load_store() -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load persisted FAISS index and metadata."""
    index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("rb") as f:
        metadata = pickle.load(f)
    return index, metadata


import time

def rewrite_query(query: str) -> str:
    """Rewrite query for technical semantic retrieval quality."""
    # DECISION: Query rewriting chosen over hybrid search because the corpus
    # is homogeneous (arXiv CS papers) — keyword matching adds less value
    # than semantic precision. Rewriting improves recall by bridging informal
    # user phrasing to formal academic language.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            model = _build_model()
            return model.invoke(REWRITE_PROMPT.format(query=query)).content.strip()
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < max_attempts:
                wait = 5 * attempt
                print(f"[retrieval] Rate limited on rewrite "
                      f"(attempt {attempt}), waiting {wait}s...")
                time.sleep(wait)
            else:
                return query  # Fall back to original query on final failure
    return query


def retrieve_chunks(query: str, trace: list[dict[str, Any]], rewrite_enabled: bool = True) -> tuple[str, list[dict[str, Any]]]:
    """Rewrite, embed, and retrieve top-k metadata-enriched chunks."""
    try:
        rewritten = rewrite_query(query) if rewrite_enabled else query
        model = SentenceTransformer(EMBEDDING_MODEL)
        index, metadata = _load_store()
        query_vec = model.encode([rewritten], convert_to_numpy=True)
        distances, indices = index.search(np.asarray(query_vec, dtype=np.float32), TOP_K)
        docs = [metadata[i] for i in indices[0] if 0 <= i < len(metadata)]
        trace.append(
            {
                "node": "retrieve",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "decision": "retrieved_top_k",
                "reasoning": "Retrieved semantically nearest technical chunks from FAISS.",
                "input_preview": query[:100],
                "output_preview": str([d.get("title", "") for d in docs])[:100],
            }
        )
        return rewritten, docs
    except Exception as exc:
        trace.append(
            {
                "node": "retrieve",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "decision": "retrieval_error",
                "reasoning": "Exceptions are captured to prevent graph crash.",
                "input_preview": query[:100],
                "output_preview": str(exc)[:100],
            }
        )
        return query, []
