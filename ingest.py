"""Corpus ingestion for arXiv cs.AI papers into FAISS."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import arxiv
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
INDEX_PATH = DATA_DIR / "index.faiss"
METADATA_PATH = DATA_DIR / "metadata.pkl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CATEGORY = "cs.AI"
MAX_PAPERS = 100
MIN_PAPERS = 50
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ARXIV_SORT = arxiv.SortCriterion.SubmittedDate


@dataclass
class PaperRecord:
    """Container for arXiv paper metadata."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: str
    pdf_path: Path | None


def ensure_dirs() -> None:
    """Create data directories required for ingestion outputs."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> dict[str, Any]:
    """Load manifest JSON if it exists, otherwise return defaults."""
    if not MANIFEST_PATH.exists():
        return {"paper_ids": [], "papers": []}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"paper_ids": [], "papers": []}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Persist manifest for idempotent ingestion runs."""
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fetch_recent_papers() -> list[arxiv.Result]:
    """Fetch recent cs.AI papers from the last 90 days."""
    date_floor = datetime.now(timezone.utc) - timedelta(days=90)
    query = f"cat:{CATEGORY}"
    # DECISION: Use explicit arXiv Client + Search flow for deterministic API usage and easier debugging.
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=400, sort_by=ARXIV_SORT)
    papers: list[arxiv.Result] = []
    for result in client.results(search):
        published = result.published.replace(tzinfo=timezone.utc)
        if published < date_floor:
            continue
        papers.append(result)
        if len(papers) >= MAX_PAPERS:
            break
    return papers


def download_paper(result: arxiv.Result) -> PaperRecord | None:
    """Download a paper PDF and normalize metadata."""
    paper_id = result.entry_id.rsplit("/", maxsplit=1)[-1]
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    try:
        if not pdf_path.exists():
            result.download_pdf(filename=str(pdf_path))
    except Exception as exc:
        print(f"PDF download failed for {paper_id}: {exc}")
        pdf_path = None
    return PaperRecord(
        paper_id=paper_id,
        title=result.title.strip(),
        authors=[author.name for author in result.authors],
        abstract=result.summary.strip(),
        published_date=result.published.date().isoformat(),
        pdf_path=pdf_path,
    )


def extract_text(pdf_path: Path | None) -> str:
    """Extract full text from PDF, returning empty string on failure."""
    if pdf_path is None:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if len(text) < 500:
            # Too short — likely a scanned PDF or extraction failure
            return ""
        return text
    except Exception:
        return ""


def chunk_text(text: str) -> list[str]:
    """Split text into retrieval chunks for embedding."""
    # DECISION: chunk_size=800 chosen for technical text — small enough for precise retrieval, large enough to preserve argument context. Overlap=100 preserves continuity at chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)


def load_existing_metadata() -> list[dict[str, Any]]:
    """Load existing metadata if available for idempotent rebuild."""
    if not METADATA_PATH.exists():
        return []
    try:
        with METADATA_PATH.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return []


def build_index(metadata: list[dict[str, Any]]) -> faiss.IndexFlatL2:
    """Embed chunks and build a FAISS L2 index."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [item["chunk_text"] for item in metadata]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def persist_outputs(index: faiss.IndexFlatL2, metadata: list[dict[str, Any]]) -> None:
    """Persist FAISS index and metadata artifacts."""
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("wb") as f:
        pickle.dump(metadata, f)


def main() -> None:
    """Run end-to-end corpus ingestion and persistence."""
    ensure_dirs()
    manifest = load_manifest()
    known_ids = set(manifest.get("paper_ids", []))
    existing_metadata = load_existing_metadata()
    results = fetch_recent_papers()
    new_metadata: list[dict[str, Any]] = []
    new_papers: list[dict[str, Any]] = []
    for result in results:
        record = download_paper(result)
        if not record or record.paper_id in known_ids:
            continue
        text = extract_text(record.pdf_path)
        if not text:
            # DECISION: When PDF extraction fails, construct a richer text
            # block from title + authors + abstract to maximize retrieval
            # signal from available metadata. This ensures every paper is
            # retrievable even without full text.
            authors_str = ", ".join(record.authors[:5])
            if len(record.authors) > 5:
                authors_str += f" et al. ({len(record.authors)} authors)"
            text = (
                f"Title: {record.title}\n"
                f"Authors: {authors_str}\n"
                f"Published: {record.published_date}\n"
                f"Abstract: {record.abstract}"
            )
            print(f"  [fallback] Using abstract for: {record.paper_id}")
        for idx, chunk in enumerate(chunk_text(text)):
            new_metadata.append(
                {
                    "paper_id": record.paper_id,
                    "title": record.title,
                    "authors": record.authors,
                    "abstract": record.abstract,
                    "chunk_index": idx,
                    "published_date": record.published_date,
                    "chunk_text": chunk,
                }
            )
        new_papers.append(
            {
                "paper_id": record.paper_id,
                "title": record.title,
                "authors": record.authors,
                "published_date": record.published_date,
            }
        )
    merged_metadata = existing_metadata + new_metadata
    if len(manifest.get("paper_ids", [])) < MIN_PAPERS and not new_papers:
        print("Warning: fewer than 50 papers available in manifest and no new papers found.")
    if merged_metadata:
        index = build_index(merged_metadata)
        persist_outputs(index, merged_metadata)
    manifest["paper_ids"] = list(dict.fromkeys(manifest.get("paper_ids", []) + [p["paper_id"] for p in new_papers]))
    manifest["papers"] = manifest.get("papers", []) + new_papers
    save_manifest(manifest)
    print(f"Ingested papers (new): {len(new_papers)}")
    print(f"Total papers tracked: {len(manifest['paper_ids'])}")
    print(f"Total chunks indexed: {len(merged_metadata)}")
    pdf_success = sum(
        1 for p in new_papers 
        if (PDF_DIR / f"{p['paper_id']}.pdf").exists()
    )
    print(f"PDFs successfully downloaded: {pdf_success}/{len(new_papers)}")
    print(f"Papers using abstract fallback: {len(new_papers) - pdf_success}/{len(new_papers)}")


if __name__ == "__main__":
    main()
