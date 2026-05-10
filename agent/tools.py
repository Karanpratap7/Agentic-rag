"""External tools used by the agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import arxiv

TOOL_RESULT_TEMPLATE = (
    "Title: {title}\nAuthors: {authors}\nAbstract: {abstract}\nURL: {url}\n"
)
ABSTRACT_PREVIEW_CHARS = 300


def _format_result(result: arxiv.Result) -> str:
    """Format one arXiv result into a compact tool output block."""
    return TOOL_RESULT_TEMPLATE.format(
        title=result.title.strip(),
        authors=", ".join(author.name for author in result.authors),
        abstract=result.summary.strip()[:ABSTRACT_PREVIEW_CHARS],
        url=result.entry_id,
    )


def search_arxiv(query: str, max_results: int = 5, trace: list[dict[str, Any]] | None = None) -> str:
    """Search arXiv and return formatted paper snippets without raising exceptions."""
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        formatted = [_format_result(item) for item in arxiv.Client().results(search)]
        output = "\n---\n".join(formatted) if formatted else "No matching arXiv results found."
        if trace is not None:
            trace.append(
                {
                    "node": "search_arxiv",
                    "timestamp": datetime.now().isoformat(),
                    "decision": "tool_success",
                    "reasoning": "Used arXiv API to satisfy recency/specific metadata need.",
                    "input_preview": query[:100],
                    "output_preview": output[:100],
                }
            )
        return output
    except Exception as exc:
        error_output = f"arXiv tool error: {exc}"
        if trace is not None:
            trace.append(
                {
                    "node": "search_arxiv",
                    "timestamp": datetime.now().isoformat(),
                    "decision": "tool_error",
                    "reasoning": "Tool exceptions are contained to keep the agent resilient.",
                    "input_preview": query[:100],
                    "output_preview": error_output[:100],
                }
            )
        return error_output
