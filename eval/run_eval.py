"""Evaluation harness for agent behavior and retrieval ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_graph

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_PATH = Path("eval/results.json")
ABLATION_IDS = {1, 5, 9}


def load_questions() -> list[dict[str, Any]]:
    """Load evaluation questions from disk."""
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def keyword_score(answer: str, expected_keywords: list[str]) -> int:
    """Count expected keywords found in answer text."""
    lower_answer = answer.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in lower_answer)


def run_question(graph, question: dict[str, Any], rewrite_enabled: bool) -> dict[str, Any]:
    """Execute one question through graph and collect scoring signals."""
    state = {"messages": [], "query": question["question"], "trace": [], "summary": "", "turn_count": 1, "rewrite_enabled": rewrite_enabled}
    result = graph.invoke(state)
    answer = result.get("answer", "")
    decision = result.get("decision", "")
    return {
        "id": question["id"],
        "question": question["question"],
        "expected_behavior": question["expected_behavior"],
        "decision": decision,
        "decision_correct": decision == question["expected_behavior"],
        "keywords_found": keyword_score(answer, question.get("expected_keywords", [])),
        "rewrite_enabled": rewrite_enabled,
    }


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    """Print a compact text table of evaluation rows."""
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        print(
            f"Q{row['id']:02d} | expected={row['expected_behavior']:<18} got={row['decision']:<18} "
            f"correct={str(row['decision_correct']):<5} keywords={row['keywords_found']}"
        )


def run_ablation(graph, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run retrieval ablation comparing query rewriting ON vs OFF."""
    # ABLATION: This demonstrates query rewriting actually improves retrieval — not just theoretical benefit
    summary = []
    selected = [q for q in questions if q["id"] in ABLATION_IDS]
    for question in selected:
        res_on = run_question(graph, question, rewrite_enabled=True)
        res_off = run_question(graph, question, rewrite_enabled=False)
        delta = res_on["keywords_found"] - res_off["keywords_found"]
        
        print(f"Q{question['id']} | {question['question'][:40]:<40} | rewrite=ON  decision={res_on['decision']:<15} correct={res_on['decision_correct']}")
        print(f"Q{question['id']} | {question['question'][:40]:<40} | rewrite=OFF decision={res_off['decision']:<15} correct={res_off['decision_correct']}")
        print("-------")
        print(f"Delta: keywords_found ON={res_on['keywords_found']} OFF={res_off['keywords_found']} improvement={delta}\n")
        
        summary.append({
            "question_id": question["id"],
            "rewrite_on": res_on,
            "rewrite_off": res_off,
            "keyword_delta": delta
        })
    return summary


def main() -> None:
    """Run full evaluation and ablation then persist JSON results."""
    graph = build_graph()
    questions = load_questions()
    base_rows = [run_question(graph, q, rewrite_enabled=True) for q in questions]
    print_table(base_rows, "Evaluation Results")
    print("\nAblation Results (Rewrite ON vs OFF)")
    print("=" * 40)
    ablation_summary = run_ablation(graph, questions)
    
    payload = {"results": base_rows, "ablation_summary": ablation_summary}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
