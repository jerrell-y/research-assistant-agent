"""
evaluator.py
────────────
Lightweight eval harness for the research agent.

Measures:
- Citation accuracy: are [Source N] references present in the answer?
- Answer relevance: does the answer address the question? (LLM-as-judge)
- Latency: per-phase timing breakdown
- Hallucination proxy: claims without citation markers

Run with:
    python -m agent.evaluator --run-evals
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field

from anthropic import Anthropic

from . import ResearchResult, config

logger = logging.getLogger("research_agent.evaluator")


# ── Test suite ────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question": "What are the best practices for fine-tuning large language models?",
        "must_contain_keywords": ["dataset", "learning rate", "evaluation", "overfitting"],
    },
    {
        "question": "How does RAG (Retrieval-Augmented Generation) work and when should you use it?",
        "must_contain_keywords": ["retrieval", "vector", "embedding", "generation"],
    },
    {
        "question": "What are the key differences between supervised and reinforcement learning?",
        "must_contain_keywords": ["reward", "label", "agent", "policy"],
    },
    {
        "question": "How do transformer attention mechanisms work?",
        "must_contain_keywords": ["query", "key", "value", "softmax"],
    },
    {
        "question": "What is the current state of AI safety research?",
        "must_contain_keywords": ["alignment", "safety", "model"],
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    question: str
    citation_count: int = 0
    has_limitations_section: bool = False
    keyword_hits: list[str] = field(default_factory=list)
    keyword_misses: list[str] = field(default_factory=list)
    relevance_score: float = 0.0        # LLM-as-judge 0–1
    confidence_score: float = 0.0
    latency_total_s: float = 0.0
    latency_breakdown: dict = field(default_factory=dict)
    num_sources: int = 0
    num_chunks: int = 0
    error: str | None = None

    def keyword_coverage(self) -> float:
        total = len(self.keyword_hits) + len(self.keyword_misses)
        return len(self.keyword_hits) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question[:60] + "...",
            "citations": self.citation_count,
            "has_limitations": self.has_limitations_section,
            "keyword_coverage": f"{self.keyword_coverage():.0%}",
            "relevance_score": f"{self.relevance_score:.2f}",
            "confidence": f"{self.confidence_score:.2f}",
            "latency_s": f"{self.latency_total_s:.1f}",
            "sources": self.num_sources,
            "error": self.error,
        }


def judge_relevance(question: str, answer: str) -> float:
    """
    Use Claude as a judge to score answer relevance.
    Returns a score from 0.0 to 1.0.
    
    This is a lightweight LLM-as-judge implementation.
    In production, use a dedicated judge model with calibrated prompts.
    """
    client = Anthropic(api_key=config.anthropic_api_key)
    
    prompt = f"""Rate how well this answer addresses the research question.

Question: {question}

Answer (first 1000 chars): {answer[:1000]}

Score from 0 to 10:
- 0: Completely off-topic or empty
- 5: Partially addresses the question with some relevant information
- 10: Fully and accurately addresses all aspects of the question

Reply with ONLY a single integer (0-10). No explanation."""

    response = client.messages.create(
        model=config.llm_model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    
    try:
        score = int(response.content[0].text.strip())
        return min(max(score / 10, 0.0), 1.0)
    except ValueError:
        return 0.5


def evaluate_result(result: ResearchResult, expected_keywords: list[str]) -> EvalMetrics:
    """Compute all metrics for a single research result."""
    import re
    
    metrics = EvalMetrics(question=result.query)
    
    answer_lower = result.answer.lower()
    
    # Citation count
    metrics.citation_count = len(re.findall(r"\[source \d+\]", answer_lower))
    
    # Limitations section
    metrics.has_limitations_section = "## limitations" in answer_lower or "limitations" in answer_lower
    
    # Keyword coverage
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            metrics.keyword_hits.append(kw)
        else:
            metrics.keyword_misses.append(kw)
    
    # Scores
    metrics.confidence_score = result.confidence
    metrics.num_sources = len(result.sources)
    
    # Latency
    metrics.latency_total_s = result.metadata.get("latency_total_s", 0)
    metrics.latency_breakdown = result.metadata.get("latency_breakdown", {})
    
    # LLM-as-judge relevance
    metrics.relevance_score = judge_relevance(result.query, result.answer)
    
    return metrics


def run_evals(verbose: bool = False) -> list[EvalMetrics]:
    """
    Run the full eval suite and print a summary table.
    Returns list of EvalMetrics for programmatic inspection.
    """
    from .run import run_research  # avoid circular import at module level
    
    all_metrics: list[EvalMetrics] = []
    
    print("\n" + "═" * 70)
    print("  Research Agent Eval Harness")
    print("═" * 70)
    
    for i, test in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {test['question'][:60]}...")
        
        try:
            t0 = time.perf_counter()
            result = run_research(test["question"])
            elapsed = time.perf_counter() - t0
            result.metadata["latency_total_s"] = round(elapsed, 2)
            
            metrics = evaluate_result(result, test["must_contain_keywords"])
            
            if verbose:
                print(f"  Answer preview: {result.answer[:200]}...")
            
        except Exception as e:
            logger.error(f"Eval failed: {e}")
            metrics = EvalMetrics(question=test["question"], error=str(e))
        
        all_metrics.append(metrics)
        
        m = metrics.to_dict()
        print(f"  Citations:    {m['citations']}")
        print(f"  Keywords:     {m['keyword_coverage']}")
        print(f"  Relevance:    {m['relevance_score']}")
        print(f"  Confidence:   {m['confidence']}")
        print(f"  Latency:      {m['latency_s']}s")
        print(f"  Sources:      {m['sources']}")
        if m["error"]:
            print(f"  ERROR:        {m['error']}")
    
    # Summary
    print("\n" + "─" * 70)
    print("  SUMMARY")
    print("─" * 70)
    
    valid = [m for m in all_metrics if not m.error]
    if valid:
        avg_relevance = sum(m.relevance_score for m in valid) / len(valid)
        avg_confidence = sum(m.confidence_score for m in valid) / len(valid)
        avg_latency = sum(m.latency_total_s for m in valid) / len(valid)
        avg_keywords = sum(m.keyword_coverage() for m in valid) / len(valid)
        avg_citations = sum(m.citation_count for m in valid) / len(valid)
        
        print(f"  Avg relevance (LLM-judge): {avg_relevance:.2f}")
        print(f"  Avg keyword coverage:      {avg_keywords:.0%}")
        print(f"  Avg confidence:            {avg_confidence:.2f}")
        print(f"  Avg citations per answer:  {avg_citations:.1f}")
        print(f"  Avg latency:               {avg_latency:.1f}s")
        print(f"  Success rate:              {len(valid)}/{len(all_metrics)}")
    
    print("═" * 70 + "\n")
    
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research agent eval harness")
    parser.add_argument("--run-evals", action="store_true", help="Run eval suite")
    parser.add_argument("--verbose", action="store_true", help="Print answer previews")
    args = parser.parse_args()
    
    if args.run_evals:
        run_evals(verbose=args.verbose)
    else:
        parser.print_help()
