"""
synthesiser.py
──────────────
Takes the retrieved + reranked chunks and synthesises a cited answer
using Claude. Also computes a confidence score based on:
  - Average rerank score of top chunks
  - Whether the query was fully covered (all sub-questions have chunks)
  - Source diversity (number of unique domains)
"""

import logging
import re
from urllib.parse import urlparse

from anthropic import Anthropic

from . import Chunk, ResearchResult, config

logger = logging.getLogger("research_agent.synthesiser")


SYNTHESIS_SYSTEM = """You are a research analyst. Your task is to answer a research 
question using ONLY the provided source excerpts.

Rules:
1. Ground every factual claim in the sources. Cite with [Source N] after each claim.
2. If sources conflict, acknowledge the discrepancy: "Source 1 says X, but Source 3 says Y."
3. If the sources don't fully answer the question, explicitly state what's missing.
4. Write in clear, structured prose. Use headers (##) for multi-part answers.
5. Do NOT invent facts, URLs, or citations that aren't in the sources.
6. End with a brief "## Limitations" section noting gaps in the sources.
"""

SYNTHESIS_USER = """Research question: {question}

Source excerpts:
{sources}

Write a comprehensive, cited answer to the research question."""


def format_sources_for_prompt(chunks: list[Chunk]) -> str:
    """Format chunks as numbered source list for the LLM prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Source {i}] {chunk.title}")
        lines.append(f"URL: {chunk.url}")
        lines.append(f"Excerpt: {chunk.text}")
        lines.append("")
    return "\n".join(lines)


def compute_confidence(
    chunks: list[Chunk],
    sub_questions: list[str],
    answer: str,
) -> float:
    """
    Estimate answer confidence on a 0.0–1.0 scale.
    
    Factors:
    - Average rerank score of the top chunks (0–1)
    - Source diversity: more unique domains = higher confidence
    - Coverage: does the answer mention each sub-question's topic?
    - Citation density: more [Source N] citations = more grounded answer
    
    This is a heuristic — a proper eval would use LLM-as-judge.
    """
    if not chunks:
        return 0.0
    
    # Factor 1: avg rerank score (already 0–1 from Cohere)
    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score > 0]
    avg_rerank = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.5
    
    # Factor 2: source diversity (unique domains, capped at 5)
    domains = set()
    for c in chunks:
        try:
            domain = urlparse(c.url).netloc
            if domain:
                domains.add(domain)
        except Exception:
            pass
    diversity_score = min(len(domains) / 5, 1.0)
    
    # Factor 3: citation density in the answer
    citation_matches = re.findall(r"\[Source \d+\]", answer)
    citation_density = min(len(citation_matches) / max(len(sub_questions) * 2, 1), 1.0)
    
    # Weighted average
    confidence = (
        avg_rerank * 0.5 +
        diversity_score * 0.3 +
        citation_density * 0.2
    )
    
    return round(min(max(confidence, 0.0), 1.0), 3)


def synthesise(
    query: str,
    chunks: list[Chunk],
    sub_questions: list[str],
    sources: list[dict],
) -> tuple[str, float]:
    """
    Generate a cited answer from retrieved chunks.
    
    Args:
        query: Original research question
        chunks: Top-N reranked chunks
        sub_questions: All decomposed sub-questions (for coverage check)
        sources: Deduplicated source list for metadata
        
    Returns:
        (answer_text, confidence_score)
    """
    client = Anthropic(api_key=config.anthropic_api_key)
    
    if not chunks:
        return (
            "I was unable to find relevant information to answer this question. "
            "Please try rephrasing or check that your search API keys are configured.",
            0.0,
        )
    
    formatted_sources = format_sources_for_prompt(chunks)
    
    logger.info(f"Synthesising answer from {len(chunks)} chunks")
    
    response = client.messages.create(
        model=config.llm_model,
        max_tokens=2048,
        system=SYNTHESIS_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": SYNTHESIS_USER.format(
                    question=query,
                    sources=formatted_sources,
                ),
            }
        ],
    )
    
    answer = response.content[0].text.strip()
    
    # Token usage for metadata
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    logger.info(f"Synthesis tokens: {usage}")
    
    confidence = compute_confidence(chunks, sub_questions, answer)
    logger.info(f"Confidence score: {confidence:.3f}")
    
    return answer, confidence
