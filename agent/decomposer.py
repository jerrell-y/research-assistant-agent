"""
decomposer.py
─────────────
Uses the LLM to break a research question into 3-5 specific sub-questions.
This is the planning step — more targeted sub-questions lead to better
search results and a more complete final answer.
"""

import json
import logging
import re
from anthropic import Anthropic
from . import config

logger = logging.getLogger("research_agent.decomposer")

DECOMPOSE_SYSTEM = """You are a research planning expert. Your job is to break down 
a complex research question into specific, searchable sub-questions.

Rules:
- Generate between 3 and {max_q} sub-questions
- Each sub-question should be independently searchable
- Sub-questions should together cover the full scope of the original question
- Avoid overlap between sub-questions
- Make sub-questions concrete and specific, not vague
- Return ONLY a JSON array of strings — no preamble, no markdown fences
"""

DECOMPOSE_USER = """Research question: {question}

Generate {max_q} specific sub-questions that, when answered together, 
fully address this research question. Return only a JSON array."""


def decompose_query(query: str) -> list[str]:
    """
    Break a research query into sub-questions using Claude.
    
    Args:
        query: The original user research question
        
    Returns:
        List of sub-question strings (3-5 items)
        
    Example:
        >>> decompose_query("What are best practices for fine-tuning LLMs?")
        [
          "What data preparation techniques work best for LLM fine-tuning?",
          "What hyperparameters matter most when fine-tuning large language models?",
          "How do you evaluate a fine-tuned LLM for quality and safety?",
          "What are common failure modes in LLM fine-tuning and how to avoid them?"
        ]
    """
    client = Anthropic(api_key=config.anthropic_api_key)
    
    logger.info(f"Decomposing query: {query!r}")
    
    system = DECOMPOSE_SYSTEM.format(max_q=config.max_sub_questions)
    user = DECOMPOSE_USER.format(question=query, max_q=config.max_sub_questions)
    
    response = client.messages.create(
        model=config.llm_model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    
    raw = response.content[0].text.strip()
    
    # Strip markdown fences if model adds them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    
    try:
        sub_questions = json.loads(raw)
        assert isinstance(sub_questions, list)
        assert all(isinstance(q, str) for q in sub_questions)
    except (json.JSONDecodeError, AssertionError) as e:
        logger.warning(f"Failed to parse decomposition JSON: {e}. Using original query.")
        return [query]
    
    # Safety clamp
    sub_questions = sub_questions[:config.max_sub_questions]
    
    logger.info(f"Decomposed into {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        logger.info(f"  {i}. {q}")
    
    return sub_questions
