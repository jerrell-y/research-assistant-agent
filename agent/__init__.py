"""
Research Assistant Agent
========================
Agentic RAG pipeline: decompose → search → chunk → embed → retrieve → synthesise
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research_agent")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # API keys
    anthropic_api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    tavily_api_key: str = field(default_factory=lambda: os.environ["TAVILY_API_KEY"])
    cohere_api_key: str = field(default_factory=lambda: os.environ["COHERE_API_KEY"])

    # Qdrant
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "research_agent"))

    # Models
    llm_model: str = "claude-sonnet-4-6"
    embedding_model: str = "embed-english-v3.0"
    embedding_dim: int = 1024
    rerank_model: str = "rerank-english-v3.0"

    # Chunking
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))

    # Retrieval
    max_search_results: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "5")))
    top_k_retrieve: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVE", "20")))
    top_k_rerank: int = field(default_factory=lambda: int(os.getenv("TOP_K_RERANK", "5")))
    max_sub_questions: int = field(default_factory=lambda: int(os.getenv("MAX_SUB_QUESTIONS", "4")))


config = Config()


# ── Shared data types ─────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """A piece of text with its source metadata."""
    text: str
    url: str
    title: str
    sub_question: str
    chunk_index: int
    score: float = 0.0          # similarity score from vector search
    rerank_score: float = 0.0   # reranker score (higher = better)


@dataclass
class ResearchResult:
    """Final output from the research pipeline."""
    query: str
    answer: str
    sources: list[dict]         # [{title, url, snippet}]
    sub_questions: list[str]
    confidence: float           # 0.0 – 1.0
    metadata: dict = field(default_factory=dict)  # latency, token counts, etc.
