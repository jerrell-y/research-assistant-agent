"""
gatherer.py
───────────
For each sub-question:
  1. Search the web via Tavily
  2. Chunk the returned content (RecursiveCharacterTextSplitter)
  3. Embed each chunk (using Cohere)
  4. Upsert into Qdrant

Tavily returns pre-cleaned text — no HTML scraping needed unless you
want to fetch the full page body (toggled with `include_raw_content`).
"""

import hashlib
import logging
import time
import uuid
import cohere
from typing import Generator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from . import Chunk, config

logger = logging.getLogger("research_agent.gatherer")


# ── Clients (lazy singletons) ─────────────────────────────────────────────────

_tavily: TavilyClient | None = None
_qdrant: QdrantClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=config.tavily_api_key)
    return _tavily


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=config.qdrant_url)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    collections = {c.name for c in client.get_collections().collections}
    if config.qdrant_collection not in collections:
        client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config=VectorParams(
                size=config.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {config.qdrant_collection!r}")


# ── Text splitter ─────────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


# ── Core functions ────────────────────────────────────────────────────────────

def search_web(sub_question: str) -> list[dict]:
    """
    Search Tavily for a sub-question.
    
    Returns list of dicts: {url, title, content, score}
    """
    logger.info(f"Searching: {sub_question!r}")
    
    results = _get_tavily().search(
        query=sub_question,
        max_results=config.max_search_results,
        search_depth="advanced",      # deeper search, still fast
        include_raw_content=False,    # Tavily's cleaned text is usually enough
        include_answer=False,
    )
    
    docs = results.get("results", [])
    logger.info(f"  → {len(docs)} results")
    return docs


def chunk_documents(docs: list[dict], sub_question: str) -> list[Chunk]:
    """
    Split document content into overlapping chunks.
    Deduplicates by content hash to avoid storing the same paragraph twice.
    """
    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    
    for doc in docs:
        content = doc.get("content", "").strip()
        if not content:
            continue
        
        raw_chunks = _splitter.split_text(content)
        
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < 50:  # skip tiny fragments
                continue
            
            content_hash = hashlib.md5(text.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            chunks.append(Chunk(
                text=text,
                url=doc.get("url", ""),
                title=doc.get("title", "Untitled"),
                sub_question=sub_question,
                chunk_index=i,
            ))
    
    logger.info(f"  → {len(chunks)} unique chunks after dedup")
    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    co = cohere.Client(api_key=config.cohere_api_key)
    texts = [c.text for c in chunks]
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings

def embed_query(query: str) -> list[float]:
    co = cohere.Client(api_key=config.cohere_api_key)
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return response.embeddings[0]

def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]], session_id: str) -> None:
    """
    Store chunks + embeddings in Qdrant.
    Each point carries the text and metadata as payload.
    """
    client = _get_qdrant()
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.text,
                "url": chunk.url,
                "title": chunk.title,
                "sub_question": chunk.sub_question,
                "chunk_index": chunk.chunk_index,
                "session_id": session_id,      # scopes this search session
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    client.upsert(collection_name=config.qdrant_collection, points=points)
    logger.info(f"  → Upserted {len(points)} points to Qdrant")


def gather_for_question(sub_question: str, session_id: str) -> int:
    """
    Full pipeline for one sub-question: search → chunk → embed → store.
    Returns the number of chunks stored.
    """
    docs = search_web(sub_question)
    if not docs:
        logger.warning(f"No results for: {sub_question!r}")
        return 0
    
    chunks = chunk_documents(docs, sub_question)
    if not chunks:
        return 0
    
    embeddings = embed_chunks(chunks)
    upsert_chunks(chunks, embeddings, session_id)
    
    return len(chunks)


def gather_all(sub_questions: list[str], session_id: str) -> dict:
    """
    Run gather_for_question for every sub-question.
    Returns summary stats: {total_chunks, per_question: [...]}
    """
    stats = {"total_chunks": 0, "per_question": []}
    
    for q in sub_questions:
        t0 = time.perf_counter()
        n = gather_for_question(q, session_id)
        elapsed = time.perf_counter() - t0
        
        stats["total_chunks"] += n
        stats["per_question"].append({
            "question": q,
            "chunks": n,
            "latency_s": round(elapsed, 2),
        })
    
    logger.info(f"Gather complete: {stats['total_chunks']} total chunks")
    return stats


def cleanup_session(session_id: str) -> None:
    """
    Delete all vectors for a session from Qdrant.
    Call this after the session is complete to keep the collection lean.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    client = _get_qdrant()
    client.delete(
        collection_name=config.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
    )
    logger.info(f"Cleaned up session {session_id!r}")
