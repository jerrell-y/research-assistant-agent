"""
retriever.py
────────────
Two-stage retrieval:
  1. Vector similarity search in Qdrant (top-K candidates)
  2. Cohere reranker to re-score and select final top-N

The reranker is a cross-encoder — it reads the query AND each chunk
together, giving far more accurate relevance scores than embedding
similarity alone (which is a bi-encoder and misses nuance).
"""

import logging

import cohere
from qdrant_client.models import Filter, FieldCondition, MatchValue

from . import Chunk, config
from .gatherer import _get_qdrant, embed_chunks, embed_query

logger = logging.getLogger("research_agent.retriever")


_cohere: cohere.Client | None = None


def _get_cohere() -> cohere.Client:
    global _cohere
    if _cohere is None:
        _cohere = cohere.Client(api_key=config.cohere_api_key)
    return _cohere

def vector_search(query: str, session_id: str, top_k: int | None = None) -> list[Chunk]:
    """
    Embed the query and search Qdrant for the most similar chunks,
    filtered to the current session.
    
    Args:
        query: The original research question (not sub-questions)
        session_id: Scopes search to this session's chunks only
        top_k: Number of candidates to retrieve (default: config.top_k_retrieve)
        
    Returns:
        List of Chunk objects with .score populated
    """
    top_k = top_k or config.top_k_retrieve
    
    # Embed the query using the same model as the chunks
    query_embedding = embed_query(query)
    
    client = _get_qdrant()
    
    results = client.query_points(
        collection_name=config.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
        with_payload=True,
    ).points

    chunks = []
    for hit in results:
        p = hit.payload
        chunk = Chunk(
            text=p["text"],
            url=p["url"],
            title=p["title"],
            sub_question=p["sub_question"],
            chunk_index=p["chunk_index"],
            score=hit.score,
        )
        chunks.append(chunk)
    
    logger.info(f"Vector search returned {len(chunks)} candidates (top-{top_k})")
    return chunks


def rerank(query: str, chunks: list[Chunk], top_n: int | None = None) -> list[Chunk]:
    """
    Use Cohere's cross-encoder reranker to re-score chunks.
    
    The reranker reads (query, document) pairs jointly — it understands
    semantic relationships that bi-encoder embeddings miss.
    
    Args:
        query: Original research question
        chunks: Candidate chunks from vector search
        top_n: How many to return after reranking (default: config.top_k_rerank)
        
    Returns:
        Top-N chunks sorted by rerank score descending
    """
    top_n = top_n or config.top_k_rerank
    
    if not chunks:
        return []
    
    if len(chunks) <= top_n:
        logger.info("Fewer candidates than top_n — skipping reranker")
        return chunks
    
    documents = [c.text for c in chunks]
    
    response = _get_cohere().rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model=config.rerank_model,
    )
    
    reranked: list[Chunk] = []
    for r in response.results:
        chunk = chunks[r.index]
        chunk.rerank_score = r.relevance_score
        reranked.append(chunk)
    
    logger.info(f"Reranked to top-{len(reranked)} chunks")
    for i, c in enumerate(reranked):
        logger.debug(f"  [{i+1}] score={c.rerank_score:.3f} url={c.url[:60]}")
    
    return reranked


def retrieve(query: str, session_id: str) -> list[Chunk]:
    """
    Full two-stage retrieval: vector search → rerank.
    This is the main entry point for the retrieval phase.
    """
    candidates = vector_search(query, session_id)
    final = rerank(query, candidates)
    return final


def deduplicate_sources(chunks: list[Chunk]) -> list[dict]:
    """
    Collapse chunks into unique sources for the citation list.
    Returns [{title, url, snippet}] with one entry per URL.
    """
    seen_urls: dict[str, dict] = {}
    
    for chunk in chunks:
        if chunk.url not in seen_urls:
            seen_urls[chunk.url] = {
                "title": chunk.title,
                "url": chunk.url,
                "snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            }
    
    return list(seen_urls.values())
