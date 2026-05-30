"""
api/server.py
─────────────
FastAPI REST server wrapping the research agent.

Endpoints:
  POST /research          — run full pipeline, return result
  GET  /research/{id}     — retrieve a cached result by session ID
  GET  /health            — health check

Run with:
    python api/server.py
    # or
    uvicorn api.server:app --reload --port 8000
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.run import run_research
from agent import ResearchResult

logger = logging.getLogger("research_agent.api")

# In-memory cache of results (swap for Redis in production)
_result_cache: dict[str, dict] = {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="Research question")
    cleanup: bool = Field(True, description="Delete session vectors after synthesis")


class SourceSchema(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: list[SourceSchema]
    sub_questions: list[str]
    confidence: float
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Research Agent API starting up")
    yield
    logger.info("Research Agent API shutting down")


app = FastAPI(
    title="Research Assistant Agent",
    description="Agentic web search + RAG pipeline for cited research answers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Run the full research pipeline:
    decompose → gather → retrieve → synthesise

    This is synchronous and may take 15–45 seconds depending on
    the number of sub-questions and network latency.
    For async streaming, wire up SSE on top of this endpoint.
    """
    session_id = str(uuid.uuid4())
    
    logger.info(f"[{session_id}] Research request: {request.query!r}")
    t0 = time.perf_counter()
    
    try:
        result: ResearchResult = run_research(
            query=request.query,
            session_id=session_id,
            cleanup=request.cleanup,
        )
    except Exception as e:
        logger.error(f"[{session_id}] Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {str(e)}")
    
    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(f"[{session_id}] Completed in {elapsed}s")
    
    response = ResearchResponse(
        session_id=session_id,
        query=result.query,
        answer=result.answer,
        sources=[SourceSchema(**s) for s in result.sources],
        sub_questions=result.sub_questions,
        confidence=result.confidence,
        metadata=result.metadata,
    )
    
    # Cache the result
    _result_cache[session_id] = response.model_dump()
    
    return response


@app.get("/research/{session_id}", response_model=ResearchResponse)
async def get_result(session_id: str):
    """Retrieve a previously completed research result by session ID."""
    if session_id not in _result_cache:
        raise HTTPException(status_code=404, detail="Session not found")
    return _result_cache[session_id]


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
