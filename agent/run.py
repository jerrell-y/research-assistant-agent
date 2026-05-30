"""
run.py
──────
Orchestrator for the full research pipeline.

Flow:
  decompose → gather → retrieve → synthesise → return ResearchResult

Also serves as the CLI entry point:
    python -m agent.run --query "What is RAG?"
"""

import argparse
import logging
import time
import uuid

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import ResearchResult, config
from .decomposer import decompose_query
from .gatherer import cleanup_session, gather_all
from .retriever import deduplicate_sources, retrieve
from .synthesiser import synthesise

logger = logging.getLogger("research_agent.run")
console = Console()


def run_research(query: str, session_id: str | None = None, cleanup: bool = True) -> ResearchResult:
    """
    Run the full research pipeline.
    
    Args:
        query: The user's research question
        session_id: Optional ID to scope this session's vectors (auto-generated if None)
        cleanup: Whether to delete vectors from Qdrant after synthesis
        
    Returns:
        ResearchResult with answer, sources, confidence, and metadata
    """
    session_id = session_id or str(uuid.uuid4())
    metadata: dict = {"session_id": session_id, "latency_breakdown": {}}
    
    logger.info(f"Starting research session {session_id!r}")
    logger.info(f"Query: {query!r}")
    
    t_total = time.perf_counter()
    
    # ── Phase 1: Decompose ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    sub_questions = decompose_query(query)
    metadata["latency_breakdown"]["decompose_s"] = round(time.perf_counter() - t0, 2)
    
    # ── Phase 2: Gather ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    gather_stats = gather_all(sub_questions, session_id)
    metadata["latency_breakdown"]["gather_s"] = round(time.perf_counter() - t0, 2)
    metadata["gather_stats"] = gather_stats
    
    if gather_stats["total_chunks"] == 0:
        logger.error("No chunks gathered — aborting")
        return ResearchResult(
            query=query,
            answer="No search results were found. Check your Tavily API key and network connection.",
            sources=[],
            sub_questions=sub_questions,
            confidence=0.0,
            metadata=metadata,
        )
    
    # ── Phase 3: Retrieve ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    chunks = retrieve(query, session_id)
    metadata["latency_breakdown"]["retrieve_s"] = round(time.perf_counter() - t0, 2)
    metadata["num_chunks_retrieved"] = len(chunks)
    
    sources = deduplicate_sources(chunks)
    
    # ── Phase 4: Synthesise ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    answer, confidence = synthesise(query, chunks, sub_questions, sources)
    metadata["latency_breakdown"]["synthesise_s"] = round(time.perf_counter() - t0, 2)
    
    # ── Cleanup ──────────────────────────────────────────────────────────────
    if cleanup:
        try:
            cleanup_session(session_id)
        except Exception as e:
            logger.warning(f"Cleanup failed (non-fatal): {e}")
    
    metadata["latency_total_s"] = round(time.perf_counter() - t_total, 2)
    
    logger.info(
        f"Research complete in {metadata['latency_total_s']}s | "
        f"confidence={confidence:.2f} | sources={len(sources)}"
    )
    
    return ResearchResult(
        query=query,
        answer=answer,
        sources=sources,
        sub_questions=sub_questions,
        confidence=confidence,
        metadata=metadata,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Research Assistant Agent — web search + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help='Research question e.g. "What are best practices for LLM fine-tuning?"',
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep vectors in Qdrant after synthesis (useful for debugging)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Decomposing query...", total=None)
        
        # We run the full pipeline — progress updates happen via logger
        result = run_research(args.query, cleanup=not args.no_cleanup)
    
    if args.json:
        import json
        print(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "sub_questions": result.sub_questions,
            "confidence": result.confidence,
            "metadata": result.metadata,
        }, indent=2))
        return
    
    # Rich formatted output
    console.print()
    console.print(Panel(
        f"[bold]{result.query}[/bold]",
        title="Research Question",
        border_style="blue",
    ))
    
    console.print(Panel(
        ", ".join(f"[dim]{q}[/dim]" for q in result.sub_questions),
        title=f"Sub-questions ({len(result.sub_questions)})",
        border_style="dim",
    ))
    
    console.print(Markdown(result.answer))
    
    console.print()
    console.print("[bold]Sources:[/bold]")
    for i, src in enumerate(result.sources, 1):
        console.print(f"  [{i}] [link={src['url']}]{src['title']}[/link]")
        console.print(f"      [dim]{src['url']}[/dim]")
    
    console.print()
    
    lb = result.metadata.get("latency_breakdown", {})
    console.print(
        f"[dim]Confidence: {result.confidence:.0%} | "
        f"Decompose: {lb.get('decompose_s', '?')}s | "
        f"Gather: {lb.get('gather_s', '?')}s | "
        f"Retrieve: {lb.get('retrieve_s', '?')}s | "
        f"Synthesise: {lb.get('synthesise_s', '?')}s | "
        f"Total: {result.metadata.get('latency_total_s', '?')}s[/dim]"
    )


if __name__ == "__main__":
    main()
