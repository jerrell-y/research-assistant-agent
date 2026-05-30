# Research Assistant Agent

An agentic AI research assistant that answers questions with **cited sources** by autonomously searching the web, processing content, and synthesising answers using Claude.

Unlike a chatbot that answers from memory, this agent retrieves **live information** at query time and grounds every claim in a real source URL.

---

## How It Works

```
Your question
     │
     ▼
 Decomposer        Claude breaks it into 3–5 targeted sub-questions
     │
     ▼
  Gatherer         Tavily search → chunk → Cohere embed → Qdrant store
     │
     ▼
 Retriever         Cosine search top-20 → Cohere rerank → top-5 chunks
     │
     ▼
Synthesiser        Claude writes a cited answer with confidence score
```

---

## Features

- **Query decomposition** — breaks broad questions into targeted sub-searches for higher quality results
- **Two-stage retrieval** — fast vector search followed by cross-encoder reranking for precision
- **Cited answers** — every claim is grounded in a real source with `[Source N]` citation
- **Confidence scoring** — heuristic score based on rerank quality, source diversity, and citation density
- **Session-scoped vectors** — Qdrant vectors are cleaned up after each query, no stale data accumulates
- **Built-in eval harness** — 5-question test suite with LLM-as-judge scoring and latency breakdown
- **REST API** — FastAPI server for integrating into other applications
- **Web UI** — dark-themed Streamlit interface with live progress, source cards, and latency metrics

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | Cohere Embed v3 (`embed-english-v3.0`) |
| Reranking | Cohere Rerank 3.5 (`rerank-english-v3.0`) |
| Web search | Tavily API |
| Vector DB | Qdrant (self-hosted via Docker) |
| API server | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Project Structure

```
research-agent/
├── agent/
│   ├── __init__.py        # Config dataclass, Chunk and ResearchResult types
│   ├── decomposer.py      # Query decomposition via Claude
│   ├── gatherer.py        # Search, chunk, embed, store
│   ├── retriever.py       # Vector search + Cohere reranking
│   ├── synthesiser.py     # Answer generation + confidence scoring
│   ├── evaluator.py       # Built-in eval harness
│   └── run.py             # Orchestrator + CLI entry point
├── api/
│   ├── __init__.py
│   └── server.py          # FastAPI REST server
├── ui/
│   └── app.py             # Streamlit web frontend
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- Docker Desktop (for Qdrant)
- API keys for: Anthropic, Cohere, Tavily

### 2. Install

```bash
git clone https://github.com/yourusername/research-agent.git
cd research-agent

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
TAVILY_API_KEY=tvly-...
```

> You do **not** need an OpenAI key. Cohere handles both embeddings and reranking.

### 4. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Leave this running in a separate terminal.

### 5. Run

**Web UI** (recommended):
```bash
streamlit run ui/app.py
```
Opens at `http://localhost:8501`

**REST API:**
```bash
python api/server.py
```
Runs at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`

**CLI:**
```bash
python -m agent.run --query "What are best practices for fine-tuning LLMs in 2025?"
```

---

## API Usage

```bash
# Run a research query
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "How does retrieval-augmented generation work?"}'

# Retrieve a past result by session ID
curl http://localhost:8000/research/{session_id}
```

Response shape:
```json
{
  "session_id": "uuid",
  "query": "...",
  "answer": "... [Source 1] ... [Source 3] ...",
  "sources": [{"title": "...", "url": "...", "snippet": "..."}],
  "sub_questions": ["...", "..."],
  "confidence": 0.82,
  "metadata": {"latency_total_s": 22.4, "latency_breakdown": {...}}
}
```

---

## Eval Harness

```bash
python -m agent.evaluator --run-evals
```

Runs 5 test questions and prints a summary table measuring citation count, keyword coverage, LLM-as-judge relevance score, confidence, and per-phase latency.

---

## Configuration

All settings are controlled via `.env` — no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `MAX_SUB_QUESTIONS` | `4` | Sub-questions generated per query |
| `MAX_SEARCH_RESULTS` | `5` | Tavily results per sub-question |
| `CHUNK_SIZE` | `500` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVE` | `20` | Candidates before reranking |
| `TOP_K_RERANK` | `5` | Final chunks passed to LLM |

---

## Cost Estimate

A typical query (4 sub-questions, 800-token answer) costs approximately **$0.05–$0.15 USD**.

| Service | Free Tier | Paid Rate |
|---|---|---|
| Anthropic | — | $3.00 / 1M input, $15.00 / 1M output |
| Cohere | 1,000 calls/month | $0.10 / 1M embed tokens |
| Tavily | 1,000 searches/month | $0.008 / search |
| Qdrant | Free (self-hosted) | Free |

---

## Known Limitations

- No persistent memory between sessions
- Synchronous pipeline (no streaming progress updates)
- In-memory result cache (lost on server restart — use Redis for production)
- Cohere free tier: 1,000 calls/month shared across embed and rerank

---

## Potential Improvements

- Streaming SSE endpoint for real-time phase updates
- Semantic caching (cache by query vector similarity, not exact string)
- LangGraph integration for retry loops and self-correction
- Redis result cache for production persistence
- Domain specialisation (medical, legal, financial literature)

---

## License

MIT
