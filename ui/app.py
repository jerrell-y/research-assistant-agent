"""
ui/app.py
─────────
Streamlit frontend for the Research Assistant Agent.

Run with:
    streamlit run ui/app.py
"""

import sys
import os
import time
import uuid

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from agent.run import run_research

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark refined palette */
  :root {
    --bg: #0f1117;
    --surface: #1a1d26;
    --border: #2a2d3a;
    --accent: #6c8fff;
    --accent-soft: #3d4f99;
    --text: #e2e5f0;
    --muted: #7a7f96;
    --success: #4caf82;
    --warn: #f0a050;
  }

  .stApp { background: var(--bg); color: var(--text); font-family: 'IBM Plex Mono', monospace; }

  /* Header */
  .agent-header {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }
  .agent-header h1 {
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0;
  }
  .agent-header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0.4rem 0 0;
  }

  /* Phase cards */
  .phase-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
  }
  .phase-label {
    font-size: 0.7rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }

  /* Source card */
  .source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
  }
  .source-title { color: var(--accent); font-weight: 500; }
  .source-url { color: var(--muted); font-size: 0.75rem; word-break: break-all; }
  .source-snippet { color: var(--text); margin-top: 0.4rem; line-height: 1.5; }

  /* Confidence badge */
  .conf-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
  }
  .conf-high  { background: #1a3329; color: var(--success); border: 1px solid #2a5040; }
  .conf-mid   { background: #2d2414; color: var(--warn); border: 1px solid #5a4020; }
  .conf-low   { background: #2a1a1a; color: #e05050; border: 1px solid #5a2a2a; }

  /* Metrics row */
  .metric-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }
  .metric-item { font-size: 0.78rem; color: var(--muted); }
  .metric-item span { color: var(--text); font-weight: 500; }

  /* Answer area */
  .answer-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    line-height: 1.7;
    font-size: 0.9rem;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 0 !important; }
</style>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("**API Keys**")
    st.info("Set keys in `.env` — don't paste secrets here in production.", icon="ℹ️")
    
    st.divider()
    
    st.markdown("**Pipeline**")
    show_sub_questions = st.toggle("Show sub-questions", value=True)
    show_sources = st.toggle("Show sources", value=True)
    show_latency = st.toggle("Show latency breakdown", value=True)
    cleanup = st.toggle("Cleanup vectors after query", value=True)
    
    st.divider()
    
    st.markdown("**Example queries**")
    examples = [
        "What are best practices for fine-tuning LLMs in 2025?",
        "How does RAG work and when should I use it vs fine-tuning?",
        "What is the current state of AI safety research?",
        "How do transformer attention mechanisms work?",
    ]
    for ex in examples:
        if st.button(ex[:50] + "...", use_container_width=True):
            st.session_state["query"] = ex


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="agent-header">
  <h1>🔬 Research Agent</h1>
  <p>Agentic web search + RAG — decompose · search · embed · retrieve · synthesise</p>
</div>
""", unsafe_allow_html=True)

# Query input
default_query = st.session_state.get("query", "")
query = st.text_area(
    "Research question",
    value=default_query,
    placeholder="e.g. What are the best practices for fine-tuning LLMs in 2025?",
    height=80,
    label_visibility="collapsed",
)

run_btn = st.button("Run Research", type="primary", use_container_width=False)

# ── Run pipeline ──────────────────────────────────────────────────────────────

if run_btn and query.strip():
    
    # Progress display
    status_container = st.empty()
    
    phases = [
        ("🧠", "Decomposing query into sub-questions..."),
        ("🔍", "Searching the web..."),
        ("✂️",  "Chunking and embedding documents..."),
        ("📊", "Retrieving and reranking chunks..."),
        ("✍️",  "Synthesising answer..."),
    ]
    
    progress_bar = st.progress(0)
    status_text  = st.empty()
    
    def update_status(phase_idx: int):
        icon, msg = phases[phase_idx]
        status_text.markdown(f"**{icon} {msg}**")
        progress_bar.progress((phase_idx + 1) / len(phases))
    
    update_status(0)
    
    try:
        # Monkey-patch logging to update UI
        # (in production use a proper streaming callback)
        import logging as _logging
        
        original_info = _logging.getLogger("research_agent").info
        phase_counter = [0]
        
        def patched_info(msg, *args, **kwargs):
            original_info(msg, *args, **kwargs)
            if "Decomposed into" in str(msg):
                update_status(1)
            elif "Searching:" in str(msg):
                update_status(2)
            elif "Upserted" in str(msg):
                update_status(3)
            elif "Synthesising" in str(msg):
                update_status(4)
        
        _logging.getLogger("research_agent").info = patched_info
        
        result = run_research(query, cleanup=cleanup)
        
        _logging.getLogger("research_agent").info = original_info
        
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        progress_bar.empty()
        status_text.empty()
        st.stop()
    
    progress_bar.empty()
    status_text.empty()
    
    # ── Results ───────────────────────────────────────────────────────────────
    
    st.divider()
    
    # Confidence badge
    conf = result.confidence
    if conf >= 0.7:
        badge_class = "conf-high"
        conf_label = f"High confidence · {conf:.0%}"
    elif conf >= 0.4:
        badge_class = "conf-mid"
        conf_label = f"Medium confidence · {conf:.0%}"
    else:
        badge_class = "conf-low"
        conf_label = f"Low confidence · {conf:.0%}"
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("### Answer")
    with col2:
        st.markdown(
            f'<span class="conf-badge {badge_class}">{conf_label}</span>',
            unsafe_allow_html=True,
        )
    
    # Answer
    with st.container():
        st.markdown(result.answer)
    
    # Latency metrics
    if show_latency:
        lb = result.metadata.get("latency_breakdown", {})
        total = result.metadata.get("latency_total_s", "?")
        gc = result.metadata.get("gather_stats", {}).get("total_chunks", 0)
        
        st.markdown(
            f"""<div class="metric-row">
              <div class="metric-item">Total <span>{total}s</span></div>
              <div class="metric-item">Decompose <span>{lb.get('decompose_s','?')}s</span></div>
              <div class="metric-item">Gather <span>{lb.get('gather_s','?')}s</span></div>
              <div class="metric-item">Retrieve <span>{lb.get('retrieve_s','?')}s</span></div>
              <div class="metric-item">Synthesise <span>{lb.get('synthesise_s','?')}s</span></div>
              <div class="metric-item">Chunks <span>{gc}</span></div>
              <div class="metric-item">Sources <span>{len(result.sources)}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    
    # Sub-questions
    if show_sub_questions and result.sub_questions:
        with st.expander(f"Sub-questions ({len(result.sub_questions)})", expanded=False):
            for i, q in enumerate(result.sub_questions, 1):
                st.markdown(f"""
                <div class="phase-card">
                  <div class="phase-label">Sub-question {i}</div>
                  {q}
                </div>
                """, unsafe_allow_html=True)
    
    # Sources
    if show_sources and result.sources:
        with st.expander(f"Sources ({len(result.sources)})", expanded=True):
            for i, src in enumerate(result.sources, 1):
                st.markdown(f"""
                <div class="source-card">
                  <div class="source-title">[{i}] {src['title']}</div>
                  <div class="source-url">{src['url']}</div>
                  <div class="source-snippet">{src['snippet']}</div>
                </div>
                """, unsafe_allow_html=True)

elif run_btn and not query.strip():
    st.warning("Please enter a research question.")
