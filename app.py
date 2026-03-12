"""
7-Layer Advanced RAG — Banking Assistant
Main Streamlit Application with Admin/User roles.
Enhanced with detailed chunk-level display for Layers 3-7.
"""
import os
import re
import sys
import time
import streamlit as st
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from rag_engine.seven_layer_rag import SevenLayerRAG, RAGResponse
from feedback.feedback_service import FeedbackService, FeedbackEntry
from ui.detailed_layer_display import render_detailed_pipeline
from testing.faq_test_runner import FAQTestRunner, GroundTruthQA, TestReport, TestResult

# ==========================================
# PAGE CONFIG
# ==========================================
settings = get_settings()
st.set_page_config(
    page_title=settings.app.app_title,
    page_icon=settings.app.app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 28px; }
    .main-header p { color: #bbdefb; margin: 5px 0 0 0; font-size: 14px; }

    /* Layer badges */
    .layer-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .layer-executed { background: #e8f5e9; color: #2e7d32; }
    .layer-skipped { background: #fff3e0; color: #e65100; }
    .layer-cache-hit { background: #e3f2fd; color: #1565c0; }
    .layer-cache-miss { background: #fce4ec; color: #c62828; }
    .layer-pass-through { background: #f3e5f5; color: #6a1b9a; }

    /* Confidence badge */
    .confidence-high { background: #e8f5e9; color: #2e7d32; padding: 4px 12px; border-radius: 8px; }
    .confidence-medium { background: #fff3e0; color: #e65100; padding: 4px 12px; border-radius: 8px; }
    .confidence-low { background: #fce4ec; color: #c62828; padding: 4px 12px; border-radius: 8px; }

    /* Feedback buttons */
    .feedback-section { margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 8px; }

    /* Admin badge */
    .admin-badge {
        background: #ff6f00;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Source card */
    .source-card {
        background: #2d2d2d;
        color: #e0e0e0;
        padding: 8px 12px;
        border-radius: 6px;
        border-left: 3px solid #42a5f5;
        margin: 4px 0;
        font-size: 13px;
    }
    .source-card strong { color: white; }
    .source-card em { color: #aaaaaa; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    defaults = {
        "authenticated": False,
        "is_admin": False,
        "user_id": "",
        "messages": [],
        "rag_engine": None,
        "feedback_service": None,
        "engine_initialized": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ==========================================
# AUTHENTICATION
# ==========================================
def render_login():
    """Render the login screen."""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 7-Layer Advanced RAG — Banking Assistant</h1>
        <p>Powered by Advanced Retrieval-Augmented Generation with 7 Intelligence Layers</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### 🔐 Sign In")
        st.markdown("Enter your credentials to access the Banking Assistant.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            login_btn = st.form_submit_button("🔑 Login", use_container_width=True)

            if login_btn:
                if username == settings.auth.admin_username and password == settings.auth.admin_password:
                    st.session_state.authenticated = True
                    st.session_state.is_admin = True
                    st.session_state.user_id = username
                    st.rerun()
                elif username and password:
                    # Any non-admin user can log in as a regular user
                    st.session_state.authenticated = True
                    st.session_state.is_admin = False
                    st.session_state.user_id = username
                    st.rerun()
                else:
                    st.error("Please enter both username and password.")

        st.markdown("---")
        st.markdown("**Admin Access:** `admin` / `admin88$`")
        st.markdown("**User Access:** Any username and password")


# ==========================================
# ENGINE INITIALIZATION
# ==========================================
def initialize_engine():
    """Initialize the RAG engine and index sample documents."""
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = SevenLayerRAG()
        st.session_state.feedback_service = FeedbackService()

    if not st.session_state.engine_initialized:
        with st.spinner("🔄 Initializing 7-Layer RAG Engine and indexing sample banking policies..."):
            st.session_state.rag_engine.initialize()
            st.session_state.engine_initialized = True


# ==========================================
# SIDEBAR
# ==========================================
def render_sidebar():
    """Render the sidebar with user info and admin controls."""
    engine = st.session_state.rag_engine

    with st.sidebar:
        # User info
        role_badge = '<span class="admin-badge">ADMIN</span>' if st.session_state.is_admin else "👤 User"
        st.markdown(f"### {settings.app.app_icon} Banking Assistant")
        st.markdown(f"Logged in as: **{st.session_state.user_id}** {role_badge}", unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")

        # System stats
        st.markdown("### 📊 System Status")
        if engine:
            col1, col2 = st.columns(2)
            col1.metric("Documents", engine.indexer.get_document_count())
            col2.metric("Chunks", engine.indexer.get_chunk_count())

            col3, col4 = st.columns(2)
            col3.metric("Cache Size", len(engine.cache.cache))
            col4.metric("Cache TTL", f"{settings.rag.cache_ttl_seconds // 60}m")

            # Persistence status
            if hasattr(engine, 'persistence') and engine.persistence:
                persistence = engine.persistence
                if persistence.is_connected:
                    st.markdown("---")
                    st.markdown("### 💾 Persistent Storage")
                    p_stats = persistence.get_stats()
                    mongo_stats = p_stats.get('mongodb', {})
                    faiss_stats = p_stats.get('faiss', {})
                    pcol1, pcol2 = st.columns(2)
                    pcol1.metric("MongoDB", f"{'🟢' if mongo_stats.get('connected') else '🔴'}")
                    pcol2.metric("FAISS", f"{faiss_stats.get('total_vectors', 0)} vectors")
                    pcol3, pcol4 = st.columns(2)
                    pcol3.metric("Stored Docs", mongo_stats.get('documents', 0))
                    pcol4.metric("Stored Chunks", mongo_stats.get('chunks', 0))
                else:
                    st.caption("💾 Persistence: Not connected (using in-memory)")

            # Indexed sources
            sources = engine.indexer.get_sources()
            if sources:
                with st.expander("📄 Indexed Documents", expanded=False):
                    for src in sources:
                        st.markdown(f"• `{src}`")

        st.markdown("---")

        # Admin: Document Upload
        if st.session_state.is_admin:
            st.markdown("### 📤 Upload Documents (Admin Only)")
            uploaded_files = st.file_uploader(
                "Upload banking policy documents",
                type=["txt", "pdf", "md"],
                accept_multiple_files=True,
                key="doc_upload"
            )

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Check if this file was already processed in this session
                    processed_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
                    if processed_key in st.session_state:
                        st.info(f"📄 `{uploaded_file.name}` already indexed ({st.session_state[processed_key]} chunks).")
                        continue

                    # Read raw bytes for advanced extraction
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0)  # Reset for fallback reading

                    text = ""
                    if uploaded_file.name.lower().endswith(".pdf"):
                        st.info(f"📄 Processing PDF: `{uploaded_file.name}` ({len(file_bytes):,} bytes)...")
                        try:
                            import fitz
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            pages_text = []
                            for page_num, page in enumerate(doc):
                                page_text = page.get_text("text")
                                if page_text.strip():
                                    pages_text.append(page_text)
                            doc.close()
                            text = "\n\n".join(pages_text)
                        except ImportError:
                            try:
                                import subprocess, tempfile
                                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                    tmp.write(file_bytes)
                                    tmp_path = tmp.name
                                result = subprocess.run(
                                    ["pdftotext", "-layout", tmp_path, "-"],
                                    capture_output=True, text=True, timeout=30
                                )
                                text = result.stdout
                                os.unlink(tmp_path)
                            except Exception as e2:
                                st.error(f"❌ PDF extraction failed for `{uploaded_file.name}`: {e2}")
                                continue
                        except Exception as e:
                            st.error(f"❌ Error reading PDF `{uploaded_file.name}`: {e}")
                            continue
                    else:
                        try:
                            text = file_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            text = file_bytes.decode("latin-1", errors="ignore")

                    if not text or len(text.strip()) < 50:
                        # Still try advanced extraction for scanned PDFs
                        if not uploaded_file.name.lower().endswith(".pdf"):
                            st.warning(f"⚠️ `{uploaded_file.name}` — extracted text is too short ({len(text.strip())} chars). Skipping.")
                            continue

                    # Determine extraction mode
                    use_advanced = (
                        engine.extraction_orchestrator is not None and
                        uploaded_file.name.lower().endswith(".pdf")
                    )

                    if use_advanced:
                        # Show extraction mode
                        fast_mode = settings.rag.extraction_config.get('fast_mode', True)
                        mode_label = "⚡ Fast" if fast_mode else "🔬 Full"
                        st.info(f"Processing `{uploaded_file.name}` in **{mode_label}** mode...")

                        # Progress bar with stage updates
                        progress_bar = st.progress(0, text="Starting extraction...")
                        status_text = st.empty()

                        def update_progress(stage_name, pct):
                            progress_bar.progress(min(pct, 1.0), text=stage_name)
                            status_text.caption(f"Stage: {stage_name}")

                        count = engine.index_uploaded_document(
                            text, uploaded_file.name, file_bytes=file_bytes,
                            progress_callback=update_progress
                        )
                        progress_bar.progress(1.0, text="Done!")
                        status_text.empty()

                        if count > 0:
                            st.session_state[processed_key] = count
                            st.success(f"✅ `{uploaded_file.name}` — {count} enriched chunks indexed ({mode_label} mode).")

                            # Show extraction details in expander
                            with st.expander(f"🔬 Extraction Details: {uploaded_file.name}", expanded=False):
                                orch = engine.extraction_orchestrator
                                if orch and orch.events:
                                    st.markdown("**Strategies Used:**")
                                    for event in orch.events:
                                        evt_type = event.get('event_type', '')
                                        evt_data = event.get('data', {})
                                        if evt_type == 'strategies_selected':
                                            for s in evt_data.get('strategies', []):
                                                st.markdown(f"  ✓ {s.title()}")
                                        elif evt_type == 'vision_completed':
                                            st.markdown(f"**Vision:** {evt_data.get('pages', 0)} pages, {evt_data.get('tables', 0)} tables, {evt_data.get('formulas', 0)} formulas")
                                        elif evt_type == 'table_completed':
                                            st.markdown(f"**Tables:** {evt_data.get('tables_found', 0)} tables via {evt_data.get('backend', 'unknown')}")
                                        elif evt_type == 'graph_completed':
                                            st.markdown(f"**Knowledge Graph:** {evt_data.get('entities', 0)} entities, {evt_data.get('relationships', 0)} relationships")
                                        elif evt_type == 'enrichment_completed':
                                            st.markdown(f"**Enrichment:** {evt_data.get('enriched_chunks', 0)} enriched chunks")
                                        elif evt_type == 'extraction_completed':
                                            time_s = evt_data.get('processing_time_ms', 0) / 1000
                                            st.markdown(f"**Total Time:** {time_s:.1f}s")
                        else:
                            st.warning(f"⚠️ `{uploaded_file.name}` — advanced extraction produced no chunks.")
                    else:
                        with st.spinner(f"🔄 Indexing `{uploaded_file.name}` ({len(text):,} chars)..."):
                            count = engine.index_uploaded_document(text, uploaded_file.name)
                            if count > 0:
                                st.session_state[processed_key] = count
                                st.success(f"✅ `{uploaded_file.name}` indexed — {count} chunks created.")
                            else:
                                st.warning(f"⚠️ `{uploaded_file.name}` — no chunks created.")
        else:
            st.info("📤 Document upload is available for Admin users only.")

        st.markdown("---")

        # Layer configuration (admin only)
        if st.session_state.is_admin:
            with st.expander("⚙️ Layer Configuration", expanded=False):
                # ═══ TURBO MODE — Bypass all heavy layers for FAQ-style documents ═══
                turbo_mode = st.toggle(
                    "🚀 TURBO MODE (Instant FAQ Answers)",
                    value=st.session_state.get('turbo_mode', False),
                    help="Bypasses ALL heavy layers (HyDE, CRAG, Re-Ranking, Agentic, Governance). "
                         "Uses FAQ matching + simple vector search + 1 LLM call max. "
                         "Best for FAQ documents where speed matters."
                )
                st.session_state['turbo_mode'] = turbo_mode
                if turbo_mode:
                    st.success("⚡ TURBO: FAQ Match → Vector Search → 1 LLM call → Answer (~1-3s)")
                    # Force-disable all heavy layers
                    settings.rag.cache_enabled = False
                    settings.rag.hyde_enabled = False
                    settings.rag.crag_enabled = False
                    settings.rag.rerank_enabled = False
                    settings.rag.agentic_enabled = False
                    settings.rag.hallucination_check_enabled = False
                    settings.rag.governance_enabled = False
                else:
                    st.info("🔬 FULL PIPELINE: All 7 layers + Governance active")
                st.markdown("---")

                # Individual layer toggles (disabled in turbo mode)
                settings.rag.cache_enabled = st.toggle("Layer 1: Semantic Caching", value=settings.rag.cache_enabled, disabled=turbo_mode)
                settings.rag.hyde_enabled = st.toggle("Layer 2: HyDE", value=settings.rag.hyde_enabled, disabled=turbo_mode)
                settings.rag.crag_enabled = st.toggle("Layer 4: CRAG", value=settings.rag.crag_enabled, disabled=turbo_mode)
                settings.rag.rerank_enabled = st.toggle("Layer 5: Re-Ranking", value=settings.rag.rerank_enabled, disabled=turbo_mode)
                settings.rag.agentic_enabled = st.toggle("Layer 6: Agentic RAG", value=settings.rag.agentic_enabled, disabled=turbo_mode)
                settings.rag.hallucination_check_enabled = st.toggle("Layer 7: Hallucination Check", value=settings.rag.hallucination_check_enabled, disabled=turbo_mode)
                st.markdown("---")
                settings.rag.detailed_display_enabled = st.toggle(
                    "🔬 Detailed Chunk Display",
                    value=settings.rag.detailed_display_enabled,
                    help="Show detailed chunk-level information for Layers 3-7 including text previews, scores, and selection reasons"
                )
                st.markdown("---")
                st.markdown("**🛡️ Governance (CBUAE Aligned)**")
                settings.rag.governance_enabled = st.toggle(
                    "Layer 8: AI Governance",
                    value=settings.rag.governance_enabled if not turbo_mode else False,
                    help="Four-Check System: Hallucination, Bias, PII, Compliance",
                    disabled=turbo_mode
                )
                if settings.rag.governance_enabled:
                    settings.rag.governance_check1_hallucination = st.toggle(
                        "  Check 1: Hallucination", value=settings.rag.governance_check1_hallucination
                    )
                    settings.rag.governance_check2_bias = st.toggle(
                        "  Check 2: Bias & Toxicity", value=settings.rag.governance_check2_bias
                    )
                    settings.rag.governance_check3_pii = st.toggle(
                        "  Check 3: PII Redaction", value=settings.rag.governance_check3_pii
                    )
                    settings.rag.governance_check4_compliance = st.toggle(
                        "  Check 4: Compliance", value=settings.rag.governance_check4_compliance
                    )
                    settings.rag.governance_audit_trail_enabled = st.toggle(
                        "  📋 Audit Trail", value=settings.rag.governance_audit_trail_enabled
                    )
                st.markdown("---")
                st.markdown("**🔬 Advanced Extraction Pipeline**")
                settings.rag.extraction_orchestrator_enabled = st.toggle(
                    "Multi-Strategy Extraction",
                    value=settings.rag.extraction_orchestrator_enabled,
                    help="Vision + Table + Graph + Contextual Enrichment for complex PDFs"
                )
                if settings.rag.extraction_orchestrator_enabled:
                    ext_cfg = settings.rag.extraction_config
                    ext_cfg['fast_mode'] = st.toggle(
                        "  ⚡ Fast Mode (~30s)",
                        value=ext_cfg.get('fast_mode', True),
                        help="Fast: Table + Enrichment only. Disable for full Vision + Graph analysis (~10min)"
                    )
                    if not ext_cfg['fast_mode']:
                        ext_cfg['vision_enabled'] = st.toggle(
                            "  👁️ Vision Extraction (LLM page analysis)",
                            value=ext_cfg.get('vision_enabled', True)
                        )
                        ext_cfg['graph_enabled'] = st.toggle(
                            "  🔗 Knowledge Graph (Entity-Relationship)",
                            value=ext_cfg.get('graph_enabled', True)
                        )
                    ext_cfg['table_enabled'] = st.toggle(
                        "  📊 Table Extraction (Docling/pdfplumber)",
                        value=ext_cfg.get('table_enabled', True)
                    )
                    ext_cfg['contextual_enabled'] = st.toggle(
                        "  📝 Contextual Enrichment (Prefix + Formulas)",
                        value=ext_cfg.get('contextual_enabled', True)
                    )
                st.markdown("---")
                st.markdown(f"**Top-K:** {settings.rag.rerank_top_k}")
                st.markdown(f"**Cache Threshold:** {settings.rag.cache_similarity_threshold:.0%}")


# ==========================================
# RESPONSE DISPLAY
# ==========================================
def render_layer_pipeline(response: RAGResponse):
    """Render the 7-layer pipeline reasoning section (compact summary)."""
    with st.expander("📋 7-Layer RAG Pipeline Reasoning", expanded=False):
        for lr in response.layer_results:
            status_class = {
                "executed": "layer-executed",
                "skipped": "layer-skipped",
                "cache_hit": "layer-cache-hit",
                "cache_miss": "layer-cache-miss",
                "pass_through": "layer-pass-through",
            }.get(lr.status, "layer-executed")

            status_icon = {
                "executed": "✅",
                "skipped": "⏭️",
                "cache_hit": "🎯",
                "cache_miss": "❌",
                "pass_through": "➡️",
            }.get(lr.status, "✅")

            st.markdown(f"""
            **Layer {lr.layer_number}: {lr.layer_name}** {status_icon}
            <span class="layer-badge {status_class}">{lr.status.upper()}</span>
            — {lr.duration_ms:.0f}ms
            """, unsafe_allow_html=True)

            # Show details
            if lr.details:
                detail_items = []
                for k, v in lr.details.items():
                    if isinstance(v, (dict, list)):
                        continue
                    detail_items.append(f"**{k.replace('_', ' ').title()}:** {v}")
                if detail_items:
                    st.markdown(" | ".join(detail_items))
            st.markdown("---")

        # Total
        st.markdown(f"**Total Pipeline Duration:** {response.total_duration_ms:.0f}ms")
        if response.pipeline_stopped_at:
            st.markdown(f"**Pipeline Stopped At:** Layer {response.pipeline_stopped_at} (Cache Hit)")


def render_source_references(response: RAGResponse):
    """Render source document references."""
    if response.sources:
        with st.expander("📄 Document References", expanded=False):
            for src in response.sources:
                st.markdown(f"""
                <div class="source-card">
                    📄 <strong>{src.get('source', 'Unknown')}</strong><br/>
                    Section: {src.get('section', 'N/A')} | Relevance: {src.get('relevance', 'N/A')}
                </div>
                """, unsafe_allow_html=True)


def render_confidence_badge(response: RAGResponse):
    """Render the confidence score badge."""
    conf = response.confidence
    if conf >= 0.70:
        css_class = "confidence-high"
        label = f"✅ Confidence: {conf:.0%}"
    elif conf >= 0.30:
        css_class = "confidence-medium"
        label = f"⚠️ Confidence: {conf:.0%}"
    else:
        css_class = "confidence-low"
        label = f"🚫 Confidence: {conf:.0%}"

    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


def render_feedback(msg_index: int, query: str, response_text: str, confidence: float):
    """Render enhanced feedback with thumbs up/down and comments."""
    feedback_key = f"feedback_{msg_index}"

    if feedback_key in st.session_state:
        rating = st.session_state[feedback_key]
        icon = "👍" if rating == "positive" else "👎"
        st.markdown(f"*{icon} Feedback submitted. Thank you!*")
        return

    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        if st.button("👍", key=f"thumbs_up_{msg_index}", help="Helpful"):
            st.session_state[feedback_key] = "positive"
            st.session_state[f"show_comment_{msg_index}"] = True
            st.rerun()
    with col2:
        if st.button("👎", key=f"thumbs_down_{msg_index}", help="Not helpful"):
            st.session_state[feedback_key] = "negative"
            st.session_state[f"show_comment_{msg_index}"] = True
            st.rerun()

    # Comment input after rating
    if st.session_state.get(f"show_comment_{msg_index}"):
        rating = st.session_state.get(feedback_key, "positive")
        prompt_text = "What did you like?" if rating == "positive" else "What could be improved?"
        comment = st.text_input(prompt_text, key=f"comment_input_{msg_index}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Submit with Comment", key=f"submit_comment_{msg_index}"):
                _save_feedback(query, response_text, rating, comment, confidence)
                st.session_state[f"show_comment_{msg_index}"] = False
                st.rerun()
        with c2:
            if st.button("Skip (no comment)", key=f"skip_comment_{msg_index}"):
                _save_feedback(query, response_text, rating, "", confidence)
                st.session_state[f"show_comment_{msg_index}"] = False
                st.rerun()


def _save_feedback(query, response_text, rating, comment, confidence):
    """Save feedback to the feedback service."""
    fs = st.session_state.feedback_service
    if fs:
        entry = FeedbackEntry(
            timestamp=time.time(),
            query=query,
            response_preview=response_text[:200],
            rating=rating,
            comment=comment,
            user_id=st.session_state.user_id,
            confidence=confidence
        )
        fs.add_feedback(entry)


# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
def render_chat():
    """Render the main chat interface."""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 7-Layer Advanced RAG — Banking Assistant</h1>
        <p>Ask questions about fund transfers, banking policies, fees, and limits. Every answer passes through 7 intelligence layers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_chat, tab_feedback, tab_admin = st.tabs(["💬 Chat", "📝 My Feedback", "🔧 Admin Panel"])

    with tab_chat:
        # AI Disclosure Banner (CBUAE Requirement)
        if settings.rag.governance_ai_disclosure:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a237e, #283593); padding: 10px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #ff9800;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 18px;">🤖</span>
                    <div>
                        <span style="color: #ffb74d; font-weight: 600; font-size: 13px;">AI-Powered Banking Assistant</span><br/>
                        <span style="color: #bbdefb; font-size: 11px;">Responses are generated by AI and verified through a 7-layer governance pipeline. 
                        For critical financial decisions, please consult a banking specialist. 
                        You may request human review of any response at any time (CBUAE Compliance).</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Display chat history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant" and "response_obj" in msg:
                    resp = msg["response_obj"]
                    render_confidence_badge(resp)
                    render_layer_pipeline(resp)
                    # Detailed chunk display (new feature — toggled via config)
                    if settings.rag.detailed_display_enabled:
                        render_detailed_pipeline(resp, show_details=True)
                    render_source_references(resp)
                    render_feedback(i, msg.get("query", ""), msg["content"], resp.confidence)

        # Chat input
        if prompt := st.chat_input("Ask about banking policies, fund transfers, fees..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process through 7-layer pipeline
            with st.chat_message("assistant"):
                is_turbo = st.session_state.get('turbo_mode', False)
                spinner_msg = "⚡ TURBO: FAQ Match → Vector Search → Answer..." if is_turbo else "🔄 Processing through 7-Layer RAG Pipeline..."
                with st.spinner(spinner_msg):
                    engine = st.session_state.rag_engine
                    response = engine.process_query(prompt, turbo_mode=is_turbo)

                st.markdown(response.answer)
                render_confidence_badge(response)
                render_layer_pipeline(response)
                # Detailed chunk display (new feature — toggled via config)
                if settings.rag.detailed_display_enabled:
                    render_detailed_pipeline(response, show_details=True)
                render_source_references(response)

                msg_idx = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "response_obj": response,
                    "query": prompt
                })
                render_feedback(msg_idx, prompt, response.answer, response.confidence)

    with tab_feedback:
        render_feedback_tab()

    with tab_admin:
        render_admin_tab()


def render_feedback_tab():
    """Render the My Feedback tab."""
    st.markdown("### 📝 My Feedback History")
    fs = st.session_state.feedback_service
    if fs:
        entries = fs.get_all_feedback()
        user_entries = [e for e in entries if e.get("user_id") == st.session_state.user_id]

        if not user_entries:
            st.info("No feedback submitted yet. Use 👍/👎 buttons on responses to provide feedback.")
        else:
            for entry in reversed(user_entries):
                icon = "👍" if entry["rating"] == "positive" else "👎"
                ts = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
                st.markdown(f"**{icon} {ts}** — {entry['query'][:80]}...")
                st.markdown(f"*Response:* {entry['response_preview'][:150]}...")
                if entry.get("comment"):
                    st.markdown(f"💬 *Comment:* {entry['comment']}")
                st.markdown("---")


def render_admin_tab():
    """Render the Admin Panel tab."""
    if not st.session_state.is_admin:
        st.warning("🔒 Admin access required. Please log in as admin.")
        return

    engine = st.session_state.rag_engine

    st.markdown("### 🔧 Admin Panel")

    # Feedback stats
    fs = st.session_state.feedback_service
    if fs:
        stats = fs.get_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Feedback", stats["total"])
        col2.metric("👍 Positive", stats["positive"])
        col3.metric("👎 Negative", stats["negative"])
        col4.metric("Satisfaction", stats["satisfaction_rate"])

        st.markdown("---")

        # All feedback
        st.markdown("#### All Feedback Entries")
        entries = fs.get_all_feedback()
        if entries:
            for entry in reversed(entries):
                icon = "👍" if entry["rating"] == "positive" else "👎"
                ts = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
                st.markdown(f"**{icon} {ts}** — User: `{entry.get('user_id', 'anonymous')}` — Confidence: {entry.get('confidence', 0):.0%}")
                st.markdown(f"*Query:* {entry['query'][:100]}...")
                if entry.get("comment"):
                    st.markdown(f"💬 *Comment:* {entry['comment']}")
                st.markdown("---")
        else:
            st.info("No feedback entries yet.")

    # ═══════════════════════════════════════════════════════════
    # DOCUMENT MANAGEMENT — View All & Delete
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 📚 Document Management")

    # Get documents from all available sources
    stored_docs = []
    persistence_connected = False
    if engine and hasattr(engine, 'persistence') and engine.persistence and engine.persistence.is_connected:
        persistence_connected = True
        p_stats = engine.persistence.get_stats()
        mongo_stats = p_stats.get('mongodb', {})
        faiss_stats = p_stats.get('faiss', {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Documents", mongo_stats.get('documents', 0))
        col2.metric("🧩 Chunks", mongo_stats.get('chunks', 0))
        col3.metric("🔢 FAISS Vectors", faiss_stats.get('total_vectors', 0))
        col4.metric("❓ FAQ Pairs", mongo_stats.get('faq_pairs', 0))

        stored_docs = engine.get_all_documents()
    elif engine:
        # Fallback: in-memory only
        stored_docs = engine.get_all_documents()
        in_mem_docs = len(stored_docs)
        in_mem_chunks = engine.indexer.get_chunk_count() if engine.indexer else 0
        faq_count = len(engine.faq_engine.faq_pairs) if engine.faq_engine else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Documents (In-Memory)", in_mem_docs)
        col2.metric("🧩 Chunks", in_mem_chunks)
        col3.metric("❓ FAQ Pairs", faq_count)
        st.caption("⚠️ MongoDB not connected. Documents are stored in-memory only and will be lost on restart.")

    if stored_docs:
        st.markdown("**Uploaded Documents:**")

        # Build a table of documents
        for idx, doc in enumerate(stored_docs):
            upload_ts = doc.get('upload_time', 0)
            if upload_ts > 0:
                upload_time = datetime.fromtimestamp(upload_ts).strftime('%Y-%m-%d %H:%M:%S')
            else:
                upload_time = 'N/A'
            chunk_count = doc.get('chunk_count', 0)
            table_count = doc.get('table_count', 0)
            entity_count = doc.get('entity_count', 0)
            faq_count_doc = doc.get('faq_pair_count', 0)
            strategies = ', '.join(doc.get('extraction_strategies', [])) or 'N/A'
            status = doc.get('status', 'indexed')
            file_size = doc.get('file_size_bytes', 0)
            if file_size > 0:
                if file_size > 1024 * 1024:
                    size_str = f"{file_size / (1024*1024):.1f} MB"
                elif file_size > 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size} B"
            else:
                size_str = 'N/A'
            page_count = doc.get('page_count', 0)
            category = doc.get('product_category', 'general')

            # Document card with delete button
            with st.container():
                doc_col1, doc_col2 = st.columns([5, 1])
                with doc_col1:
                    st.markdown(
                        f"📄 **{doc['filename']}**  \n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;📊 {chunk_count} chunks | "
                        f"{table_count} tables | "
                        f"{entity_count} entities | "
                        f"Pages: {page_count if page_count > 0 else 'N/A'} | "
                        f"Size: {size_str} | "
                        f"Category: {category}  \n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;🕐 Uploaded: {upload_time} | "
                        f"Strategies: {strategies} | "
                        f"Status: {'✅' if status == 'indexed' else '⏳'} {status}"
                    )
                with doc_col2:
                    delete_key = f"delete_doc_{idx}_{doc['filename']}"
                    if st.button("🗑️ Delete", key=delete_key, type="secondary"):
                        st.session_state[f'confirm_delete_{idx}'] = doc['filename']

            # Confirmation dialog
            if st.session_state.get(f'confirm_delete_{idx}') == doc['filename']:
                st.warning(
                    f"⚠️ Are you sure you want to delete **{doc['filename']}**? "
                    f"This will remove it from MongoDB, FAISS, in-memory index, FAQ pairs, and hybrid search."
                )
                confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 4])
                with confirm_col1:
                    if st.button("✅ Yes, Delete", key=f"confirm_yes_{idx}", type="primary"):
                        with st.spinner(f"Deleting {doc['filename']}..."):
                            delete_result = engine.delete_document(doc['filename'])
                        if delete_result.get('deleted'):
                            st.success(
                                f"✅ Deleted **{doc['filename']}**: "
                                f"{delete_result['chunks_removed']} chunks, "
                                f"{delete_result['faq_pairs_removed']} FAQ pairs, "
                                f"{delete_result['hybrid_docs_removed']} hybrid docs removed. "
                                f"Persistence: {'✅' if delete_result['persistence_deleted'] else '⚠️ N/A'}"
                            )
                        else:
                            st.error(f"❌ Failed to delete: {delete_result.get('error', 'Unknown error')}")
                        # Clear confirmation state
                        del st.session_state[f'confirm_delete_{idx}']
                        time.sleep(1)
                        st.rerun()
                with confirm_col2:
                    if st.button("❌ Cancel", key=f"confirm_no_{idx}"):
                        del st.session_state[f'confirm_delete_{idx}']
                        st.rerun()

            st.markdown("---")

        st.caption("💡 Documents persist across server restarts when MongoDB is connected. No re-upload or re-embedding needed.")
    else:
        st.info("📭 No documents uploaded yet. Use the sidebar to upload banking policy documents.")

    # Engine stats
    st.markdown("---")
    st.markdown("#### 🏗️ Engine Configuration")
    engine = st.session_state.rag_engine
    if engine:
        config_data = {
            "Layer 1 - Caching": f"{'Enabled' if settings.rag.cache_enabled else 'Disabled'} | Threshold: {settings.rag.cache_similarity_threshold:.0%} | TTL: {settings.rag.cache_ttl_seconds}s",
            "Layer 2 - HyDE": f"{'Enabled' if settings.rag.hyde_enabled else 'Disabled'} | Min Query Length: {settings.rag.hyde_min_query_length}",
            "Layer 3 - Retrieval": f"Top-K: {settings.rag.retrieval_top_k} | Chunk Size: {settings.rag.chunk_size_min}-{settings.rag.chunk_size_max}",
            "Layer 4 - CRAG": f"{'Enabled' if settings.rag.crag_enabled else 'Disabled'} | Web Fallback: {settings.rag.crag_web_fallback_enabled}",
            "Layer 5 - Re-Ranking": f"{'Enabled' if settings.rag.rerank_enabled else 'Disabled'} | Top-K: {settings.rag.rerank_top_k} | Diversity: {settings.rag.rerank_diversity_enabled}",
            "Layer 6 - Agentic": f"{'Enabled' if settings.rag.agentic_enabled else 'Disabled'} | Max Iterations: {settings.rag.agentic_max_iterations}",
            "Layer 7 - Validation": f"{'Enabled' if settings.rag.validation_enabled else 'Disabled'} | Hallucination Check: {settings.rag.hallucination_check_enabled}",
            "Layer 8 - Governance": f"{'Enabled' if settings.rag.governance_enabled else 'Disabled'} | Checks: Hall={settings.rag.governance_check1_hallucination}, Bias={settings.rag.governance_check2_bias}, PII={settings.rag.governance_check3_pii}, Compliance={settings.rag.governance_check4_compliance} | Audit: {settings.rag.governance_audit_trail_enabled}",
        }
        for layer, config in config_data.items():
            st.markdown(f"**{layer}:** {config}")

        # Innovation Configuration
        st.markdown("---")
        st.markdown("#### 🚀 RAG Innovations (9 Techniques)")
        innovation_data = {
            "🔗 Hybrid Search (H1)": f"{'✅ Enabled' if getattr(settings.rag, 'hybrid_search_enabled', False) else '❌ Disabled'} | BM25 Weight: {getattr(settings.rag, 'hybrid_search_bm25_weight', 0.3)} | Vector Weight: {getattr(settings.rag, 'hybrid_search_vector_weight', 0.7)} | RRF k={getattr(settings.rag, 'hybrid_search_rrf_k', 60)}",
            "📝 Contextual Retrieval (H1)": f"{'✅ Enabled' if getattr(settings.rag, 'contextual_retrieval_enabled', False) else '❌ Disabled'} | Anthropic technique — context prefix per chunk",
            "📊 RAGAS Evaluation (H1)": f"{'✅ Enabled' if getattr(settings.rag, 'ragas_evaluation_enabled', False) else '❌ Disabled'} | Sample Rate: {getattr(settings.rag, 'ragas_sample_rate', 1.0):.0%}",
            "🕸️ GraphRAG (H2)": f"{'✅ Enabled' if getattr(settings.rag, 'graph_rag_enabled', False) else '❌ Disabled'} | Max Hops: {getattr(settings.rag, 'graph_rag_max_hops', 3)}",
            "🧭 Adaptive RAG (H2)": f"{'✅ Enabled' if getattr(settings.rag, 'adaptive_rag_enabled', False) else '❌ Disabled'} | Routes queries to optimal retrieval strategy",
            "🔀 Query Decomposition (H2)": f"{'✅ Enabled' if getattr(settings.rag, 'query_decomposition_enabled', False) else '❌ Disabled'} | Max Sub-Queries: {getattr(settings.rag, 'query_decomposition_max_sub_queries', 4)}",
            "🪞 Self-RAG (H3)": f"{'✅ Enabled' if getattr(settings.rag, 'self_rag_enabled', False) else '❌ Disabled'} | Max Iterations: {getattr(settings.rag, 'self_rag_max_iterations', 3)}",
            "⚡ Speculative RAG (H3)": f"{'✅ Enabled' if getattr(settings.rag, 'speculative_rag_enabled', False) else '❌ Disabled'} | Drafts: {getattr(settings.rag, 'speculative_rag_num_drafts', 3)} | Parallel draft generation + verification",
            "🌲 RAPTOR (H3)": f"{'✅ Enabled' if getattr(settings.rag, 'raptor_enabled', False) else '❌ Disabled'} | Max Levels: {getattr(settings.rag, 'raptor_max_levels', 4)} | Hierarchical tree indexing",
        }
        for name, config in innovation_data.items():
            st.markdown(f"**{name}:** {config}")

    # ═══════════════════════════════════════════════════════════
    # FAQ TEST PACK — Zero-Cost Test All Questions
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 🧪 FAQ Test Pack — Test All Questions")
    st.caption(
        "**Zero LLM Token Cost** — Compares stored FAQ answers against original PDF answers "
        "using local similarity metrics (cosine similarity, token overlap, key phrase coverage). "
        "No API calls are made during testing."
    )

    engine = st.session_state.rag_engine
    if not engine:
        st.warning("RAG engine not initialized. Please wait for initialization.")
        return

    # Show stored FAQ pair count
    stored_count = 0
    if hasattr(engine, 'faq_engine') and engine.faq_engine:
        stored_count = len(engine.faq_engine.faq_pairs)
    st.info(f"**Stored FAQ Pairs:** {stored_count} | **Test Mode:** Zero-Cost (Local Embeddings Only)")

    # Test configuration
    test_col1, test_col2, test_col3 = st.columns(3)
    with test_col1:
        pass_threshold = st.slider(
            "Pass Threshold",
            min_value=0.3, max_value=0.95, value=0.70, step=0.05,
            key="test_pass_threshold",
            help="Composite score >= this value = PASS"
        )
    with test_col2:
        sample_count = st.number_input(
            "Number of Questions to Test (0 = all)",
            min_value=0, max_value=5000, value=0, step=5,
            key="test_sample_count",
            help="Randomly sample this many questions. 0 = test all questions."
        )
    with test_col3:
        random_seed = st.number_input(
            "Random Seed (for reproducibility)",
            min_value=0, max_value=99999, value=42, step=1,
            key="test_random_seed",
            help="Set a seed for reproducible random sampling. Same seed = same questions."
        )

    # Upload ground truth PDF
    test_pdf = st.file_uploader(
        "Upload Original FAQ PDF (Ground Truth)",
        type=["pdf", "txt"],
        key="test_pdf_upload",
        help="Upload the original FAQ document. The system extracts Q&A pairs and compares them against stored FAQ pairs."
    )

    # Run Test button
    if st.button("🚀 Run Test — Test All Questions (Zero Cost)", key="run_faq_test", type="primary", use_container_width=True):
        if not test_pdf:
            st.warning("Please upload the original FAQ PDF document as ground truth.")
            return

        embed_model = None
        if hasattr(engine, 'indexer') and hasattr(engine.indexer, 'embed_model'):
            embed_model = engine.indexer.embed_model

        test_runner = FAQTestRunner(engine, embedding_model=embed_model)
        test_runner.PASS_THRESHOLD = pass_threshold
        test_runner.WARNING_THRESHOLD = pass_threshold - 0.20

        # Extract ground truth from uploaded PDF
        file_bytes = test_pdf.read()
        test_pdf.seek(0)

        ground_truth_pairs = []
        if test_pdf.name.lower().endswith(".pdf"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                ground_truth_pairs = test_runner.extract_ground_truth_from_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
            ground_truth_pairs = test_runner.extract_ground_truth_from_text(text, test_pdf.name)

        if not ground_truth_pairs:
            st.error("Could not extract any Q&A pairs from the uploaded document. Ensure it contains Q1, Q2, etc. markers.")
            return

        total_in_pdf = len(ground_truth_pairs)
        actual_sample = sample_count if sample_count > 0 else total_in_pdf
        st.info(
            f"Extracted **{total_in_pdf}** Q&A pairs from PDF. "
            f"Testing **{min(actual_sample, total_in_pdf)}** questions "
            f"({'random sample' if sample_count > 0 and sample_count < total_in_pdf else 'all'}) "
            f"against **{stored_count}** stored FAQ pairs. **Zero LLM cost.**"
        )

        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run tests with progress callback
        def progress_callback(current, total, result):
            progress_bar.progress(current / total)
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "NOT_FOUND": "🔍", "ERROR": "💥"}.get(result.status, "❓")
            status_text.markdown(
                f"Testing **{current}/{total}**: {result.question_number} "
                f"{status_icon} {result.status} "
                f"(Composite: {result.composite_score:.2f}, Time: {result.comparison_time_ms:.0f}ms)"
            )

        report = test_runner.run_all_tests(
            ground_truth_pairs,
            sample_count=sample_count,
            random_seed=random_seed if sample_count > 0 else None,
            progress_callback=progress_callback
        )

        # Store report in session state
        st.session_state['last_test_report'] = report

        progress_bar.progress(1.0)
        status_text.markdown("**Testing complete! Zero LLM tokens used.**")

        # Display results
        _render_test_report(report, test_runner)

    # Show previous report if available
    elif 'last_test_report' in st.session_state:
        st.markdown("---")
        st.markdown("**Previous Test Report:**")
        report = st.session_state['last_test_report']
        embed_model = None
        if hasattr(engine, 'indexer') and hasattr(engine.indexer, 'embed_model'):
            embed_model = engine.indexer.embed_model
        test_runner = FAQTestRunner(engine, embedding_model=embed_model)
        _render_test_report(report, test_runner)


def _render_test_report(report: TestReport, test_runner: FAQTestRunner):
    """Render the zero-cost test report in the Admin Panel."""
    st.markdown("---")
    st.markdown("### 📊 Test Report (Zero LLM Cost)")
    st.markdown(
        f"**Report ID:** `{report.report_id}` | **Timestamp:** {report.timestamp} | "
        f"**Sampling:** {report.sampling_method} | **LLM Cost:** $0.00"
    )

    # Summary metrics
    total = report.total_tested
    pass_rate = (report.passed / total * 100) if total > 0 else 0

    sum_col1, sum_col2, sum_col3, sum_col4, sum_col5, sum_col6 = st.columns(6)
    sum_col1.metric("PDF Q&A Pairs", report.total_ground_truth)
    sum_col2.metric("Tested", report.total_tested)
    sum_col3.metric("✅ Passed", report.passed)
    sum_col4.metric("❌ Failed", report.failed)
    sum_col5.metric("⚠️ Warnings", report.warnings)
    sum_col6.metric("🔍 Not Found", report.not_found)

    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    score_col1.metric("Pass Rate", f"{pass_rate:.1f}%")
    score_col2.metric("Avg Composite", f"{report.avg_composite_score:.3f}")
    score_col3.metric("Avg Cosine Sim", f"{report.avg_cosine_similarity:.3f}")
    score_col4.metric("Total Duration", f"{report.total_test_duration_s:.1f}s")

    time_col1, time_col2, time_col3, time_col4 = st.columns(4)
    time_col1.metric("Avg Token Overlap", f"{report.avg_token_overlap:.3f}")
    time_col2.metric("Avg Key Phrase", f"{report.avg_key_phrase_coverage:.3f}")
    time_col3.metric("Avg Completeness", f"{report.avg_answer_completeness:.3f}")
    time_col4.metric("Stored FAQ Pairs", report.total_stored)

    # Pass/Fail bar
    if total > 0:
        pass_pct = report.passed / total
        warn_pct = report.warnings / total
        fail_pct = report.failed / total
        nf_pct = report.not_found / total
        err_pct = report.errors / total
        st.markdown(
            f'<div style="display:flex; height:24px; border-radius:6px; overflow:hidden; margin:10px 0;">'
            f'<div style="width:{pass_pct*100}%; background:#4caf50;" title="Pass {report.passed}"></div>'
            f'<div style="width:{warn_pct*100}%; background:#ff9800;" title="Warning {report.warnings}"></div>'
            f'<div style="width:{fail_pct*100}%; background:#f44336;" title="Fail {report.failed}"></div>'
            f'<div style="width:{nf_pct*100}%; background:#2196f3;" title="Not Found {report.not_found}"></div>'
            f'<div style="width:{err_pct*100}%; background:#9e9e9e;" title="Error {report.errors}"></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Detailed results table
    st.markdown("---")
    st.markdown("#### Detailed Results")

    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        status_filter = st.multiselect(
            "Filter by Status",
            ["PASS", "FAIL", "WARNING", "NOT_FOUND", "ERROR"],
            default=["PASS", "FAIL", "WARNING", "NOT_FOUND", "ERROR"],
            key="test_status_filter"
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Question Number", "Composite Score (Low→High)", "Composite Score (High→Low)", "Completeness"],
            key="test_sort_by"
        )

    filtered_results = [r for r in report.results if r.status in status_filter]

    # Sort
    if sort_by == "Composite Score (Low→High)":
        filtered_results.sort(key=lambda r: r.composite_score)
    elif sort_by == "Composite Score (High→Low)":
        filtered_results.sort(key=lambda r: r.composite_score, reverse=True)
    elif sort_by == "Completeness":
        filtered_results.sort(key=lambda r: r.answer_completeness)
    else:
        # Sort by question number naturally
        def q_sort_key(r):
            nums = re.findall(r'\d+', r.question_number)
            return int(nums[0]) if nums else 0
        filtered_results.sort(key=q_sort_key)

    # Render each result
    for r in filtered_results:
        status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "NOT_FOUND": "🔍", "ERROR": "💥"}.get(r.status, "❓")

        with st.expander(
            f"{status_icon} {r.question_number}: {r.question[:80]}{'...' if len(r.question) > 80 else ''} "
            f"— Composite: {r.composite_score:.2f} | Match: {r.match_type} | {r.comparison_time_ms:.0f}ms",
            expanded=(r.status in ["FAIL", "ERROR", "NOT_FOUND"])
        ):
            # Metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Cosine Sim", f"{r.cosine_similarity:.3f}")
            m2.metric("Token Overlap", f"{r.token_overlap_score:.3f}")
            m3.metric("Key Phrases", f"{r.key_phrase_coverage:.3f}")
            m4.metric("Completeness", f"{r.answer_completeness:.3f}")
            m5.metric("Composite", f"{r.composite_score:.3f}")

            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.markdown(f"**Match Type:** {r.match_type}")
            info_col2.markdown(f"**Question Similarity:** {r.question_similarity:.3f}")
            info_col3.markdown(f"**Time:** {r.comparison_time_ms:.0f}ms")

            if r.failure_reason:
                st.error(f"**Failure Reason:** {r.failure_reason}")

            # Side-by-side comparison
            ans_col1, ans_col2 = st.columns(2)
            with ans_col1:
                st.markdown("**Original Answer (Ground Truth from PDF):**")
                st.text_area(
                    "Original", value=r.original_answer, height=150,
                    key=f"orig_{r.question_number}", disabled=True
                )
            with ans_col2:
                st.markdown("**Stored Answer (FAQ Engine):**")
                st.text_area(
                    "Stored", value=r.stored_answer, height=150,
                    key=f"stored_{r.question_number}", disabled=True
                )

            # Show stored question if different
            if r.stored_question and r.stored_question != r.question:
                st.markdown(f"**Stored Question:** {r.stored_question}")

    # Download buttons
    st.markdown("---")
    st.markdown("#### 📥 Download Report")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_data = test_runner.generate_csv_report(report)
        st.download_button(
            "📄 Download CSV Report",
            data=csv_data,
            file_name=f"faq_test_report_{report.report_id}.csv",
            mime="text/csv",
            key="download_csv_report",
            use_container_width=True
        )
    with dl_col2:
        json_data = test_runner.generate_json_report(report)
        st.download_button(
            "📋 Download JSON Report",
            data=json_data,
            file_name=f"faq_test_report_{report.report_id}.json",
            mime="application/json",
            key="download_json_report",
            use_container_width=True
        )


# ==========================================
# MAIN
# ==========================================
def main():
    if not st.session_state.authenticated:
        render_login()
    else:
        initialize_engine()
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()
