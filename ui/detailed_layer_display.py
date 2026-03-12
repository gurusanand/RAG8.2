"""
Detailed Layer Display — Renders the 7-Layer pipeline results in the Streamlit UI.
Shows chunk-level details, innovation results, governance checks, and orchestration info.
"""
import streamlit as st
from typing import Any


def render_detailed_pipeline(response: Any, show_details: bool = True):
    """Render the detailed pipeline results for a RAG response."""
    if not show_details or not hasattr(response, 'layer_results') or not response.layer_results:
        return

    with st.expander("🔍 Pipeline Execution Details", expanded=False):
        # Summary metrics
        total_ms = response.total_duration_ms if hasattr(response, 'total_duration_ms') else 0
        num_layers = len(response.layer_results)
        st.markdown(f"**Total Pipeline Time:** {total_ms:.0f}ms | **Layers Executed:** {num_layers}")

        if hasattr(response, 'pipeline_stopped_at') and response.pipeline_stopped_at is not None:
            st.info(f"⚡ Pipeline stopped at Layer {response.pipeline_stopped_at} (cache hit)")

        st.markdown("---")

        # Orchestration Result (Layer 0)
        if hasattr(response, 'orchestration_result') and response.orchestration_result:
            _render_orchestration(response.orchestration_result)

        # Layer Results
        for layer in response.layer_results:
            _render_layer(layer)

        # Innovation Results
        if hasattr(response, 'innovation_results') and response.innovation_results:
            _render_innovations(response.innovation_results)

        # Governance Result
        if hasattr(response, 'governance_result') and response.governance_result:
            _render_governance(response.governance_result)


def _render_orchestration(orch_result):
    """Render orchestration/classification details."""
    with st.container():
        st.markdown("#### 🎯 Layer 0: Product Orchestration")
        c = orch_result.classification
        col1, col2, col3 = st.columns(3)
        col1.metric("Product", c.primary_product_name)
        col2.metric("Intent", c.intent.intent_name)
        col3.metric("Risk", c.risk.risk_label)

        if c.is_cross_product:
            st.warning(f"Cross-product query detected. Secondary products: {c.secondary_products}")

        st.markdown("---")


def _render_layer(layer):
    """Render a single layer result."""
    status_icons = {
        "executed": "✅",
        "skipped": "⏭️",
        "cache_hit": "⚡",
        "pass_through": "➡️",
    }
    icon = status_icons.get(layer.status, "🔹")
    duration_str = f"{layer.duration_ms:.0f}ms" if layer.duration_ms else "—"

    with st.container():
        col1, col2, col3 = st.columns([4, 1, 1])
        col1.markdown(f"**{icon} Layer {layer.layer_number}: {layer.layer_name}**")
        col2.markdown(f"`{layer.status}`")
        col3.markdown(f"`{duration_str}`")

        if layer.details:
            details = layer.details

            # Layer-specific rendering
            if layer.layer_number == 1:  # Cache
                if "similarity_score" in details:
                    st.markdown(f"Cache similarity: **{details['similarity_score']}**")

            elif layer.layer_number == 3:  # Retrieval
                if "chunks_retrieved" in details:
                    st.markdown(f"Chunks retrieved: **{details['chunks_retrieved']}**")
                if "retrieval_details" in details:
                    _render_chunk_table(details["retrieval_details"], "Retrieved Chunks")

            elif layer.layer_number == 4:  # CRAG
                if "grading_details" in details:
                    _render_chunk_table(details["grading_details"], "CRAG Grading")

            elif layer.layer_number == 5:  # Re-Ranking
                if "ranking_details" in details:
                    _render_chunk_table(details["ranking_details"], "Re-Ranking Scores")

            elif layer.layer_number == 6:  # Agentic
                if "complexity" in details:
                    st.markdown(f"Query complexity: **{details.get('complexity', 'N/A')}** "
                                f"(score: {details.get('complexity_score', 'N/A')})")
                if "reasoning_steps" in details:
                    for step in details["reasoning_steps"]:
                        st.markdown(f"  - Step {step.get('step', '?')}: {step.get('observation', '')[:200]}")

            elif layer.layer_number == 7:  # Validation
                if "confidence" in details:
                    st.markdown(f"Confidence: **{details['confidence']}** | "
                                f"Status: **{details.get('validation_status', 'N/A')}**")
                if "hallucination_check" in details:
                    hall = details["hallucination_check"]
                    if hall.get("is_hallucinated"):
                        st.warning(f"⚠️ Hallucination detected: {hall.get('unsupported_claims', [])}")

            elif layer.layer_number == 8:  # Governance
                if "checks" in details:
                    for check in details["checks"]:
                        check_icon = "✅" if check.get("status") == "pass" else "⚠️" if check.get("status") == "warning" else "❌"
                        st.markdown(f"  {check_icon} Check {check.get('check_number', '?')}: "
                                    f"{check.get('check_name', 'Unknown')} — "
                                    f"Score: {check.get('score', 'N/A')} | Action: {check.get('action_taken', 'N/A')}")


def _render_chunk_table(chunks_data, title):
    """Render chunk details as a compact table."""
    if not chunks_data:
        return

    with st.expander(f"📋 {title} ({len(chunks_data)} items)", expanded=False):
        for i, chunk in enumerate(chunks_data[:10]):  # Limit to 10
            source = chunk.get("source", "Unknown")
            section = chunk.get("section", "General")
            score = chunk.get("score", chunk.get("similarity_score", chunk.get("combined_score", "N/A")))
            selected = chunk.get("selected", None)

            marker = "✅" if selected else "❌" if selected is False else "🔹"
            st.markdown(f"{marker} **#{i+1}** `{source}` → {section} | Score: `{score}`")

            # Show text preview if available
            text_preview = chunk.get("text_preview", chunk.get("text", ""))
            if text_preview:
                st.caption(text_preview[:200])


def _render_innovations(innovation_results):
    """Render innovation module results."""
    st.markdown("#### 🚀 Innovation Modules")
    for name, result in innovation_results.items():
        if isinstance(result, dict):
            if "error" in result:
                st.markdown(f"❌ **{name}**: `{result['error'][:100]}`")
            else:
                duration = result.get("duration_ms", "—")
                st.markdown(f"✅ **{name}** ({duration}ms)")

                # Show key metrics
                key_metrics = {k: v for k, v in result.items()
                               if k not in ["duration_ms", "error"] and not isinstance(v, (list, dict))}
                if key_metrics:
                    metrics_str = " | ".join([f"{k}: `{v}`" for k, v in list(key_metrics.items())[:5]])
                    st.caption(metrics_str)
    st.markdown("---")


def _render_governance(gov_result):
    """Render governance check results."""
    st.markdown("#### 🛡️ AI Governance (Four-Check System)")
    status_colors = {"approved": "🟢", "warning": "🟡", "blocked": "🔴", "escalated": "🟠"}
    icon = status_colors.get(gov_result.overall_status, "⚪")
    st.markdown(f"**Overall Status:** {icon} {gov_result.overall_status.upper()}")

    if hasattr(gov_result, 'checks'):
        for check in gov_result.checks:
            check_icon = "✅" if check.status == "pass" else "⚠️" if check.status == "warning" else "❌"
            st.markdown(f"  {check_icon} **{check.check_name}** — Score: {check.score:.0%} | Action: {check.action_taken}")

    if hasattr(gov_result, 'modifications_made') and gov_result.modifications_made:
        st.info(f"Modifications applied: {', '.join(gov_result.modifications_made)}")
