"""
RAG Pipeline Tracer — Instruments the 7-Layer RAG Pipeline with Langfuse.

This module provides a high-level wrapper that instruments the SevenLayerRAG
pipeline without modifying the core engine code. It wraps each layer execution,
LLM call, and governance check with Langfuse tracing.

Design Principles:
  - Non-invasive: Called FROM the pipeline, does not modify pipeline logic
  - Graceful degradation: All methods are no-ops if Langfuse is disabled
  - Production-safe: Tracing errors never break the RAG pipeline
  - Comprehensive: Captures inputs, outputs, latency, tokens, and scores

Integration Points:
  1. process_query() → start_pipeline_trace() / end_pipeline_trace()
  2. _layerN_*()    → trace_layer()
  3. _call_llm()    → trace_llm_call()
  4. governance     → trace_governance()
  5. FAQ router     → trace_faq_routing()
  6. turbo pipeline → trace_turbo_pipeline()
"""

import time
import traceback
from typing import Optional, Dict, Any, List, Tuple

from observability.langfuse_integration import (
    get_langfuse_tracker,
    TraceHandle,
    SpanHandle,
    GenerationHandle,
)


class RAGPipelineTracer:
    """
    Instruments the 7-Layer RAG pipeline with Langfuse observability.
    
    Usage in SevenLayerRAG.process_query():
    
        tracer = RAGPipelineTracer()
        trace = tracer.start_pipeline_trace(query, turbo_mode=False)
        
        # Layer 1
        l1_span = tracer.trace_layer_start(trace, 1, "Semantic Cache", {"query": query})
        l1_result, cached = self._layer1_cache(query)
        tracer.trace_layer_end(l1_span, l1_result)
        
        # ... repeat for each layer ...
        
        # Governance
        tracer.trace_governance(trace, gov_result)
        
        # End
        tracer.end_pipeline_trace(trace, response)
    """

    def __init__(self):
        self._tracker = get_langfuse_tracker()

    @property
    def is_enabled(self) -> bool:
        return self._tracker.is_enabled

    # ═══════════════════════════════════════════════════════════════
    # PIPELINE TRACE — One trace per query
    # ═══════════════════════════════════════════════════════════════

    def start_pipeline_trace(
        self,
        query: str,
        turbo_mode: bool = False,
        user_id: str = "anonymous",
        session_id: str = "",
    ) -> TraceHandle:
        """Start a pipeline trace for a user query.
        
        Args:
            query: The user's question
            turbo_mode: Whether turbo mode is active
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            TraceHandle for the pipeline
        """
        tags = ['banking-rag']
        if turbo_mode:
            tags.append('turbo-mode')

        metadata = {
            'turbo_mode': turbo_mode,
            'pipeline_version': '7-layer-v2.0',
        }

        return self._tracker.start_trace(
            query=query,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )

    def end_pipeline_trace(
        self,
        trace: TraceHandle,
        response: Any,  # RAGResponse
    ):
        """End the pipeline trace with the final response.
        
        Args:
            trace: The TraceHandle from start_pipeline_trace()
            response: The RAGResponse object
        """
        try:
            output = {
                'answer_preview': response.answer[:500] if response.answer else "",
                'confidence': response.confidence,
                'validation_status': response.validation_status,
                'total_duration_ms': response.total_duration_ms,
                'pipeline_stopped_at': response.pipeline_stopped_at,
                'sources_count': len(response.sources),
                'layers_executed': len(response.layer_results),
            }

            metadata = {
                'sources': response.sources[:5],  # Top 5 sources
            }

            self._tracker.end_trace(trace, output=output, metadata=metadata)

            # Emit pipeline quality scores
            faq_tier = ""
            faq_sim = 0.0
            for lr in response.layer_results:
                if lr.layer_name == "FAQ Smart Router":
                    if lr.status == "TIER_1_EXACT":
                        faq_tier = "exact"
                    elif lr.status == "TIER_2_FUZZY":
                        faq_tier = "fuzzy"
                    elif lr.status == "TIER_3_NOVEL":
                        faq_tier = "novel"
                    faq_sim_str = lr.details.get('similarity', lr.details.get('best_similarity', '0'))
                    try:
                        faq_sim = float(faq_sim_str)
                    except (ValueError, TypeError):
                        faq_sim = 0.0

            self._tracker.emit_pipeline_scores(
                trace=trace,
                confidence=response.confidence,
                validation_status=response.validation_status,
                faq_tier=faq_tier,
                faq_similarity=faq_sim,
                innovation_results=getattr(response, 'innovation_results', None),
            )

        except Exception as e:
            print(f"[LANGFUSE_TRACER] Error ending pipeline trace: {e}")

    # ═══════════════════════════════════════════════════════════════
    # LAYER TRACING — One span per pipeline layer
    # ═══════════════════════════════════════════════════════════════

    def trace_layer_start(
        self,
        trace: TraceHandle,
        layer_number: int,
        layer_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> SpanHandle:
        """Start tracing a pipeline layer.
        
        Args:
            trace: Parent TraceHandle
            layer_number: Layer number (0-8)
            layer_name: Layer name
            input_data: Input to this layer
            
        Returns:
            SpanHandle for the layer
        """
        span_name = f"Layer {layer_number}: {layer_name}"
        metadata = {
            'layer_number': layer_number,
            'layer_name': layer_name,
        }

        return self._tracker.start_span(
            trace=trace,
            name=span_name,
            input_data=input_data,
            metadata=metadata,
        )

    def trace_layer_end(
        self,
        span: SpanHandle,
        layer_result: Any,  # LayerResult
    ):
        """End tracing a pipeline layer.
        
        Args:
            span: The SpanHandle from trace_layer_start()
            layer_result: The LayerResult from the layer execution
        """
        try:
            output = {
                'status': layer_result.status,
                'duration_ms': layer_result.duration_ms,
            }

            # Include key details but limit size for Langfuse
            details = layer_result.details.copy() if layer_result.details else {}
            # Remove large nested lists to keep traces readable
            for key in ['chunk_details', 'grading_details', 'ranking_details', 'reasoning_steps']:
                if key in details and isinstance(details[key], list) and len(details[key]) > 5:
                    details[key] = details[key][:5]  # Keep first 5 items
                    details[f'{key}_truncated'] = True

            output['details'] = details

            # Set level based on status
            level = "DEFAULT"
            if layer_result.status in ("skipped", "cache_miss"):
                level = "DEBUG"
            elif layer_result.status in ("cache_hit",):
                level = "DEFAULT"
            elif layer_result.status in ("error",):
                level = "ERROR"

            self._tracker.end_span(
                handle=span,
                output=output,
                level=level,
                status_message=f"{layer_result.layer_name}: {layer_result.status} ({layer_result.duration_ms:.0f}ms)",
            )
        except Exception as e:
            print(f"[LANGFUSE_TRACER] Error ending layer span: {e}")

    # ═══════════════════════════════════════════════════════════════
    # LLM CALL TRACING — One generation per OpenAI API call
    # ═══════════════════════════════════════════════════════════════

    def trace_llm_call(
        self,
        trace: TraceHandle,
        name: str,
        model: str,
        prompt: str,
        response_text: str,
        usage: Optional[Dict[str, int]] = None,
        parent_span: Optional[SpanHandle] = None,
        duration_ms: float = 0.0,
    ):
        """Track a complete LLM call (prompt → response).
        
        This is a convenience method that creates and immediately ends a generation.
        For long-running LLM calls, use start_generation/end_generation directly.
        
        Args:
            trace: Parent TraceHandle
            name: Generation name (e.g., "HyDE Generation")
            model: Model name
            prompt: The prompt text (truncated for storage)
            response_text: The LLM response
            usage: Token usage {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
            parent_span: Optional parent span
            duration_ms: Duration in milliseconds
        """
        gen = self._tracker.start_generation(
            trace=trace,
            name=name,
            model=model,
            input_text=prompt[:2000],  # Truncate long prompts
            parent_span=parent_span,
        )

        self._tracker.end_generation(
            handle=gen,
            output_text=response_text[:2000],  # Truncate long responses
            usage=usage,
            metadata={'duration_ms': duration_ms} if duration_ms > 0 else None,
        )

    # ═══════════════════════════════════════════════════════════════
    # GOVERNANCE TRACING — Track all 4 governance checks
    # ═══════════════════════════════════════════════════════════════

    def trace_governance(
        self,
        trace: TraceHandle,
        governance_result: Any,  # GovernanceResult
    ):
        """Trace governance checks and emit scores.
        
        Creates a span for the governance layer and emits individual scores
        for each of the 4 checks (hallucination, bias, PII, compliance).
        
        Args:
            trace: Parent TraceHandle
            governance_result: GovernanceResult from governance_engine.py
        """
        if governance_result is None:
            return

        try:
            # Create governance span
            gov_span = self._tracker.start_span(
                trace=trace,
                name="Layer 8: AI Governance (Four-Check System)",
                input_data={'checks_count': len(governance_result.checks)},
                metadata={
                    'overall_status': governance_result.overall_status,
                    'escalated_to_human': governance_result.escalated_to_human,
                },
            )

            # Build output with check details
            checks_summary = []
            for check in governance_result.checks:
                checks_summary.append({
                    'check_number': check.check_number,
                    'check_name': check.check_name,
                    'status': check.status,
                    'score': check.score,
                    'action_taken': check.action_taken,
                    'duration_ms': check.duration_ms,
                })

            self._tracker.end_span(
                handle=gov_span,
                output={
                    'overall_status': governance_result.overall_status,
                    'checks': checks_summary,
                    'modifications_made': governance_result.modifications_made,
                    'escalated_to_human': governance_result.escalated_to_human,
                    'retry_count': governance_result.retry_count,
                    'total_duration_ms': governance_result.total_duration_ms,
                },
            )

            # Emit governance scores to Langfuse
            self._tracker.emit_governance_scores(
                trace=trace,
                governance_result=governance_result,
                governance_span=gov_span,
            )

        except Exception as e:
            print(f"[LANGFUSE_TRACER] Error tracing governance: {e}")

    # ═══════════════════════════════════════════════════════════════
    # FAQ ROUTING TRACING — Track 3-tier FAQ decisions
    # ═══════════════════════════════════════════════════════════════

    def trace_faq_routing(
        self,
        trace: TraceHandle,
        tier: str,
        similarity: float,
        matched_question: str = "",
        question_number: str = "",
        llm_calls: int = 0,
        duration_ms: float = 0.0,
    ):
        """Trace FAQ routing decision.
        
        Args:
            trace: Parent TraceHandle
            tier: "exact", "fuzzy", or "novel"
            similarity: Best match similarity score
            matched_question: The matched FAQ question
            question_number: FAQ question number
            llm_calls: Number of LLM calls used
            duration_ms: Duration in milliseconds
        """
        span = self._tracker.start_span(
            trace=trace,
            name="FAQ Smart Router",
            input_data={'tier': tier, 'similarity': similarity},
            metadata={
                'faq_tier': tier,
                'matched_question': matched_question[:200],
                'question_number': question_number,
                'llm_calls': llm_calls,
            },
        )

        self._tracker.end_span(
            handle=span,
            output={
                'tier': tier,
                'similarity': similarity,
                'matched_question': matched_question[:200],
                'question_number': question_number,
                'pipeline_bypassed': tier in ('exact', 'fuzzy'),
                'llm_calls': llm_calls,
            },
            status_message=f"FAQ Tier {tier.upper()}: sim={similarity:.4f}, LLM calls={llm_calls}",
        )

    # ═══════════════════════════════════════════════════════════════
    # INNOVATION TRACING — Track RAG innovations
    # ═══════════════════════════════════════════════════════════════

    def trace_innovations(
        self,
        trace: TraceHandle,
        innovation_results: Dict[str, Any],
    ):
        """Trace RAG innovation results (Hybrid Search, GraphRAG, RAPTOR, etc.).
        
        Args:
            trace: Parent TraceHandle
            innovation_results: Dict of innovation results from the pipeline
        """
        if not innovation_results:
            return

        try:
            span = self._tracker.start_span(
                trace=trace,
                name="RAG Innovations",
                input_data={'innovations_active': list(innovation_results.keys())},
            )

            self._tracker.end_span(
                handle=span,
                output=innovation_results,
                status_message=f"Innovations: {', '.join(innovation_results.keys())}",
            )
        except Exception as e:
            print(f"[LANGFUSE_TRACER] Error tracing innovations: {e}")

    # ═══════════════════════════════════════════════════════════════
    # DOCUMENT INDEXING TRACING — Track document uploads
    # ═══════════════════════════════════════════════════════════════

    def trace_document_indexing(
        self,
        filename: str,
        chunk_count: int,
        extraction_method: str = "standard",
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Trace a document indexing operation.
        
        Args:
            filename: Document filename
            chunk_count: Number of chunks created
            extraction_method: "standard", "advanced", "vision"
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
        """
        trace = self._tracker.start_trace(
            query=f"[Document Indexing] {filename}",
            metadata={
                'operation': 'document_indexing',
                'filename': filename,
                'extraction_method': extraction_method,
                **(metadata or {}),
            },
            tags=['banking-rag', 'document-indexing'],
        )

        self._tracker.end_trace(
            trace,
            output={
                'filename': filename,
                'chunk_count': chunk_count,
                'extraction_method': extraction_method,
                'duration_ms': duration_ms,
            },
        )

    # ═══════════════════════════════════════════════════════════════
    # FLUSH — Ensure events are sent
    # ═══════════════════════════════════════════════════════════════

    def flush(self):
        """Flush pending events to Langfuse."""
        self._tracker.flush()
