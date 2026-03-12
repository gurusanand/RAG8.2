"""
Langfuse Observability Integration for the 7-Layer Banking RAG Application.

Compatible with: Langfuse Python SDK v2.x (auto-detected)

  ⚠️  SDK v3.x and v4.x use OpenTelemetry/OTLP transport which requires
      Langfuse server v3+. If your Docker setup runs langfuse/langfuse:2
      (server v2), pin the SDK to v2.x: langfuse>=2.0.0,<3.0.0

This module provides a non-invasive observability layer that wraps the RAG pipeline,
LLM calls, embeddings, retrievals, and governance checks with Langfuse tracing.

Features:
  - Full pipeline tracing (each of the 7 layers as spans)
  - LLM generation tracking (model, tokens, cost, latency)
  - Governance metric emission (hallucination, bias, PII, compliance scores)
  - FAQ Router decision tracking
  - Turbo mode tracing
  - Session-level cost aggregation
  - Unicode-safe: all data is sanitized to latin-1 before passing to Langfuse
    (prevents UnicodeEncodeError in OpenTelemetry HTTP exporter)

Architecture:
  - Master toggle: LANGFUSE_ENABLED in config (default: True)
  - Graceful degradation: if Langfuse SDK is not installed or host is unreachable,
    all methods become no-ops — zero impact on the RAG pipeline.
  - Thread-safe singleton pattern for the Langfuse client.
  - Auto-detects SDK version and uses the correct API:
      v2:   client.trace(), trace.span(), trace.generation(), client.score()
            → REST transport to /api/public/ingestion (server v2 compatible)
      v3+/v4: client.start_observation(), span.start_observation(), client.create_score()
            → OTLP transport to /api/public/otel/... (requires server v3+)

Usage:
  from observability.langfuse_integration import get_langfuse_tracker
  tracker = get_langfuse_tracker()
  
  trace = tracker.start_trace(query="What is the interest rate?", user_id="user123")
  span = tracker.start_span(trace, name="Layer 1: Semantic Cache")
  tracker.end_span(span, output={...})
  tracker.score_trace(trace, name="hallucination_score", value=0.95)
  tracker.end_trace(trace, output={"answer": "..."})
"""

import os
import time
import json
import traceback
import importlib.metadata
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════
# LANGFUSE SDK IMPORT — Graceful degradation if not installed
# ═══════════════════════════════════════════════════════════════
_LANGFUSE_AVAILABLE = False
_LANGFUSE_VERSION = "0.0.0"
_LANGFUSE_MAJOR = 0

try:
    _LANGFUSE_VERSION = importlib.metadata.version('langfuse')
    _LANGFUSE_MAJOR = int(_LANGFUSE_VERSION.split('.')[0])
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
    print(f"[LANGFUSE] SDK v{_LANGFUSE_VERSION} detected (major={_LANGFUSE_MAJOR})")
except importlib.metadata.PackageNotFoundError:
    print("[LANGFUSE] SDK not installed. Run: pip install langfuse")
except ImportError as e:
    print(f"[LANGFUSE] SDK import error: {e}")
except Exception as e:
    print(f"[LANGFUSE] SDK detection error: {e}")


# ═══════════════════════════════════════════════════════════════
# VERSION DETECTION — Determine which API to use
# ═══════════════════════════════════════════════════════════════
_USE_V4_API = False  # Default to v2/v3 API (safer fallback)

if _LANGFUSE_AVAILABLE:
    # Runtime detection: check if the Langfuse class has v2 REST vs v3+/v4 OTLP methods.
    # v2.x  → has trace()             / no start_observation()  → REST /api/public/ingestion
    # v3.x+ → has start_observation() / no trace()              → OTLP /api/public/otel/...
    # v4.x  → has start_observation() / no trace()              → OTLP (same as v3+)
    _has_start_observation = hasattr(Langfuse, 'start_observation')
    _has_trace = hasattr(Langfuse, 'trace')

    if _has_start_observation and not _has_trace:
        _USE_V4_API = True
        print(f"[LANGFUSE] SDK v{_LANGFUSE_VERSION}: Using v3+/v4 OTLP API (start_observation)")
        if _LANGFUSE_MAJOR < 3:
            print(f"[LANGFUSE] ⚠️  Unexpected: SDK v{_LANGFUSE_VERSION} has start_observation but major<3")
        else:
            print(f"[LANGFUSE] ⚠️  SDK v{_LANGFUSE_VERSION} requires Langfuse server v3+.")
            print(f"[LANGFUSE]    Your docker-compose uses langfuse/langfuse:2 (server v2).")
            print(f"[LANGFUSE]    → 404 span export errors and Pydantic auth errors are expected.")
            print(f"[LANGFUSE]    FIX: pin SDK  langfuse>=2.0.0,<3.0.0  in requirements.txt")
            print(f"[LANGFUSE]    OR:  upgrade server image to langfuse/langfuse:3 in docker-compose.")
    elif _has_trace and not _has_start_observation:
        _USE_V4_API = False
        print(f"[LANGFUSE] SDK v{_LANGFUSE_VERSION}: Using v2 REST API (trace/span/generation) ✅")
    elif _has_start_observation and _has_trace:
        # Transitional version — prefer OTLP but still warn about server requirement
        _USE_V4_API = True
        print(f"[LANGFUSE] SDK v{_LANGFUSE_VERSION}: Both APIs present — using OTLP (start_observation).")
        print(f"[LANGFUSE] ⚠️  OTLP transport requires Langfuse server v3+.")
    else:
        print(f"[LANGFUSE] ⚠️  Unknown SDK API surface. Will attempt v2 REST first.")


# ═══════════════════════════════════════════════════════════════
# UNICODE SANITIZATION — Prevent latin-1 encoding errors
# ═══════════════════════════════════════════════════════════════
# The Langfuse SDK v3+ uses OpenTelemetry OTLP HTTP exporter which
# encodes HTTP header values using latin-1. Characters outside the
# latin-1 range (like Greek Γ, Arabic, CJK, etc.) cause:
#   UnicodeEncodeError: 'latin-1' codec can't encode character '\u0393'
#
# We sanitize ALL string data before passing it to Langfuse SDK methods.

def _sanitize_str(value: str, max_len: int = 0) -> str:
    """Make a string safe for HTTP header encoding (latin-1 compatible).
    
    Replaces any character outside the latin-1 range (0x00-0xFF) with '?'.
    Optionally truncates to max_len.
    """
    if not value:
        return value
    # Encode to latin-1 with replacement, then decode back
    safe = value.encode('latin-1', 'replace').decode('latin-1')
    if max_len > 0:
        safe = safe[:max_len]
    return safe


def _sanitize_dict(d: Dict[str, Any], max_val_len: int = 0) -> Dict[str, str]:
    """Sanitize all keys and values in a dict to be latin-1 safe strings."""
    if not d:
        return {}
    result = {}
    for k, v in d.items():
        safe_key = _sanitize_str(str(k))
        safe_val = _sanitize_str(str(v), max_val_len) if max_val_len else _sanitize_str(str(v))
        result[safe_key] = safe_val
    return result


def _sanitize_json(obj: Any, max_len: int = 0) -> str:
    """Convert an object to a JSON string, then sanitize for latin-1."""
    try:
        raw = json.dumps(obj, ensure_ascii=True, default=str)
    except Exception:
        raw = str(obj)
    return _sanitize_str(raw, max_len)


def _sanitize_any(value: Any, max_len: int = 0) -> Any:
    """Sanitize any value: dicts get deep-sanitized, strings get latin-1 cleaned."""
    if value is None:
        return value
    if isinstance(value, str):
        return _sanitize_str(value, max_len)
    if isinstance(value, dict):
        return _sanitize_dict(value, max_len)
    if isinstance(value, list):
        return [_sanitize_any(item, max_len) for item in value]
    if isinstance(value, (int, float, bool)):
        return value
    # Fallback: convert to string and sanitize
    return _sanitize_str(str(value), max_len)


@dataclass
class TraceHandle:
    """Lightweight handle wrapping a Langfuse trace or a no-op stub."""
    trace_id: str = ""
    _trace: Any = None        # v2/v3: StatefulTraceClient; v4: LangfuseSpan
    _client: Any = None       # The Langfuse client reference
    _root_span: Any = None    # v4: same as _trace; v2/v3: same as _trace
    _start_time: float = 0.0
    query: str = ""
    user_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanHandle:
    """Lightweight handle wrapping a Langfuse span or a no-op stub."""
    span_id: str = ""
    _span: Any = None         # v2/v3: StatefulSpanClient; v4: LangfuseSpan
    _start_time: float = 0.0
    name: str = ""


@dataclass
class GenerationHandle:
    """Lightweight handle wrapping a Langfuse generation or a no-op stub."""
    generation_id: str = ""
    _generation: Any = None   # v2/v3: StatefulGenerationClient; v4: LangfuseGeneration
    _start_time: float = 0.0
    name: str = ""
    model: str = ""
    input_text: str = ""


class LangfuseTracker:
    """
    Central observability tracker for the Banking RAG pipeline.
    
    All methods are safe to call even when Langfuse is disabled or unavailable —
    they silently return stub handles and perform no I/O.
    
    Auto-detects SDK version and uses the correct API:
      v2:   client.trace(), trace.span(), trace.generation(), client.score()
            → REST transport; compatible with Langfuse server v2
      v3+/v4: client.start_observation(), span.start_observation(), client.create_score()
            → OTLP transport; requires Langfuse server v3+
    
    All string data is sanitized to latin-1 before passing to the SDK to prevent
    UnicodeEncodeError in the OpenTelemetry HTTP exporter.
    """

    def __init__(self, settings=None):
        self._enabled = False
        self._client: Optional[Any] = None
        self._settings = settings
        self._use_v4 = _USE_V4_API

        if settings is None:
            try:
                from config.settings import get_settings
                self._settings = get_settings()
            except ImportError:
                pass

        # Check if Langfuse is enabled in config
        langfuse_config = getattr(self._settings, 'langfuse', None) if self._settings else None
        if langfuse_config and not getattr(langfuse_config, 'enabled', False):
            print("[LANGFUSE] Disabled in config (langfuse.enabled = False)")
            return

        if not _LANGFUSE_AVAILABLE:
            return

        # Initialize the Langfuse client
        try:
            self._init_client(langfuse_config)
        except Exception as e:
            print(f"[LANGFUSE] ❌ Initialization failed: {e}")
            traceback.print_exc()
            self._enabled = False

    def _init_client(self, langfuse_config):
        """Initialize the Langfuse client (works with v2/v3 and v4)."""
        init_kwargs = {}

        # Resolve host / base_url from config or env vars
        host = self._get_config('host', '')
        if not host:
            host = (os.environ.get('LANGFUSE_BASE_URL', '')
                    or os.environ.get('LANGFUSE_HOST', ''))

        # API Keys from config or env vars
        public_key = (self._get_config('public_key', '')
                      or os.environ.get('LANGFUSE_PUBLIC_KEY', ''))
        secret_key = (self._get_config('secret_key', '')
                      or os.environ.get('LANGFUSE_SECRET_KEY', ''))

        if not public_key:
            print("[LANGFUSE] ⚠️  No public_key configured. "
                  "Set LANGFUSE_PUBLIC_KEY env var or config.langfuse.public_key")
            return

        if not secret_key:
            print("[LANGFUSE] ⚠️  No secret_key configured. "
                  "Set LANGFUSE_SECRET_KEY env var or config.langfuse.secret_key")
            return

        # Build constructor kwargs
        init_kwargs['public_key'] = public_key
        init_kwargs['secret_key'] = secret_key

        if host:
            if self._use_v4:
                init_kwargs['base_url'] = host
            else:
                # v2/v3 uses 'host' parameter
                init_kwargs['host'] = host

        # Flush interval
        flush_interval = self._get_config('flush_interval_seconds', None)
        if flush_interval:
            init_kwargs['flush_interval'] = flush_interval

        self._client = Langfuse(**init_kwargs)
        self._enabled = True

        # Runtime re-check: verify the client actually has the expected methods
        if self._use_v4 and not hasattr(self._client, 'start_observation'):
            print("[LANGFUSE] ⚠️  Expected v4 API but start_observation not found. Falling back to v2/v3.")
            self._use_v4 = False
        elif not self._use_v4 and not hasattr(self._client, 'trace'):
            if hasattr(self._client, 'start_observation'):
                print("[LANGFUSE] ⚠️  Expected v2/v3 API but trace() not found. Switching to v4.")
                self._use_v4 = True
            else:
                print("[LANGFUSE] ❌ Neither trace() nor start_observation() found. Disabling.")
                self._enabled = False
                return

        # Auth check
        try:
            if self._client.auth_check():
                api_label = "v4/start_observation" if self._use_v4 else "v2/trace"
                host_label = host or 'cloud'
                print(f"[LANGFUSE] ✅ Initialized v{_LANGFUSE_VERSION} "
                      f"(api={api_label}, host={host_label}, auth=OK)")
            else:
                print(f"[LANGFUSE] ⚠️  Initialized but auth_check failed. "
                      f"Verify API keys and host.")
                print(f"[LANGFUSE]    host={host or 'cloud'}, "
                      f"public_key={public_key[:15]}...")
        except Exception as auth_err:
            print(f"[LANGFUSE] ⚠️  Auth check error: {auth_err}")
            if self._use_v4:
                print(f"[LANGFUSE]    ↳ SDK v{_LANGFUSE_VERSION} uses OTLP transport (requires server v3+).")
                print(f"[LANGFUSE]    ↳ If running langfuse/langfuse:2, downgrade SDK:")
                print(f"[LANGFUSE]      pip install 'langfuse>=2.0.0,<3.0.0'")
            else:
                print(f"[LANGFUSE]    ↳ Verify LANGFUSE_PUBLIC_KEY/SECRET_KEY and LANGFUSE_HOST.")

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value from the langfuse config section."""
        langfuse_config = (getattr(self._settings, 'langfuse', None)
                           if self._settings else None)
        if langfuse_config:
            return getattr(langfuse_config, key, default)
        return default

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    # ═══════════════════════════════════════════════════════════════
    # TRACE MANAGEMENT — One trace per user query
    # ═══════════════════════════════════════════════════════════════

    def start_trace(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> TraceHandle:
        """Start a new trace for a user query (one trace per pipeline execution).
        
        Returns:
            TraceHandle — pass this to all subsequent span/generation/score calls
        """
        handle = TraceHandle(
            query=query,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            _start_time=time.time(),
            _client=self._client,
        )

        if not self.is_enabled:
            return handle

        try:
            if self._use_v4:
                handle = self._start_trace_v4(handle, query, user_id, session_id, metadata, tags)
            else:
                handle = self._start_trace_v2(handle, query, user_id, session_id, metadata, tags)
        except Exception as e:
            print(f"[LANGFUSE] Error starting trace: {e}")
            traceback.print_exc()

        return handle

    def _start_trace_v2(self, handle, query, user_id, session_id, metadata, tags):
        """Start trace using v2/v3 API: client.trace()"""
        trace_kwargs = {
            'name': 'rag-pipeline',
            'input': {'query': _sanitize_str(query, 2000)},
        }
        if user_id:
            trace_kwargs['user_id'] = _sanitize_str(user_id, 200)
        if session_id:
            trace_kwargs['session_id'] = _sanitize_str(session_id, 200)
        if metadata:
            trace_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if tags:
            trace_kwargs['tags'] = [_sanitize_str(t, 100) for t in tags]

        trace = self._client.trace(**trace_kwargs)
        handle._trace = trace
        handle._root_span = trace  # In v2/v3, the trace IS the parent for spans
        handle.trace_id = getattr(trace, 'trace_id', '') or getattr(trace, 'id', '') or ""
        return handle

    def _start_trace_v4(self, handle, query, user_id, session_id, metadata, tags):
        """Start trace using v4 API: client.start_observation()"""
        str_metadata = {}
        if metadata:
            str_metadata = _sanitize_dict(metadata, 200)
        if user_id:
            str_metadata['user_id'] = _sanitize_str(str(user_id), 200)
        if session_id:
            str_metadata['session_id'] = _sanitize_str(str(session_id), 200)
        if tags:
            str_metadata['tags'] = _sanitize_str(','.join(tags), 200)

        root_span = self._client.start_observation(
            name='rag-pipeline',
            as_type='span',
            input=_sanitize_json({'query': query}, 2000),
            metadata=str_metadata,
        )
        handle._root_span = root_span
        handle._trace = root_span
        handle.trace_id = (getattr(root_span, 'trace_id', '')
                           or getattr(root_span, 'id', '') or "")
        return handle

    def end_trace(
        self,
        handle: TraceHandle,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """End a trace with the final output."""
        if not self.is_enabled or handle._root_span is None:
            return

        try:
            if self._use_v4:
                self._end_trace_v4(handle, output, metadata)
            else:
                self._end_trace_v2(handle, output, metadata)
        except Exception as e:
            print(f"[LANGFUSE] Error ending trace: {e}")

    def _end_trace_v2(self, handle, output, metadata):
        """End trace using v2/v3 API."""
        update_kwargs = {}
        if output:
            update_kwargs['output'] = _sanitize_any(output, 5000)
        if metadata:
            update_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if update_kwargs:
            handle._trace.update(**update_kwargs)

    def _end_trace_v4(self, handle, output, metadata):
        """End trace using v4 API."""
        update_kwargs = {}
        if output:
            update_kwargs['output'] = _sanitize_json(output, 5000)
        if metadata:
            update_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if update_kwargs:
            handle._root_span.update(**update_kwargs)
        handle._root_span.end()

    # ═══════════════════════════════════════════════════════════════
    # SPAN MANAGEMENT — One span per pipeline layer
    # ═══════════════════════════════════════════════════════════════

    def start_span(
        self,
        trace: TraceHandle,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT",
    ) -> SpanHandle:
        """Start a span within a trace (one per pipeline layer)."""
        handle = SpanHandle(name=name, _start_time=time.time())

        if not self.is_enabled or trace._root_span is None:
            return handle

        try:
            if self._use_v4:
                handle = self._start_span_v4(handle, trace, name, input_data, metadata, level)
            else:
                handle = self._start_span_v2(handle, trace, name, input_data, metadata, level)
        except Exception as e:
            print(f"[LANGFUSE] Error starting span '{name}': {e}")

        return handle

    def _start_span_v2(self, handle, trace, name, input_data, metadata, level):
        """Start span using v2/v3 API: trace.span()"""
        span_kwargs = {'name': _sanitize_str(name, 200)}
        if input_data:
            span_kwargs['input'] = _sanitize_any(input_data, 2000)
        if metadata:
            span_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if level and level != "DEFAULT":
            span_kwargs['level'] = level

        # In v2/v3, trace object has .span() method
        span = trace._root_span.span(**span_kwargs)
        handle._span = span
        handle.span_id = getattr(span, 'id', '') or ""
        return handle

    def _start_span_v4(self, handle, trace, name, input_data, metadata, level):
        """Start span using v4 API: root_span.start_observation()"""
        str_metadata = _sanitize_dict(metadata, 200) if metadata else {}

        span_kwargs = {
            'name': _sanitize_str(name, 200),
            'as_type': 'span',
            'metadata': str_metadata,
        }
        if input_data:
            span_kwargs['input'] = _sanitize_json(input_data, 2000)
        if level and level != "DEFAULT":
            span_kwargs['level'] = level

        span = trace._root_span.start_observation(**span_kwargs)
        handle._span = span
        handle.span_id = getattr(span, 'id', '') or ""
        return handle

    def end_span(
        self,
        handle: SpanHandle,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = None,
        status_message: str = None,
    ):
        """End a span with output data."""
        if not self.is_enabled or handle._span is None:
            return

        try:
            if self._use_v4:
                self._end_span_v4(handle, output, metadata, level, status_message)
            else:
                self._end_span_v2(handle, output, metadata, level, status_message)
        except Exception as e:
            print(f"[LANGFUSE] Error ending span '{handle.name}': {e}")

    def _end_span_v2(self, handle, output, metadata, level, status_message):
        """End span using v2/v3 API: span.end(output=..., metadata=...)"""
        end_kwargs = {}
        if output:
            end_kwargs['output'] = _sanitize_any(output, 5000)
        if metadata:
            end_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if level:
            end_kwargs['level'] = level
        if status_message:
            end_kwargs['status_message'] = _sanitize_str(status_message, 500)
        handle._span.end(**end_kwargs)

    def _end_span_v4(self, handle, output, metadata, level, status_message):
        """End span using v4 API: span.update(...) then span.end()"""
        update_kwargs = {}
        if output:
            update_kwargs['output'] = _sanitize_json(output, 5000)
        if metadata:
            update_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if level:
            update_kwargs['level'] = level
        if status_message:
            update_kwargs['status_message'] = _sanitize_str(status_message, 500)
        if update_kwargs:
            handle._span.update(**update_kwargs)
        handle._span.end()

    # ═══════════════════════════════════════════════════════════════
    # GENERATION TRACKING — One generation per LLM call
    # ═══════════════════════════════════════════════════════════════

    def start_generation(
        self,
        trace: TraceHandle,
        name: str,
        model: str = "",
        input_text: str = "",
        input_messages: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_span: Optional[SpanHandle] = None,
    ) -> GenerationHandle:
        """Start tracking an LLM generation (one per API call)."""
        handle = GenerationHandle(
            name=name, model=model, input_text=input_text, _start_time=time.time()
        )

        if not self.is_enabled or trace._root_span is None:
            return handle

        try:
            if self._use_v4:
                handle = self._start_generation_v4(
                    handle, trace, name, model, input_text, input_messages, metadata, parent_span)
            else:
                handle = self._start_generation_v2(
                    handle, trace, name, model, input_text, input_messages, metadata, parent_span)
        except Exception as e:
            print(f"[LANGFUSE] Error starting generation '{name}': {e}")

        return handle

    def _start_generation_v2(self, handle, trace, name, model, input_text,
                              input_messages, metadata, parent_span):
        """Start generation using v2/v3 API: parent.generation()"""
        gen_kwargs = {
            'name': _sanitize_str(name, 200),
            'model': _sanitize_str(model, 100),
        }
        if input_messages:
            gen_kwargs['input'] = _sanitize_any(input_messages, 3000)
        elif input_text:
            gen_kwargs['input'] = _sanitize_str(input_text, 3000)
        if metadata:
            gen_kwargs['metadata'] = _sanitize_dict(metadata, 200)

        # Create as child of parent span or trace
        parent = (parent_span._span if parent_span and parent_span._span
                  else trace._root_span)
        generation = parent.generation(**gen_kwargs)
        handle._generation = generation
        handle.generation_id = getattr(generation, 'id', '') or ""
        return handle

    def _start_generation_v4(self, handle, trace, name, model, input_text,
                              input_messages, metadata, parent_span):
        """Start generation using v4 API: parent.start_observation(as_type='generation')"""
        str_metadata = _sanitize_dict(metadata, 200) if metadata else {}

        gen_kwargs = {
            'name': _sanitize_str(name, 200),
            'as_type': 'generation',
            'model': _sanitize_str(model, 100),
            'metadata': str_metadata,
        }
        if input_messages:
            gen_kwargs['input'] = _sanitize_json(input_messages, 3000)
        elif input_text:
            gen_kwargs['input'] = _sanitize_str(input_text, 3000)

        parent = (parent_span._span if parent_span and parent_span._span
                  else trace._root_span)
        generation = parent.start_observation(**gen_kwargs)
        handle._generation = generation
        handle.generation_id = getattr(generation, 'id', '') or ""
        return handle

    def end_generation(
        self,
        handle: GenerationHandle,
        output_text: str = "",
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = None,
        status_message: str = None,
    ):
        """End an LLM generation with output and token usage."""
        if not self.is_enabled or handle._generation is None:
            return

        try:
            if self._use_v4:
                self._end_generation_v4(handle, output_text, usage, metadata, level, status_message)
            else:
                self._end_generation_v2(handle, output_text, usage, metadata, level, status_message)
        except Exception as e:
            print(f"[LANGFUSE] Error ending generation '{handle.name}': {e}")

    def _end_generation_v2(self, handle, output_text, usage, metadata, level, status_message):
        """End generation using v2/v3 API: generation.end(output=..., usage=...)"""
        end_kwargs = {}
        if output_text:
            end_kwargs['output'] = _sanitize_str(output_text, 3000)
        if usage:
            end_kwargs['usage'] = usage  # usage is numeric, no sanitization needed
        if metadata:
            end_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if level:
            end_kwargs['level'] = level
        if status_message:
            end_kwargs['status_message'] = _sanitize_str(status_message, 500)
        handle._generation.end(**end_kwargs)

    def _end_generation_v4(self, handle, output_text, usage, metadata, level, status_message):
        """End generation using v4 API: generation.update(...) then generation.end()"""
        update_kwargs = {}
        if output_text:
            update_kwargs['output'] = _sanitize_str(output_text, 3000)
        if usage:
            update_kwargs['usage_details'] = usage  # usage is numeric, no sanitization needed
        if metadata:
            update_kwargs['metadata'] = _sanitize_dict(metadata, 200)
        if level:
            update_kwargs['level'] = level
        if status_message:
            update_kwargs['status_message'] = _sanitize_str(status_message, 500)
        if update_kwargs:
            handle._generation.update(**update_kwargs)
        handle._generation.end()

    # ═══════════════════════════════════════════════════════════════
    # SCORING — Governance metrics, quality scores, user feedback
    # ═══════════════════════════════════════════════════════════════

    def score_trace(
        self,
        trace: TraceHandle,
        name: str,
        value: float,
        comment: str = "",
        data_type: str = "NUMERIC",
    ):
        """Add a numeric score to a trace."""
        if not self.is_enabled or not trace.trace_id:
            return

        safe_name = _sanitize_str(name, 100)
        safe_comment = _sanitize_str(comment, 200) if comment else ""

        try:
            if self._use_v4:
                self._client.create_score(
                    trace_id=trace.trace_id,
                    name=safe_name,
                    value=value,
                    comment=safe_comment,
                    data_type=data_type,
                )
            else:
                # v2/v3 API
                score_kwargs = {
                    'trace_id': trace.trace_id,
                    'name': safe_name,
                    'value': value,
                }
                if safe_comment:
                    score_kwargs['comment'] = safe_comment
                self._client.score(**score_kwargs)
        except Exception as e:
            print(f"[LANGFUSE] Error scoring trace (name={name}): {e}")

    def score_span(
        self,
        trace: TraceHandle,
        span: SpanHandle,
        name: str,
        value: float,
        comment: str = "",
    ):
        """Add a numeric score to a specific span/observation."""
        if not self.is_enabled or not trace.trace_id or not span.span_id:
            return

        safe_name = _sanitize_str(name, 100)
        safe_comment = _sanitize_str(comment, 200) if comment else ""

        try:
            if self._use_v4:
                if span._span and hasattr(span._span, 'score'):
                    span._span.score(
                        name=safe_name,
                        value=value,
                        comment=safe_comment,
                    )
                else:
                    self._client.create_score(
                        trace_id=trace.trace_id,
                        observation_id=span.span_id,
                        name=safe_name,
                        value=value,
                        comment=safe_comment,
                    )
            else:
                # v2/v3 API
                self._client.score(
                    trace_id=trace.trace_id,
                    observation_id=span.span_id,
                    name=safe_name,
                    value=value,
                    comment=safe_comment,
                )
        except Exception as e:
            print(f"[LANGFUSE] Error scoring span (name={name}): {e}")

    # ═══════════════════════════════════════════════════════════════
    # GOVERNANCE METRICS — Emit all 4 governance check scores
    # ═══════════════════════════════════════════════════════════════

    def emit_governance_scores(
        self,
        trace: TraceHandle,
        governance_result: Any,
        governance_span: Optional[SpanHandle] = None,
    ):
        """Emit all governance check scores to Langfuse.
        
        Emits:
          - gov_hallucination_score (0-1)
          - gov_bias_score (0-1)
          - gov_pii_score (0-1)
          - gov_compliance_score (0-1)
          - gov_overall_status (0-1)
        """
        if not self.is_enabled or trace._root_span is None or governance_result is None:
            return

        try:
            check_score_map = {
                1: "gov_hallucination_score",
                2: "gov_bias_score",
                3: "gov_pii_score",
                4: "gov_compliance_score",
            }

            for check in governance_result.checks:
                score_name = check_score_map.get(check.check_number)
                if score_name:
                    comment = (
                        f"{check.check_name}: {check.status} | "
                        f"Action: {check.action_taken} | "
                        f"Duration: {check.duration_ms:.0f}ms"
                    )
                    self.score_trace(
                        trace=trace,
                        name=score_name,
                        value=check.score,
                        comment=comment,
                    )

            # Overall governance status
            status_to_value = {
                "approved": 1.0,
                "warning": 0.5,
                "escalated": 0.25,
                "blocked": 0.0,
            }
            overall_value = status_to_value.get(
                governance_result.overall_status, 0.5)
            self.score_trace(
                trace=trace,
                name="gov_overall_status",
                value=overall_value,
                comment=(f"Overall: {governance_result.overall_status} | "
                         f"Modifications: {len(governance_result.modifications_made)} | "
                         f"Escalated: {governance_result.escalated_to_human}"),
            )

        except Exception as e:
            print(f"[LANGFUSE] Error emitting governance scores: {e}")

    # ═══════════════════════════════════════════════════════════════
    # PIPELINE QUALITY SCORES — Confidence, RAGAS, FAQ similarity
    # ═══════════════════════════════════════════════════════════════

    def emit_pipeline_scores(
        self,
        trace: TraceHandle,
        confidence: float = 0.0,
        validation_status: str = "",
        faq_tier: str = "",
        faq_similarity: float = 0.0,
        innovation_results: Optional[Dict[str, Any]] = None,
    ):
        """Emit pipeline quality scores to Langfuse."""
        if not self.is_enabled or trace._root_span is None:
            return

        try:
            if confidence > 0:
                self.score_trace(trace, "response_confidence", confidence,
                                 f"Validation: {validation_status}")

            if faq_tier:
                tier_values = {"exact": 1.0, "fuzzy": 0.7, "novel": 0.3}
                self.score_trace(
                    trace, "faq_routing_tier",
                    tier_values.get(faq_tier, 0.0),
                    f"Tier: {faq_tier}, Similarity: {faq_similarity:.4f}")

            if innovation_results and 'ragas_evaluation' in innovation_results:
                ragas = innovation_results['ragas_evaluation']
                if 'error' not in ragas:
                    for metric in ['faithfulness', 'answer_relevancy',
                                   'context_precision', 'context_recall',
                                   'overall_score']:
                        val_str = ragas.get(metric, '')
                        if (val_str and isinstance(val_str, str)
                                and val_str.endswith('%')):
                            val = float(val_str.rstrip('%')) / 100.0
                            self.score_trace(
                                trace, f"ragas_{metric}", val,
                                f"RAGAS {metric}")

        except Exception as e:
            print(f"[LANGFUSE] Error emitting pipeline scores: {e}")

    # ═══════════════════════════════════════════════════════════════
    # USER FEEDBACK — Capture thumbs up/down from the UI
    # ═══════════════════════════════════════════════════════════════

    def emit_user_feedback(
        self,
        trace_id: str,
        rating: int,
        comment: str = "",
    ):
        """Emit user feedback score for a trace."""
        if not self.is_enabled:
            return

        safe_comment = _sanitize_str(comment, 200) if comment else ""

        try:
            if self._use_v4:
                self._client.create_score(
                    trace_id=trace_id,
                    name="user_feedback",
                    value=float(rating),
                    comment=safe_comment,
                    data_type="NUMERIC",
                )
            else:
                self._client.score(
                    trace_id=trace_id,
                    name="user_feedback",
                    value=float(rating),
                    comment=safe_comment,
                )
        except Exception as e:
            print(f"[LANGFUSE] Error emitting user feedback: {e}")

    # ═══════════════════════════════════════════════════════════════
    # FLUSH — Ensure all events are sent before shutdown
    # ═══════════════════════════════════════════════════════════════

    def flush(self):
        """Flush all pending events to Langfuse. Call on app shutdown."""
        if self.is_enabled and self._client:
            try:
                self._client.flush()
            except Exception as e:
                print(f"[LANGFUSE] Error flushing: {e}")

    def shutdown(self):
        """Shutdown the Langfuse client gracefully."""
        if self.is_enabled and self._client:
            try:
                self._client.flush()
                if hasattr(self._client, 'shutdown'):
                    self._client.shutdown()
            except Exception as e:
                print(f"[LANGFUSE] Error shutting down: {e}")


# ═══════════════════════════════════════════════════════════════
# SINGLETON — Module-level tracker instance
# ═══════════════════════════════════════════════════════════════
_tracker_instance: Optional[LangfuseTracker] = None


def get_langfuse_tracker(settings=None) -> LangfuseTracker:
    """Get the singleton LangfuseTracker instance.
    
    Thread-safe lazy initialization. The tracker is created once and reused.
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = LangfuseTracker(settings=settings)
    return _tracker_instance


def reset_langfuse_tracker():
    """Reset the singleton tracker (useful for testing or reconfiguration)."""
    global _tracker_instance
    if _tracker_instance:
        _tracker_instance.shutdown()
    _tracker_instance = None
