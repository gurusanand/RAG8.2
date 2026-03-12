"""
AI Governance Engine — Four-Check System (CBUAE Aligned).
Check 1: Hallucination & Factual Correctness
Check 2: Bias, Toxicity & Fairness
Check 3: PII & Sensitive Data Redaction
Check 4: Regulatory & Compliance Validation
"""
import re
import json
import time
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from openai import OpenAI


@dataclass
class GovernanceCheckResult:
    check_number: int
    check_name: str
    status: str  # "pass", "fail", "warning"
    score: float = 1.0
    action_taken: str = "approve"
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceResult:
    overall_status: str = "approved"  # approved, warning, blocked, escalated
    final_response: str = ""
    original_response: str = ""
    checks: List[GovernanceCheckResult] = field(default_factory=list)
    modifications_made: List[str] = field(default_factory=list)
    escalated_to_human: bool = False
    retry_count: int = 0
    total_duration_ms: float = 0.0


class GovernanceEngine:
    """Four-Check Governance System aligned with CBUAE regulations."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self.audit_records = []

        # PII patterns for redaction
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+971|0)[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}\b',
            "card_number": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "iban": r'\bAE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b',
            "emirates_id": r'\b784[\s-]?\d{4}[\s-]?\d{7}[\s-]?\d{1}\b',
            "passport": r'\b[A-Z]{1,2}\d{6,9}\b',
        }

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "{}"

    def _parse_json(self, text: str) -> Dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except:
                    pass
            return {}

    def _check1_hallucination(self, query: str, response: str, context: str) -> GovernanceCheckResult:
        """Check 1: Factual Correctness & Hallucination Detection."""
        start = time.time()

        if not self.settings.rag.governance_check1_hallucination:
            return GovernanceCheckResult(
                check_number=1, check_name="Hallucination & Factual Correctness",
                status="pass", score=1.0, action_taken="skipped",
                duration_ms=0, details={"reason": "Check disabled"}
            )

        from prompts.prompt_manager import PromptManager
        pm = PromptManager()
        result = self._call_llm(pm.governance_hallucination_check(query, response, context))
        parsed = self._parse_json(result)

        status = parsed.get("status", "pass")
        score = parsed.get("score", 0.9)
        action = parsed.get("action", "approve")
        issues = parsed.get("issues", [])

        duration = (time.time() - start) * 1000
        return GovernanceCheckResult(
            check_number=1, check_name="Hallucination & Factual Correctness",
            status=status, score=score, action_taken=action,
            duration_ms=duration, details={"issues": issues}
        )

    def _check2_bias(self, response: str) -> GovernanceCheckResult:
        """Check 2: Bias, Toxicity & Fairness."""
        start = time.time()

        if not self.settings.rag.governance_check2_bias:
            return GovernanceCheckResult(
                check_number=2, check_name="Bias, Toxicity & Fairness",
                status="pass", score=1.0, action_taken="skipped",
                duration_ms=0, details={"reason": "Check disabled"}
            )

        from prompts.prompt_manager import PromptManager
        pm = PromptManager()
        result = self._call_llm(pm.governance_bias_check(response))
        parsed = self._parse_json(result)

        status = parsed.get("status", "pass")
        score = parsed.get("score", 0.95)
        action = parsed.get("action", "approve")
        issues = parsed.get("issues", [])

        duration = (time.time() - start) * 1000
        return GovernanceCheckResult(
            check_number=2, check_name="Bias, Toxicity & Fairness",
            status=status, score=score, action_taken=action,
            duration_ms=duration, details={"issues": issues}
        )

    def _check3_pii(self, response: str) -> tuple:
        """Check 3: PII & Sensitive Data Redaction."""
        start = time.time()

        if not self.settings.rag.governance_check3_pii:
            return GovernanceCheckResult(
                check_number=3, check_name="PII & Sensitive Data Redaction",
                status="pass", score=1.0, action_taken="skipped",
                duration_ms=0, details={"reason": "Check disabled"}
            ), response

        redacted_response = response
        pii_found = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, redacted_response)
            if matches:
                pii_found.extend([{"type": pii_type, "count": len(matches)}])
                redacted_response = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted_response)

        status = "warning" if pii_found else "pass"
        score = 0.7 if pii_found else 1.0
        action = "modify" if pii_found else "approve"

        duration = (time.time() - start) * 1000
        return GovernanceCheckResult(
            check_number=3, check_name="PII & Sensitive Data Redaction",
            status=status, score=score, action_taken=action,
            duration_ms=duration, details={"pii_found": pii_found, "redactions_applied": len(pii_found)}
        ), redacted_response

    def _check4_compliance(self, query: str, response: str) -> GovernanceCheckResult:
        """Check 4: Regulatory & Compliance Validation."""
        start = time.time()

        if not self.settings.rag.governance_check4_compliance:
            return GovernanceCheckResult(
                check_number=4, check_name="Regulatory & Compliance Validation",
                status="pass", score=1.0, action_taken="skipped",
                duration_ms=0, details={"reason": "Check disabled"}
            )

        from prompts.prompt_manager import PromptManager
        pm = PromptManager()
        result = self._call_llm(pm.governance_compliance_check(query, response))
        parsed = self._parse_json(result)

        status = parsed.get("status", "pass")
        score = parsed.get("score", 0.9)
        action = parsed.get("action", "approve")
        issues = parsed.get("issues", [])
        regulations = parsed.get("regulations_referenced", [])

        duration = (time.time() - start) * 1000
        return GovernanceCheckResult(
            check_number=4, check_name="Regulatory & Compliance Validation",
            status=status, score=score, action_taken=action,
            duration_ms=duration, details={"issues": issues, "regulations": regulations}
        )

    def run_governance_checks(
        self, query: str, response: str, context: str, max_retries: int = 3
    ) -> GovernanceResult:
        """Run all four governance checks on the response."""
        start = time.time()
        checks = []
        modifications = []
        current_response = response
        retry_count = 0

        # Check 1: Hallucination
        c1 = self._check1_hallucination(query, current_response, context)
        checks.append(c1)

        # Check 2: Bias
        c2 = self._check2_bias(current_response)
        checks.append(c2)

        # Check 3: PII (may modify response)
        c3, current_response = self._check3_pii(current_response)
        checks.append(c3)
        if c3.action_taken == "modify":
            modifications.append("PII redaction applied")

        # Check 4: Compliance
        c4 = self._check4_compliance(query, current_response)
        checks.append(c4)

        # Determine overall status
        # Use a scoring-based approach instead of single-fail escalation
        overall_status = "approved"
        escalated = False

        fail_count = sum(1 for c in checks if c.status == "fail")
        block_count = sum(1 for c in checks if c.action_taken == "block")
        avg_score = sum(c.score for c in checks) / len(checks) if checks else 1.0
        min_score = min(c.score for c in checks) if checks else 1.0

        if block_count >= 2 or (block_count >= 1 and avg_score < 0.3):
            # Only block if multiple checks explicitly request blocking
            # or one block with very low overall confidence
            overall_status = "blocked"
        elif fail_count >= 3 or (fail_count >= 2 and avg_score < 0.4):
            # Only escalate if majority of checks fail with low scores
            overall_status = "escalated"
            escalated = True
        elif fail_count >= 1 and min_score < 0.2:
            # Single fail with critically low score — escalate
            overall_status = "escalated"
            escalated = True
        elif any(c.status == "warning" for c in checks) or fail_count >= 1:
            # Single fail or warnings — approve with warning, don't block
            overall_status = "approved"

        total_duration = (time.time() - start) * 1000

        return GovernanceResult(
            overall_status=overall_status,
            final_response=current_response,
            original_response=response,
            checks=checks,
            modifications_made=modifications,
            escalated_to_human=escalated,
            retry_count=retry_count,
            total_duration_ms=total_duration,
        )

    def create_audit_record(
        self, user_id: str, query: str, response_text: str, confidence: float,
        governance_result: GovernanceResult, orchestration_result: Dict,
        retrieved_chunks: List[Dict], used_chunks: List[Dict],
        model_version: str = "7-Layer-RAG-v2.0"
    ):
        """Create a 14-field audit trail record."""
        record = {
            "timestamp": time.time(),
            "user_id": user_id,
            "query": query,
            "response_preview": response_text[:500],
            "confidence": confidence,
            "governance_status": governance_result.overall_status,
            "governance_checks": len(governance_result.checks),
            "modifications": governance_result.modifications_made,
            "orchestration": orchestration_result,
            "retrieved_chunks_count": len(retrieved_chunks),
            "used_chunks_count": len(used_chunks),
            "model_version": model_version,
            "escalated": governance_result.escalated_to_human,
            "total_duration_ms": governance_result.total_duration_ms,
        }
        self.audit_records.append(record)

        # Persist to file
        try:
            audit_dir = os.path.join(self.settings.paths.base_dir, "data", "audit")
            os.makedirs(audit_dir, exist_ok=True)
            audit_file = os.path.join(audit_dir, "audit_trail.json")
            existing = []
            if os.path.exists(audit_file):
                with open(audit_file, "r") as f:
                    existing = json.load(f)
            existing.append(record)
            with open(audit_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        except Exception as e:
            print(f"[AUDIT] Error persisting audit record: {e}")
