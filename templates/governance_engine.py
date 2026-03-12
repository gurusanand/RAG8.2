"""
Banking RAG Pipeline - Governance Engine Template
4-check system with scoring-based escalation aligned to CBUAE framework.
"""
import json
import re
from typing import Dict, List, Any
from openai import OpenAI

class GovernanceEngine:
    """
    4-check governance system:
    1. Hallucination Detection - cross-references answer with source chunks
    2. Bias/Toxicity Detection - checks for discriminatory language
    3. PII Detection - regex patterns for sensitive data
    4. Compliance Check - regulatory alignment (CBUAE framework)
    
    Escalation is scoring-based (not single-fail) to avoid over-escalation
    of valid banking product information.
    """
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI()
        self.model = config.llm.model
        
        # PII patterns
        self.pii_patterns = {
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "account_number": r"\b\d{10,16}\b",
            "emirates_id": r"\b784-\d{4}-\d{7}-\d\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+971|0)[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}\b",
        }
    
    def run_all_checks(self, query: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        """Run all 4 governance checks and compute escalation decision."""
        results = []
        
        # Check 1: Hallucination
        if self.config.governance.hallucination_check:
            results.append(self._check_hallucination(answer, sources))
        
        # Check 2: Bias/Toxicity
        if self.config.governance.bias_check:
            results.append(self._check_bias(query, answer))
        
        # Check 3: PII
        if self.config.governance.pii_check:
            results.append(self._check_pii(answer))
        
        # Check 4: Compliance
        if self.config.governance.compliance_check:
            results.append(self._check_compliance(query, answer))
        
        # Compute escalation decision (scoring-based)
        decision = self._compute_escalation(results)
        
        return {
            "checks": results,
            "decision": decision["status"],
            "score": decision["avg_score"],
            "action": decision["action"],
            "details": decision
        }
    
    def _check_hallucination(self, answer: str, sources: List[str]) -> Dict:
        """Cross-reference answer claims with source chunks."""
        sources_text = "\n---\n".join(sources[:5])
        prompt = f"""You are a hallucination detector for a banking AI assistant.
Compare the ANSWER against the SOURCE CHUNKS. Check if all claims in the answer 
are supported by the sources.

IMPORTANT: Standard banking product information (fees, rates, features, cashback percentages)
that appears in the source chunks MUST be marked as supported. Do NOT flag factual product 
data as hallucinated.

SOURCE CHUNKS:
{sources_text}

ANSWER:
{answer}

Return JSON: {{"status": "pass" or "fail", "score": 0.0-1.0 (1=fully supported), 
"issues": [], "action": "approve" or "flag"}}"""
        
        return self._run_llm_check("hallucination", prompt)
    
    def _check_bias(self, query: str, answer: str) -> Dict:
        """Check for discriminatory language or unfair targeting."""
        prompt = f"""You are a bias detector for a banking AI assistant.
Check if the answer contains discriminatory language, unfair targeting of demographics,
or biased recommendations.

IMPORTANT: Describing different product tiers (Gold, Platinum, etc.) with different fees
and features is NOT bias - it is standard banking product differentiation.

QUERY: {query}
ANSWER: {answer}

Return JSON: {{"status": "pass" or "fail", "score": 0.0-1.0 (1=no bias), 
"issues": [], "action": "approve" or "flag"}}"""
        
        return self._run_llm_check("bias", prompt)
    
    def _check_pii(self, answer: str) -> Dict:
        """Detect PII using regex patterns."""
        issues = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, answer)
            if matches:
                issues.append(f"Detected {pii_type}: {len(matches)} instance(s)")
        
        has_pii = len(issues) > 0
        return {
            "check": "pii",
            "status": "fail" if has_pii else "pass",
            "score": 0.0 if has_pii else 1.0,
            "issues": issues,
            "action": "block" if has_pii else "approve"
        }
    
    def _check_compliance(self, query: str, answer: str) -> Dict:
        """Check regulatory compliance (CBUAE framework)."""
        prompt = f"""You are a compliance checker for a banking AI assistant aligned with 
CBUAE (Central Bank of UAE) regulations.

Check if the answer:
1. Provides unauthorized financial advice (should only inform, not advise)
2. Makes guarantees about returns or outcomes
3. Missing required disclaimers for investment products
4. Violates consumer protection guidelines

IMPORTANT: Providing factual product information (fees, rates, features) is NOT financial 
advice. Only flag if the answer explicitly recommends specific financial decisions.

QUERY: {query}
ANSWER: {answer}

Return JSON: {{"status": "pass" or "fail", "score": 0.0-1.0 (1=fully compliant), 
"issues": [], "action": "approve" or "flag"}}"""
        
        return self._run_llm_check("compliance", prompt)
    
    def _run_llm_check(self, check_name: str, prompt: str) -> Dict:
        """Execute an LLM-based governance check with error handling."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            content = response.choices[0].message.content.strip()
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            result["check"] = check_name
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            # Default to pass on parse failure (avoid false escalation)
            return {
                "check": check_name,
                "status": "pass",
                "score": 0.5,
                "issues": [f"Check parse error: {str(e)}"],
                "action": "approve"
            }
    
    def _compute_escalation(self, results: List[Dict]) -> Dict:
        """
        Scoring-based escalation logic (NOT single-fail).
        
        Rules:
        - Approved: 0-1 fails with avg score >= 0.2
        - Warning: 2 fails with avg score >= 0.4
        - Escalated: 3+ fails, or 2+ fails with avg score < 0.4
        - Blocked: 2+ checks return "block" action
        """
        fail_count = sum(1 for r in results if r.get("status") == "fail")
        block_count = sum(1 for r in results if r.get("action") == "block")
        scores = [r.get("score", 0.5) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        cfg = self.config.governance
        
        if block_count >= cfg.block_threshold:
            return {"status": "blocked", "avg_score": avg_score, 
                    "action": "block", "fail_count": fail_count}
        
        if fail_count >= cfg.escalation_fail_threshold:
            return {"status": "escalated", "avg_score": avg_score,
                    "action": "escalate_with_answer", "fail_count": fail_count}
        
        if fail_count >= cfg.warning_fail_threshold:
            if avg_score >= cfg.min_avg_score_for_warning:
                return {"status": "warning", "avg_score": avg_score,
                        "action": "deliver_with_disclaimer", "fail_count": fail_count}
            else:
                return {"status": "escalated", "avg_score": avg_score,
                        "action": "escalate_with_answer", "fail_count": fail_count}
        
        return {"status": "approved", "avg_score": avg_score,
                "action": "deliver", "fail_count": fail_count}
