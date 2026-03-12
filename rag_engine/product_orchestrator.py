"""
Product Orchestrator — Product Classification, Intent Analysis, Risk Assessment & Routing.
Routes queries to the appropriate product-specific vector stores.
"""
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from openai import OpenAI


# ==========================================
# PRODUCT CATALOG
# ==========================================
PRODUCT_CATALOG = {
    "accounts": {"name": "Accounts", "collection": "accounts", "keywords": ["account", "savings", "current", "deposit", "balance", "statement"]},
    "cards": {"name": "Cards", "collection": "cards", "keywords": ["card", "credit card", "debit card", "visa", "mastercard", "chargeback", "limit"]},
    "loans": {"name": "Loans", "collection": "loans", "keywords": ["loan", "mortgage", "personal loan", "interest rate", "emi", "installment"]},
    "transfers": {"name": "Fund Transfers", "collection": "transfers", "keywords": ["transfer", "wire", "remittance", "swift", "iban", "beneficiary"]},
    "insurance": {"name": "Insurance", "collection": "insurance", "keywords": ["insurance", "policy", "claim", "premium", "coverage"]},
    "investments": {"name": "Investments", "collection": "investments", "keywords": ["investment", "mutual fund", "stocks", "portfolio", "trading"]},
}

GENERAL_COLLECTION = "general"


# ==========================================
# DATA CLASSES
# ==========================================
@dataclass
class IntentResult:
    primary_intent: str = "inquiry"
    intent_name: str = "General Inquiry"
    intent_confidence: float = 0.8
    secondary_intents: List[str] = field(default_factory=list)
    requires_human_handoff: bool = False
    reasoning: str = ""


@dataclass
class RiskResult:
    risk_score: float = 0.1
    risk_level: str = "low"
    risk_label: str = "Low Risk"
    risk_color: str = "#4caf50"
    risk_action: str = "proceed"
    risk_factors: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ConfidenceResult:
    product_confidence: float = 0.8
    overall_confidence: float = 0.8
    keyword_match_score: float = 0.7
    semantic_alignment_score: float = 0.8
    reasoning: str = ""


@dataclass
class RoutingResult:
    routing_reasons: List[Dict[str, Any]] = field(default_factory=list)
    fallback_applied: bool = False
    cross_product_routing: bool = False


@dataclass
class ClassificationResult:
    primary_product: str = GENERAL_COLLECTION
    primary_product_name: str = "General"
    confidence: ConfidenceResult = field(default_factory=ConfidenceResult)
    classification_method: str = "keyword"
    is_cross_product: bool = False
    secondary_products: List[str] = field(default_factory=list)
    intent: IntentResult = field(default_factory=IntentResult)
    risk: RiskResult = field(default_factory=RiskResult)
    routing: RoutingResult = field(default_factory=RoutingResult)
    query_fingerprint: str = ""


@dataclass
class OrchestrationResult:
    classification: ClassificationResult = field(default_factory=ClassificationResult)
    routed_collections: List[str] = field(default_factory=list)
    should_handoff_to_human: bool = False
    context_summary: str = ""


# ==========================================
# PRODUCT ORCHESTRATOR
# ==========================================
class ProductOrchestrator:
    """Routes queries to product-specific collections based on classification."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings

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

    def _keyword_classify(self, query: str) -> tuple:
        """Fast keyword-based classification."""
        query_lower = query.lower()
        best_product = GENERAL_COLLECTION
        best_score = 0.0
        best_name = "General"

        for product_id, info in PRODUCT_CATALOG.items():
            score = sum(1 for kw in info["keywords"] if kw in query_lower) / len(info["keywords"])
            if score > best_score:
                best_score = score
                best_product = product_id
                best_name = info["name"]

        return best_product, best_name, best_score

    def route(self, query: str) -> OrchestrationResult:
        """Classify query and route to appropriate product collections."""
        # Step 1: Keyword classification
        product_id, product_name, kw_score = self._keyword_classify(query)
        fingerprint = hashlib.md5(query.encode()).hexdigest()[:12]

        classification_method = "keyword"
        is_cross_product = False
        secondary_products = []

        # Step 2: LLM classification if keyword confidence is low
        if kw_score < self.settings.rag.orchestrator_keyword_confidence_threshold and self.settings.rag.orchestrator_use_llm:
            classification_method = "llm"
            try:
                from prompts.prompt_manager import PromptManager
                pm = PromptManager()
                product_list = "\n".join([f"- {pid}: {info['name']}" for pid, info in PRODUCT_CATALOG.items()])
                result = self._call_llm(pm.product_classifier(query, product_list))
                parsed = self._parse_json(result)
                if parsed.get("primary_product") in PRODUCT_CATALOG:
                    product_id = parsed["primary_product"]
                    product_name = PRODUCT_CATALOG[product_id]["name"]
                    kw_score = parsed.get("confidence", kw_score)
                    is_cross_product = parsed.get("is_cross_product", False)
                    secondary_products = parsed.get("secondary_products", [])
            except Exception:
                pass

        # Step 3: Intent classification
        intent = IntentResult()
        try:
            from prompts.prompt_manager import PromptManager
            pm = PromptManager()
            intent_result = self._call_llm(pm.intent_classifier(query))
            intent_parsed = self._parse_json(intent_result)
            intent = IntentResult(
                primary_intent=intent_parsed.get("primary_intent", "inquiry"),
                intent_name=intent_parsed.get("intent_name", "General Inquiry"),
                intent_confidence=intent_parsed.get("intent_confidence", 0.8),
                secondary_intents=intent_parsed.get("secondary_intents", []),
                requires_human_handoff=intent_parsed.get("requires_human_handoff", False),
                reasoning=intent_parsed.get("reasoning", ""),
            )
        except Exception:
            pass

        # Step 4: Risk assessment
        risk = RiskResult()
        try:
            from prompts.prompt_manager import PromptManager
            pm = PromptManager()
            risk_result = self._call_llm(pm.risk_assessor(query, intent.intent_name))
            risk_parsed = self._parse_json(risk_result)
            risk_score = risk_parsed.get("risk_score", 0.1)
            risk_level = risk_parsed.get("risk_level", "low")

            risk_labels = {"low": "Low Risk", "medium": "Medium Risk", "high": "High Risk", "critical": "Critical Risk"}
            risk_colors = {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336", "critical": "#b71c1c"}
            risk_actions = {"low": "proceed", "medium": "proceed_with_caution", "high": "escalate", "critical": "block_and_handoff"}

            risk = RiskResult(
                risk_score=risk_score,
                risk_level=risk_level,
                risk_label=risk_labels.get(risk_level, "Unknown"),
                risk_color=risk_colors.get(risk_level, "#9e9e9e"),
                risk_action=risk_actions.get(risk_level, "proceed"),
                risk_factors=risk_parsed.get("risk_factors", []),
                reasoning=risk_parsed.get("reasoning", ""),
            )
        except Exception:
            pass

        # Build routing
        routed_collections = [product_id]
        routing_reasons = [{
            "collection": product_id,
            "product": product_name,
            "reason": f"Primary product match ({classification_method})",
            "weight": 1.0,
        }]

        if is_cross_product and secondary_products:
            for sp in secondary_products:
                if sp in PRODUCT_CATALOG:
                    routed_collections.append(sp)
                    routing_reasons.append({
                        "collection": sp,
                        "product": PRODUCT_CATALOG[sp]["name"],
                        "reason": "Secondary product (cross-product query)",
                        "weight": 0.5,
                    })

        # Always include general collection as fallback
        if GENERAL_COLLECTION not in routed_collections:
            routed_collections.append(GENERAL_COLLECTION)
            routing_reasons.append({
                "collection": GENERAL_COLLECTION,
                "product": "General",
                "reason": "Fallback collection",
                "weight": 0.2,
            })

        classification = ClassificationResult(
            primary_product=product_id,
            primary_product_name=product_name,
            confidence=ConfidenceResult(
                product_confidence=kw_score,
                overall_confidence=kw_score * 0.7 + intent.intent_confidence * 0.3,
                keyword_match_score=kw_score,
                semantic_alignment_score=kw_score,
                reasoning=f"Classified as {product_name} via {classification_method}",
            ),
            classification_method=classification_method,
            is_cross_product=is_cross_product,
            secondary_products=secondary_products,
            intent=intent,
            risk=risk,
            routing=RoutingResult(
                routing_reasons=routing_reasons,
                fallback_applied=GENERAL_COLLECTION in routed_collections,
                cross_product_routing=is_cross_product,
            ),
            query_fingerprint=fingerprint,
        )

        should_handoff = risk.risk_level == "critical" or intent.requires_human_handoff

        return OrchestrationResult(
            classification=classification,
            routed_collections=routed_collections,
            should_handoff_to_human=should_handoff,
            context_summary=f"Product: {product_name} | Intent: {intent.intent_name} | Risk: {risk.risk_label}",
        )
