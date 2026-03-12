"""
Centralized Prompt Management Layer.
All prompts for the 7-Layer RAG pipeline are managed here.
Called dynamically by all services — no prompts are hardcoded in service code.
"""


class PromptManager:
    """Centralized prompt manager for all RAG pipeline layers."""

    def __init__(self):
        self._templates = {}

    # ==========================================
    # LAYER 2: HyDE
    # ==========================================
    def hyde_generator(self, query: str) -> str:
        return f"""You are a banking domain expert. Given the following customer query, 
write a hypothetical ideal paragraph that would perfectly answer this question.
This paragraph will be used to improve document retrieval.

Customer Query: {query}

Write a detailed, factual paragraph (150-250 words) that a banking policy document might contain 
to answer this query. Include specific details about procedures, fees, limits, and conditions.
Do NOT say "I don't know" — generate a plausible, detailed answer."""

    # ==========================================
    # LAYER 4: CRAG Quality Grading
    # ==========================================
    def crag_quality_grader(self, query: str, chunk_text: str) -> str:
        return f"""You are a document relevance grader for a banking RAG system.

Grade how relevant the following document chunk is to the customer's query.

Customer Query: {query}

Document Chunk:
{chunk_text}

Respond in JSON format:
{{
    "relevance": "correct" | "ambiguous" | "incorrect",
    "score": 0.0 to 1.0,
    "reason": "Brief explanation of why this chunk is or isn't relevant"
}}

- "correct" (score >= 0.7): Chunk directly answers or is highly relevant to the query
- "ambiguous" (score 0.4-0.7): Chunk is partially relevant or tangentially related
- "incorrect" (score < 0.4): Chunk is not relevant to the query at all

CRITICAL — PRODUCT MATCHING: If the query asks about a SPECIFIC product by name (e.g., "SmartSaver Credit Card"), a chunk that discusses a DIFFERENT product (e.g., "Mashreq Cashback Card", "PlatinumPlus") should be graded as "incorrect" (score < 0.3) UNLESS it is a comparison that explicitly includes the queried product. Only grade as "correct" if the chunk contains information specifically about the product named in the query."""

    # ==========================================
    # LAYER 5: Re-Ranking
    # ==========================================
    def rerank_scorer(self, query: str, chunk_text: str) -> str:
        return f"""You are a precision relevance scorer for banking documents.

Score the following document chunk's relevance to the customer query on a scale of 0-10.

Customer Query: {query}

Document Chunk:
{chunk_text}

Respond in JSON format:
{{
    "score": 0.0 to 10.0,
    "reason": "Brief explanation of the relevance score"
}}

Scoring guide:
- 9-10: Directly and completely answers the query about the SPECIFIC product mentioned
- 7-8: Highly relevant, contains key information about the queried product
- 5-6: Moderately relevant, some useful information
- 3-4: Tangentially related
- 0-2: Not relevant or about a DIFFERENT product than the one asked about

CRITICAL: If the query asks about a specific product (e.g., "SmartSaver") but the chunk discusses a different product (e.g., "Cashback Card", "PlatinumPlus"), score it 0-2 regardless of topic similarity."""

    # ==========================================
    # LAYER 6: Agentic RAG
    # ==========================================
    def query_complexity_classifier(self, query: str) -> str:
        return f"""Classify the complexity of this banking customer query.

Query: {query}

Respond in JSON format:
{{
    "complexity": "simple" | "moderate" | "complex",
    "score": 0.0 to 1.0,
    "reason": "Brief explanation"
}}

- "simple" (score < 0.4): Single-topic, straightforward question
- "moderate" (score 0.4-0.7): Multi-part or requires some reasoning
- "complex" (score > 0.7): Cross-product comparison, multi-step calculation, or regulatory analysis"""

    def agentic_planner(self, query: str, context: str) -> str:
        return f"""You are an intelligent banking assistant analyzing whether the retrieved context 
is sufficient to answer the customer's query.

Customer Query: {query}

Retrieved Context:
{context[:3000]}

Analyze the context and respond in JSON format:
{{
    "status": "SUFFICIENT" | "INSUFFICIENT",
    "reasoning": "Detailed analysis of what information is available and what might be missing",
    "missing_info": "Description of any missing information needed to fully answer the query",
    "suggested_sub_queries": ["list of sub-queries to fill gaps, if any"]
}}"""

    # ==========================================
    # LAYER 7: Response Generation & Validation
    # ==========================================
    def response_generator(self, query: str, context: str, sources: str) -> str:
        return f"""You are a professional banking assistant. Answer the customer's query using ONLY 
the provided context. Be accurate, helpful, and specific.

Customer Query: {query}

Context from Banking Documents:
{context[:4000]}

Sources: {sources}

Instructions:
1. Answer ONLY based on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific details (fees, limits, procedures) when available
4. Use professional banking language
5. Structure your answer clearly with bullet points or numbered steps if appropriate
6. Cite the source document when referencing specific policies
7. CRITICAL — PRODUCT SPECIFICITY: If the query asks about a SPECIFIC product by name (e.g., "SmartSaver Credit Card", "PlatinumPlus", "Gold Card"), your answer MUST ONLY include information about THAT EXACT product. Do NOT mix in details from other card products or tiers. If the context contains information about multiple products, carefully extract ONLY the data that belongs to the product mentioned in the query. If you cannot find information specifically about the named product, say so — do NOT substitute information from a different product.
8. CRITICAL — COMPLETE ANSWERS: When the context contains FAQ entries (e.g., Q29, Q42), you MUST reproduce the FULL and COMPLETE answer text from those FAQ entries. Do NOT summarize, shorten, or paraphrase FAQ answers. Include ALL details, conditions, amounts, fees, thresholds, and specifics exactly as stated in the document. The customer expects to see the same complete information that appears in the original banking document."""

    def hallucination_checker(self, query: str, answer: str, context: str) -> str:
        return f"""You are a hallucination detection system for a banking RAG application.

Compare the generated answer against the source context to identify any unsupported claims.

Customer Query: {query}

Generated Answer:
{answer}

Source Context:
{context[:3000]}

Respond in JSON format:
{{
    "is_hallucinated": true | false,
    "confidence": 0.0 to 1.0,
    "verified_claims": ["list of claims that ARE supported by the context"],
    "unsupported_claims": ["list of claims that are NOT supported by the context"],
    "reasoning": "Brief explanation"
}}

A claim is "unsupported" if it states specific facts (numbers, procedures, policies) 
that cannot be found in or reasonably inferred from the source context."""

    # ==========================================
    # PRODUCT ORCHESTRATOR PROMPTS
    # ==========================================
    def product_classifier(self, query: str, product_list: str) -> str:
        return f"""Classify the following banking customer query to the most relevant product category.

Query: {query}

Available Products:
{product_list}

Respond in JSON format:
{{
    "primary_product": "product_id",
    "confidence": 0.0 to 1.0,
    "secondary_products": ["product_id_2"],
    "is_cross_product": true | false,
    "reasoning": "Brief explanation"
}}"""

    def intent_classifier(self, query: str) -> str:
        return f"""Classify the intent of this banking customer query.

Query: {query}

Respond in JSON format:
{{
    "primary_intent": "inquiry" | "complaint" | "transaction" | "account_management" | "general",
    "intent_name": "Human-readable intent name",
    "intent_confidence": 0.0 to 1.0,
    "secondary_intents": [],
    "requires_human_handoff": true | false,
    "reasoning": "Brief explanation"
}}"""

    def risk_assessor(self, query: str, intent: str) -> str:
        return f"""Assess the risk level of this banking customer query.

Query: {query}
Intent: {intent}

Respond in JSON format:
{{
    "risk_score": 0.0 to 1.0,
    "risk_level": "low" | "medium" | "high" | "critical",
    "risk_factors": ["list of risk factors identified"],
    "reasoning": "Brief explanation"
}}

Risk factors to consider:
- Transaction amounts mentioned
- Account closure or modification requests
- Complaints about unauthorized transactions
- Requests involving sensitive personal information
- Regulatory or compliance-related queries"""

    # ==========================================
    # GOVERNANCE PROMPTS
    # ==========================================
    def governance_hallucination_check(self, query: str, response: str, context: str) -> str:
        return f"""You are a governance hallucination checker for a banking AI system.

Verify factual accuracy of the response against the source context.

Query: {query}
Response: {response}
Context: {context[:2000]}

Respond in JSON format:
{{
    "status": "pass" | "fail" | "warning",
    "score": 0.0 to 1.0,
    "issues": ["list of factual issues found"],
    "action": "approve" | "modify" | "block"
}}

IMPORTANT RULES:
- If the response accurately reflects information from the context, status MUST be "pass"
- If the response paraphrases or summarizes context correctly, status MUST be "pass"
- Only set status to "fail" if the response contains fabricated numbers, invented policies, or claims that directly contradict the context
- If the response says it cannot find information and the context genuinely lacks it, status MUST be "pass"
- When in doubt, default to "pass" with a lower score rather than "fail"
- "warning" is for minor inaccuracies that don't change the core meaning"""

    def governance_bias_check(self, response: str) -> str:
        return f"""Check the following banking AI response for bias, toxicity, or unfairness.

Response: {response}

Respond in JSON format:
{{
    "status": "pass" | "fail" | "warning",
    "score": 0.0 to 1.0,
    "issues": ["list of bias/toxicity issues found"],
    "action": "approve" | "modify" | "block"
}}

IMPORTANT RULES:
- Standard banking product descriptions are NOT biased — status MUST be "pass"
- Only set status to "fail" for responses containing discriminatory language, hate speech, or content that unfairly targets specific groups
- Factual differences between product tiers (e.g., different fees for different cards) are NOT bias
- When in doubt, default to 'pass'"""

    def governance_compliance_check(self, query: str, response: str) -> str:
        return f"""Check if this banking AI response complies with CBUAE regulations and banking standards.

Query: {query}
Response: {response}

Respond in JSON format:
{{
    "status": "pass" | "fail" | "warning",
    "score": 0.0 to 1.0,
    "issues": ["list of compliance issues found"],
    "action": "approve" | "modify" | "block",
    "regulations_referenced": ["list of relevant regulations"]
}}

IMPORTANT RULES:
- Standard banking product information (fees, features, cashback rates, eligibility) is NOT a compliance issue — status MUST be "pass"
- Only set status to "fail" for responses that give unauthorized financial advice, promise guaranteed returns, or violate consumer protection regulations
- Informational responses about bank products, procedures, and policies should always "pass"
- If the response includes appropriate disclaimers or refers to official channels, that is a positive compliance signal
- When in doubt, default to 'pass' with a note rather than 'fail'"""

    # ==========================================
    # INNOVATION PROMPTS
    # ==========================================
    def contextual_prefix_generator(self, document_summary: str, chunk_text: str) -> str:
        return f"""Generate a brief contextual prefix (1-2 sentences) for this document chunk.
The prefix should situate the chunk within the broader document context.

Document Summary: {document_summary}

Chunk Text: {chunk_text[:500]}

Respond with ONLY the contextual prefix text (no JSON, no explanation)."""

    def graph_entity_extractor(self, text: str) -> str:
        return f"""Extract entities and relationships from this banking document text.

Text: {text[:1500]}

Respond in JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "product|fee|limit|procedure|regulation|organization", "attributes": {{}}}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "type": "has_fee|requires|applies_to|governed_by", "attributes": {{}}}}
    ]
}}"""
