"""
Banking RAG Pipeline - Centralized Prompt Manager Template
All LLM prompts in one place for easy maintenance and versioning.
"""

class PromptManager:
    """Centralized prompt management layer (called dynamically by all services)."""
    
    # ============================================================
    # EXTRACTION PROMPTS
    # ============================================================
    
    VISION_EXTRACTION = """You are analyzing a banking document page image.
Extract ALL content with precise structure:

1. **Text Content**: Extract all visible text, preserving hierarchy (headers, body, footnotes)
2. **Tables**: Convert to Markdown table format with exact column headers and all rows
3. **Formulas**: Linearize mathematical expressions (e.g., "APR = (interest / principal) × (365 / days) × 100")
4. **Visual Elements**: Describe charts, diagrams, logos with their data
5. **Bilingual Content**: Separate English and Arabic text clearly

CRITICAL for product comparison tables:
- Each row must have the correct product name in the first column
- Fees, rates, and features must be aligned to the correct product
- If card names appear as images/logos, identify them from context

Return structured JSON:
{{
    "text": "extracted text content",
    "tables": [{{
        "headers": ["Column1", "Column2", ...],
        "rows": [["val1", "val2", ...], ...],
        "caption": "table description"
    }}],
    "formulas": ["linearized formula 1", ...],
    "entities": [{{
        "name": "entity name",
        "type": "product|fee|rate|policy",
        "value": "associated value"
    }}]
}}"""

    CONTEXTUAL_ENRICHMENT = """You are enriching a document chunk with context.
Given the document summary and a specific chunk, generate a 1-2 sentence context 
prefix that situates this chunk within the broader document.

Document Summary: {document_summary}

Chunk Content: {chunk_text}

Generate a brief context prefix (1-2 sentences) that explains what this chunk 
is about in the context of the overall document. Start with "This chunk describes..."
or "This section covers..." """

    ENTITY_EXTRACTION = """Extract banking entities and relationships from this text.
Focus on: products, fees, rates, policies, requirements, benefits.

Text: {text}

Return JSON:
{{
    "entities": [
        {{"name": "entity name", "type": "product|fee|rate|policy|requirement|benefit", "value": "if applicable"}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "relation": "has_fee|requires|applies_to|governed_by"}}
    ]
}}"""

    # ============================================================
    # RAG LAYER PROMPTS
    # ============================================================
    
    HYDE_GENERATION = """You are a banking domain expert. Given a user query about banking 
products or services, generate a hypothetical ideal answer paragraph that would perfectly 
answer this query. This will be used for semantic search, so include specific banking 
terminology and details.

Query: {query}

Generate a detailed hypothetical answer (1 paragraph):"""

    CRAG_GRADING = """You are grading the relevance of a retrieved chunk to a user query.
Rate how well this chunk answers or relates to the query.

Query: {query}

Chunk: {chunk}

Return JSON:
{{"grade": "correct" or "ambiguous" or "incorrect", 
 "confidence": 0.0-1.0, 
 "reason": "brief explanation"}}"""

    RERANKING = """Score the relevance of this chunk to the query on a scale of 0-10.
Consider: exact match of terms, semantic relevance, completeness of information.

Query: {query}
Chunk: {chunk}

Return JSON: {{"score": N, "reason": "brief explanation"}}"""

    AGENTIC_SUFFICIENCY = """You are evaluating whether the retrieved context is sufficient 
to answer the user's query completely and accurately.

Query: {query}

Retrieved Context:
{context}

Evaluate:
1. Does the context contain enough information to fully answer the query?
2. Are there any aspects of the query that are NOT addressed by the context?
3. Would additional retrieval help?

Return JSON:
{{"sufficient": true/false, 
 "missing_aspects": ["aspect1", ...],
 "sub_queries": ["additional query 1", ...] if not sufficient}}"""

    ANSWER_GENERATION = """You are a professional banking assistant. Answer the user's 
query based ONLY on the provided context. Be precise with numbers, fees, and rates.

RULES:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say so clearly
3. Quote exact figures (fees, rates, percentages) from the context
4. Do not provide financial advice - only factual product information
5. Include relevant disclaimers for investment/insurance products

Context:
{context}

Query: {query}

Provide a clear, accurate answer:"""

    PRODUCT_CLASSIFICATION = """Classify this banking query into one of these categories:
- credit_cards: Credit card products, fees, rewards, cashback
- loans: Personal loans, mortgages, auto loans
- accounts: Savings, current, fixed deposit accounts
- insurance: Life, health, travel insurance
- investments: Mutual funds, stocks, bonds
- general: General banking queries, branch info, digital banking
- unknown: Cannot determine category

Query: {query}

Return JSON: {{"category": "category_name", "confidence": 0.0-1.0, "reasoning": "brief"}}"""

    # ============================================================
    # GOVERNANCE PROMPTS
    # ============================================================
    
    HALLUCINATION_CHECK = """You are a hallucination detector for a banking AI assistant.
Compare the ANSWER against the SOURCE CHUNKS. Check if all claims in the answer 
are supported by the sources.

IMPORTANT: Standard banking product information (fees, rates, features, cashback percentages)
that appears in the source chunks MUST be marked as supported. Do NOT flag factual product 
data as hallucinated.

SOURCE CHUNKS:
{sources}

ANSWER:
{answer}

Return JSON: {{"status": "pass" or "fail", "score": 0.0-1.0 (1=fully supported), 
"issues": [], "action": "approve" or "flag"}}"""

    BIAS_CHECK = """You are a bias detector for a banking AI assistant.
Check if the answer contains discriminatory language, unfair targeting of demographics,
or biased recommendations.

IMPORTANT: Describing different product tiers (Gold, Platinum, etc.) with different fees
and features is NOT bias - it is standard banking product differentiation.

QUERY: {query}
ANSWER: {answer}

Return JSON: {{"status": "pass" or "fail", "score": 0.0-1.0 (1=no bias), 
"issues": [], "action": "approve" or "flag"}}"""

    COMPLIANCE_CHECK = """You are a compliance checker for a banking AI assistant aligned with 
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

    @classmethod
    def get_prompt(cls, name: str, **kwargs) -> str:
        """Get a prompt template by name, optionally formatted with kwargs."""
        template = getattr(cls, name.upper(), None)
        if template is None:
            raise ValueError(f"Prompt template '{name}' not found")
        if kwargs:
            return template.format(**kwargs)
        return template
