"""Query router to determine which tools/agents to use based on the query."""

import logging, sys
from typing import List, Dict, Any, Tuple
from enum import Enum

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from ..config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),                 # Console output
        logging.FileHandler("app.log", mode='a', encoding='utf-8')  # File output
    ]
)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    STRUCTURED_ONLY = "structured_only"
    UNSTRUCTURED_ONLY = "unstructured_only"
    HYBRID = "hybrid"
    UNCLEAR = "unclear"


class QueryAnalysis(BaseModel):
    """Result of query analysis."""
    query_type: QueryType
    confidence: float
    structured_components: List[str]
    unstructured_components: List[str]
    reasoning: str


class QueryRouter:
    """Routes queries to appropriate tools based on content analysis."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

        # Patterns to help LLM & fallback
        self.structured_patterns = [
            "how many", "count", "total", "average", "sum",
            "patient", "died", "death", "age", "gender",
            "score", "pressure", "glucose", "when", "time", "date"
        ]

        self.unstructured_patterns = [
            "what is", "define", "explain", "how to",
            "treatment", "protocol", "guidelines",
            "research", "study", "evidence",
            "mechanism", "diagnosis",
            "symptoms", "signs"
        ]

        # Prompt template for LLM analysis
        self.analysis_prompt = PromptTemplate(
            input_variables=["query", "memory_context", "structured_patterns",
                             "unstructured_patterns"],
            template = """
                <ROLE>
                You are a clinical AI query router. Your task is to classify the query given.
                </ROLE>

                <Data Sources>
                1. Structured clinical data: patient tables (Excel/CSV) containing patient-level information such as measurements, lab results, outcomes, and demographics.
                2. Unstructured clinical documents: PDFs, research papers, clinical guidelines, journal articles, and free-text clinical notes.
                </Data Sources>

                <Structured Indicators>
                {structured_patterns}
                </Structured Indicators>

                <Unstructured Indicators>
                {unstructured_patterns}
                </Unstructured Indicators>

                <Query>
                {query}
                </Query>

                <Context>
                {memory_context}
                </Context>

                <INSTRUCTIONS>
                - Your goal is to determine whether the query should be answered from structured data, unstructured data, both, or if it is unclear.
                - Output QUERY_TYPE as one of: [structured_only, unstructured_only, hybrid, unclear].
                - If the query contains multiple distinct questions separated by punctuation (e.g., ".", "?", ";"), classify and analyze each separately.
                - If the query refers to a specific clinical trial, research paper, journal article, published study, or guideline documents, classify this part as unstructured  — even if similar data fields exist in structured data.
                - Structured queries are generally includes counts, calculations, aggregation, comparisions, filtering,  Time-based queries, etc on patient level structured data.
                - Hybrid applies only when the answer requires combining both structured patient data and unstructured document content.
                - Assign a CONFIDENCE score between 0 and 1 for your classification.
                - List relevant STRUCTURED_COMPONENTS to extract from structured data (empty list if none).
                - List relevant UNSTRUCTURED_COMPONENTS to retrieve from unstructured data (empty list if none).
                - Provide a short REASONING (2–4 sentences) explaining your decision without restating the instructions or query.
                - Respond in exactly the format shown below — no extra commentary or rephrasing.

                <Response format>
                QUERY_TYPE: ...
                CONFIDENCE: ...
                STRUCTURED_COMPONENTS: [...]
                UNSTRUCTURED_COMPONENTS: [...]
                REASONING: ...
                </Response format>

                <RESPONSE_5348_TAG>
                """

            )

    

    def analyze_query(self, query: str, memory_context: Dict[str, Any] = None) -> QueryAnalysis:
        """Analyze query to determine routing strategy."""
        try:
            # processed_query, matched_columns, needs_domain_knowledge, matched_domain_terms = self.preprocess_query(query)
            memory_text = str(memory_context) if memory_context else "No prior context"

            prompt_text = self.analysis_prompt.format(
                query=query,
                memory_context=memory_text,
                structured_patterns=", ".join(self.structured_patterns),
                unstructured_patterns=", ".join(self.unstructured_patterns),
            )

            analysis_text = self.llm(prompt_text)
            analysis_text = analysis_text.split("</RESPONSE_5348_TAG>")[0]
            analysis = self._parse_analysis(analysis_text, query)

            # Inject preprocessed info
            # if matched_columns:
            #     analysis.structured_components = list(set(analysis.structured_components + matched_columns))
            # if needs_domain_knowledge:
            #     analysis.needs_domain_knowledge = True
            #     if "domain_knowledge" not in analysis.suggested_tools:
            #         analysis.suggested_tools.insert(0, "domain_knowledge")
            # if matched_domain_terms:
            #     analysis.reasoning += f" | Domain mapping terms: {', '.join(matched_domain_terms)}"
            return analysis

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._fallback_analysis(query)
        
    def _parse_analysis(self, analysis_text: str, original_query: str) -> QueryAnalysis:
        """Parse LLM analysis response robustly."""
        try:
            print(analysis_text)
            parsed = {
                'query_type': None,
                'confidence': None,
                'structured_components': [],
                'unstructured_components': [],
                'reasoning': ""
            }

            for line in analysis_text.strip().splitlines():
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()

                if key == "QUERY_TYPE":
                    parsed['query_type'] = QueryType(value.lower())
                elif key == "CONFIDENCE" or key == "CONFIDANCE":
                    try:
                        parsed['confidence'] = float(value)
                    except ValueError:
                        parsed['confidence'] = 0.5
                elif key in ["STRUCTURED_COMPONENTS", "UNSTRUCTURED_COMPONENTS"]:
                    parsed[key.lower()] = [i.strip(" []'\"") for i in value.split(',') if i.strip()]
                elif key == "REASONING":
                    parsed['reasoning'] = value

            return QueryAnalysis(
                query_type=parsed['query_type'] or QueryType.HYBRID,
                confidence=parsed['confidence'] or 0.5,
                structured_components=parsed['structured_components'],
                unstructured_components=parsed['unstructured_components'],
                reasoning=parsed['reasoning'] or "LLM analysis complete"
            )
        except Exception as e:
            logger.error(f"Failed to parse analysis: {e}")
            return self._fallback_analysis(original_query)

    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Fallback analysis using simple pattern matching & preprocessing."""
        # processed_query, matched_columns, needs_domain_knowledge, matched_domain_terms = self.preprocess_query(query)
        # q_lower = processed_query.lower()
        q_lower = query.lower()

        structured_score = sum(1 for pattern in self.structured_patterns if pattern in q_lower)
        unstructured_score = sum(1 for pattern in self.unstructured_patterns if pattern in q_lower)

        if structured_score and unstructured_score:
            q_type = QueryType.HYBRID
        elif structured_score:
            q_type = QueryType.STRUCTURED_ONLY
        elif unstructured_score:
            q_type = QueryType.UNSTRUCTURED_ONLY
        else:
            q_type = QueryType.UNCLEAR

        # suggested_tools = []
        # if needs_domain_knowledge:
        #     suggested_tools.append("domain_knowledge")
        # if q_type in [QueryType.STRUCTURED_ONLY, QueryType.HYBRID]:
        #     suggested_tools.append("structured_data_query")
        # if q_type in [QueryType.UNSTRUCTURED_ONLY, QueryType.HYBRID]:
        #     suggested_tools.append("rag_search")

        reasoning = "Fallback pattern + preprocessing analysis"
        return QueryAnalysis(
            query_type=q_type,
            confidence=min(1.0, (structured_score + unstructured_score) / 2),
            structured_components=["patient_data"] if structured_score else [],
            unstructured_components=["clinical_knowledge"] if unstructured_score else [],
            reasoning="Fallback pattern matching analysis"
            # reasoning = reasoning + f" | Domain mapping terms: {', '.join(matched_domain_terms)}" if matched_domain_terms else reasoning
        )

    def create_execution_plan(self, analysis: QueryAnalysis, query: str) -> List[Dict[str, Any]]:
        """Create execution plan based on analysis."""
        plan = []

        if analysis.query_type == QueryType.UNCLEAR:
            plan.append({"tool": "clarification", "action": "request_clarification", "query": query})
        if analysis.query_type in [QueryType.STRUCTURED_ONLY, QueryType.HYBRID]:
            plan.append({"tool": "structured_data_query", "action": "query_database", "query": query})
        if analysis.query_type in [QueryType.UNSTRUCTURED_ONLY, QueryType.HYBRID]:
            plan.append({"tool": "rag_search", "action": "search_documents", "query": query})

        # Add confidence and metadata
        for step in plan:
            step["confidence"] = analysis.confidence
            step["original_analysis"] = analysis
            
        print(f"*****Execution plan:\n {plan} \n*****")

        return plan

    def should_use_parallel_execution(self, plan: List[Dict[str, Any]]) -> bool:
        """Determine if tools can be executed in parallel."""
        tool_types = set(step["tool"] for step in plan)
        parallel_safe = {"structured_data_query", "rag_search"}
        return tool_types.issubset(parallel_safe) and len(plan) > 1
