"""Query router to determine which tools/agents to use based on the query."""

import logging
from typing import List, Dict, Any, Tuple
from enum import Enum
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from ..config.settings import settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    STRUCTURED_ONLY = "structured_only"
    UNSTRUCTURED_ONLY = "unstructured_only"
    HYBRID = "hybrid"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CLARIFICATION_NEEDED = "clarification_needed"


class QueryAnalysis(BaseModel):
    """Result of query analysis."""
    query_type: QueryType
    confidence: float
    structured_components: List[str]
    unstructured_components: List[str]
    domain_knowledge_needed: List[str]
    suggested_tools: List[str]
    reasoning: str


class QueryRouter:
    """Routes queries to appropriate tools based on content analysis."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        
        # Define query patterns
        self.structured_patterns = [
            "how many", "count", "total", "average", "sum", "statistics",
            "patient", "died", "death", "age", "gender", "score", "pressure",
            "glucose", "when", "what time", "date", "specific patient"
        ]
        
        self.unstructured_patterns = [
            "what is", "define", "explain", "how to", "treatment", "protocol",
            "guidelines", "research", "study", "evidence", "mechanism",
            "pathophysiology", "diagnosis", "symptoms", "signs"
        ]
        
        self.domain_knowledge_patterns = [
            "synonym", "meaning", "abbreviation", "term", "clinical term",
            "medical term", "what does", "column", "field", "database"
        ]
    
    def analyze_query(self, query: str, memory_context: Dict[str, Any] = None) -> QueryAnalysis:
        """Analyze query to determine routing strategy."""
        try:
            # Use LLM for sophisticated query analysis
            analysis_prompt = PromptTemplate(
                input_variables=["query", "memory_context", "structured_patterns", 
                               "unstructured_patterns", "clinical_synonyms"],
                template="""
You are a clinical AI query router. Analyze the following query and determine the best approach to answer it.

Query: {query}

Recent conversation context: {memory_context}

Available data sources:
1. Structured clinical data (CSV/Excel with patient records, measurements, outcomes)
2. Unstructured clinical documents (PDFs, research papers, guidelines)
3. Clinical domain knowledge (terminology, synonyms, mappings)

Structured data indicators: {structured_patterns}
Unstructured data indicators: {unstructured_patterns}
Clinical synonyms available: {clinical_synonyms}

Analyze the query and provide:
1. Primary query type (structured_only/unstructured_only/hybrid/domain_knowledge/clarification_needed)
2. Confidence score (0-1)
3. What structured data components are needed (if any)
4. What unstructured information is needed (if any)
5. What domain knowledge is required (if any)
6. Suggested execution order of tools
7. Brief reasoning

Format your response as:
QUERY_TYPE: [type]
CONFIDENCE: [0-1]
STRUCTURED_COMPONENTS: [list or none]
UNSTRUCTURED_COMPONENTS: [list or none]
DOMAIN_KNOWLEDGE: [list or none]
SUGGESTED_TOOLS: [ordered list]
REASONING: [explanation]
"""
            )
            
            # Format context
            memory_text = str(memory_context) if memory_context else "No prior context"
            
            prompt_text = analysis_prompt.format(
                query=query,
                memory_context=memory_text,
                structured_patterns=", ".join(self.structured_patterns),
                unstructured_patterns=", ".join(self.unstructured_patterns),
                clinical_synonyms=str(list(settings.clinical_synonyms.keys()))
            )
            
            # Get LLM analysis
            analysis_text = self.llm(prompt_text)
            
            # Parse the response
            return self._parse_analysis(analysis_text, query)
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback to simple pattern matching
            return self._fallback_analysis(query)
    
    def _parse_analysis(self, analysis_text: str, original_query: str) -> QueryAnalysis:
        """Parse LLM analysis response."""
        try:
            lines = analysis_text.strip().split('\n')
            parsed = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == "QUERY_TYPE":
                        parsed['query_type'] = QueryType(value.lower())
                    elif key == "CONFIDENCE":
                        parsed['confidence'] = min(1.0, max(0.0, float(value)))
                    elif key in ["STRUCTURED_COMPONENTS", "UNSTRUCTURED_COMPONENTS", 
                                "DOMAIN_KNOWLEDGE", "SUGGESTED_TOOLS"]:
                        if value.lower() in ["none", "[]", ""]:
                            parsed[key.lower()] = []
                        else:
                            # Parse list-like strings
                            items = [item.strip().strip('[]"\'') for item in value.split(',')]
                            parsed[key.lower()] = [item for item in items if item]
                    elif key == "REASONING":
                        parsed['reasoning'] = value
            
            return QueryAnalysis(
                query_type=parsed.get('query_type', QueryType.HYBRID),
                confidence=parsed.get('confidence', 0.5),
                structured_components=parsed.get('structured_components', []),
                unstructured_components=parsed.get('unstructured_components', []),
                domain_knowledge_needed=parsed.get('domain_knowledge', []),
                suggested_tools=parsed.get('suggested_tools', []),
                reasoning=parsed.get('reasoning', "Analysis completed")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse analysis: {e}")
            return self._fallback_analysis(original_query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Fallback analysis using simple pattern matching."""
        query_lower = query.lower()
        
        structured_score = sum(1 for pattern in self.structured_patterns 
                              if pattern in query_lower)
        unstructured_score = sum(1 for pattern in self.unstructured_patterns 
                                if pattern in query_lower)
        domain_score = sum(1 for pattern in self.domain_knowledge_patterns 
                          if pattern in query_lower)
        
        # Determine query type based on scores
        if domain_score > 0:
            query_type = QueryType.DOMAIN_KNOWLEDGE
        elif structured_score > 0 and unstructured_score > 0:
            query_type = QueryType.HYBRID
        elif structured_score > unstructured_score:
            query_type = QueryType.STRUCTURED_ONLY
        elif unstructured_score > 0:
            query_type = QueryType.UNSTRUCTURED_ONLY
        else:
            query_type = QueryType.CLARIFICATION_NEEDED
        
        # Suggest tools based on query type
        if query_type == QueryType.STRUCTURED_ONLY:
            suggested_tools = ["structured_data_query"]
        elif query_type == QueryType.UNSTRUCTURED_ONLY:
            suggested_tools = ["rag_search"]
        elif query_type == QueryType.DOMAIN_KNOWLEDGE:
            suggested_tools = ["domain_knowledge"]
        elif query_type == QueryType.HYBRID:
            suggested_tools = ["domain_knowledge", "structured_data_query", "rag_search"]
        else:
            suggested_tools = []
        
        confidence = min(1.0, (structured_score + unstructured_score + domain_score) / 3)
        
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            structured_components=["patient_data"] if structured_score > 0 else [],
            unstructured_components=["clinical_knowledge"] if unstructured_score > 0 else [],
            domain_knowledge_needed=["terminology"] if domain_score > 0 else [],
            suggested_tools=suggested_tools,
            reasoning="Fallback pattern matching analysis"
        )
    
    def create_execution_plan(self, analysis: QueryAnalysis, query: str) -> List[Dict[str, Any]]:
        """Create detailed execution plan based on analysis."""
        plan = []
        
        if analysis.query_type == QueryType.CLARIFICATION_NEEDED:
            plan.append({
                "tool": "clarification",
                "action": "request_clarification",
                "query": query,
                "reason": "Query needs clarification to determine appropriate data sources"
            })
        
        elif analysis.query_type == QueryType.DOMAIN_KNOWLEDGE:
            plan.append({
                "tool": "domain_knowledge",
                "action": "lookup_terminology",
                "query": query,
                "reason": "Need domain knowledge for terminology mapping"
            })
        
        elif analysis.query_type == QueryType.STRUCTURED_ONLY:
            plan.append({
                "tool": "structured_data_query",
                "action": "query_database",
                "query": query,
                "reason": "Query requires structured data analysis"
            })
        
        elif analysis.query_type == QueryType.UNSTRUCTURED_ONLY:
            plan.append({
                "tool": "rag_search",
                "action": "search_documents",
                "query": query,
                "reason": "Query requires unstructured document search"
            })
        
        elif analysis.query_type == QueryType.HYBRID:
            # For hybrid queries, we might need multiple steps
            if "terminology" in analysis.domain_knowledge_needed:
                plan.append({
                    "tool": "domain_knowledge",
                    "action": "lookup_terminology", 
                    "query": query,
                    "reason": "First get domain knowledge for terminology mapping"
                })
            
            if analysis.structured_components:
                plan.append({
                    "tool": "structured_data_query",
                    "action": "query_database",
                    "query": query,
                    "reason": "Query structured data for specific patient/measurement data"
                })
            
            if analysis.unstructured_components:
                plan.append({
                    "tool": "rag_search",
                    "action": "search_documents",
                    "query": query,
                    "reason": "Search clinical documents for additional context/explanations"
                })
        
        # Add confidence and metadata to each step
        for step in plan:
            step["confidence"] = analysis.confidence
            step["original_analysis"] = analysis
        
        return plan
    
    def should_use_parallel_execution(self, plan: List[Dict[str, Any]]) -> bool:
        """Determine if tools can be executed in parallel."""
        # Don't parallelize if domain knowledge is needed first
        domain_knowledge_steps = [step for step in plan if step["tool"] == "domain_knowledge"]
        
        if domain_knowledge_steps:
            return False  # Execute sequentially to use domain knowledge in subsequent steps
        
        # Can parallelize if only structured and unstructured queries
        tool_types = set(step["tool"] for step in plan)
        parallel_safe_tools = {"structured_data_query", "rag_search"}
        
        return tool_types.issubset(parallel_safe_tools) and len(plan) > 1