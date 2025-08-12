"""RAG tool for querying unstructured clinical documents."""

import logging
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import Field
from sentence_transformers import SentenceTransformer

from .data_loader import DataLoader
from ..config.settings import settings

logger = logging.getLogger(__name__)


class RAGTool(BaseTool):
    """Tool for retrieving information from unstructured clinical documents."""
    
    name: str = "rag_search"
    description: str = """
    Search through clinical documents, research papers, and medical texts to find relevant information.
    Use this tool when you need:
    - Clinical domain knowledge and definitions
    - Medical terminology explanations  
    - Treatment protocols and guidelines
    - Research findings and best practices
    - Context about medical conditions
    
    Input should be a clear search query about medical/clinical topics.
    """
    
    data_loader: DataLoader = Field(exclude=True)
    llm: BaseLLM = Field(exclude=True)
    embedding_model: SentenceTransformer = Field(exclude=True)
    
    def __init__(self, data_loader: DataLoader, llm: BaseLLM, embedding_model: SentenceTransformer):
        super().__init__(data_loader=data_loader, llm=llm, embedding_model=embedding_model)
    
    def _run(self, query: str) -> str:
        """Execute RAG search and return synthesized answer."""
        try:
            # Search for relevant documents
            relevant_docs = self.data_loader.search_similar_documents(
                query=query,
                limit=5
            )
            
            if not relevant_docs:
                return "No relevant clinical documents found for your query. Please try rephrasing or provide more specific terms."
            
            # Merge chunks from same source and avoid duplicates
            context_parts = []
            seen_sources = set()
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"[Source {i+1}]: {doc['content']}")
                if doc['source'] not in seen_sources:
                    sources.append(doc['source'])
                    seen_sources.add(doc['source'])
            
            context = "\n\n".join(context_parts)
            
            # Create synthesis prompt
            synthesis_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                <INSTRUCTION>
                You are a clinical AI assistant. Based on the following clinical documents, provide a comprehensive response to the user's query.

                Query: {query}

                Clinical Documents:
                {context}

                Instructions:
                1. Provide accurate, evidence-based information
                2. Explain medical terms clearly
                3. Include relevant clinical context
                4. If information is incomplete, acknowledge limitations
                5. Prioritize patient safety considerations
                6. Use professional medical language while remaining accessible
                </INSTRUCTION>
                
                <RESPONSE>
                """
            )
            
            # Generate synthesized response
            prompt_text = synthesis_prompt.format(query=query, context=context)
            response = self.llm(prompt_text)
            response = response.split("</RESPONSE>")[0]
            
            # Append clean list of sources
            source_info = "\n\nSources consulted:\n" + "\n".join([
                f"- {src}" for src in sources
            ])
            
            logger.info(f"RAG query processed: {query[:50]}...")
            return response.strip() + source_info
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return f"I encountered an error while searching clinical documents: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)


# class DomainKnowledgeTool(BaseTool):
#     """Tool for accessing clinical domain knowledge and terminology mapping."""
    
#     name: str = "domain_knowledge"
#     description: str = """
#     Access clinical domain knowledge for terminology mapping and concept understanding.
#     Use this tool to:
#     - Map clinical terms to database column names
#     - Find synonyms and related medical concepts
#     - Understand clinical abbreviations and terminology
#     - Get context about medical measurements and scales
    
#     Input should be a medical term, abbreviation, or concept you need clarification on.
#     """
    
#     llm: BaseLLM = Field(exclude=True)
    
#     def __init__(self, llm: BaseLLM):
#         super().__init__(llm=llm)
    
#     def _run(self, query: str) -> str:
#         """Provide domain knowledge and terminology mapping."""
#         try:
#             # Check if query matches known clinical synonyms
#             query_lower = query.lower()
#             matched_concepts = []
            
#             for concept, synonyms in settings.clinical_synonyms.items():
#                 if any(synonym.lower() in query_lower for synonym in synonyms):
#                     matched_concepts.append((concept, synonyms))
            
#             # Check column mappings
#             relevant_columns = []
#             for category, columns in settings.column_mappings.items():
#                 for column in columns:
#                     if any(word in column.lower() for word in query_lower.split()):
#                         relevant_columns.append((category, column))
            
#             # Create knowledge prompt
#             knowledge_prompt = PromptTemplate(
#                 input_variables=["query", "matched_concepts", "relevant_columns"],
#                 template="""
# You are a clinical informatics expert. Provide comprehensive information about the following clinical query.

# Query: {query}

# Matched Clinical Concepts:
# {matched_concepts}

# Relevant Database Columns:
# {relevant_columns}

# Provide information about:
# 1. Clinical definition and context
# 2. Common synonyms and alternative terms
# 3. Relevant database fields that might contain this information
# 4. Clinical significance and typical values/ranges
# 5. Related concepts and measurements

# Be precise and clinically accurate in your response.

# Response:
# """
#             )
            
#             # Format matched concepts and columns
#             concepts_text = "\n".join([
#                 f"- {concept}: {', '.join(synonyms)}" 
#                 for concept, synonyms in matched_concepts
#             ]) if matched_concepts else "None found in predefined mappings"
            
#             columns_text = "\n".join([
#                 f"- {category}: {column}" 
#                 for category, column in relevant_columns
#             ]) if relevant_columns else "None found in database schema"
            
#             prompt_text = knowledge_prompt.format(
#                 query=query,
#                 matched_concepts=concepts_text,
#                 relevant_columns=columns_text
#             )
            
#             response = self.llm(prompt_text)
            
#             # Add specific column suggestions if found
#             if relevant_columns:
#                 column_suggestions = "\n\nSuggested database columns to query:\n" + "\n".join([
#                     f"- {column}" for _, column in relevant_columns
#                 ])
#                 response += column_suggestions
            
#             logger.info(f"Domain knowledge query processed: {query[:50]}...")
#             return response
            
#         except Exception as e:
#             logger.error(f"Domain knowledge lookup failed: {e}")
#             return f"I encountered an error while looking up domain knowledge: {str(e)}"
    
#     async def _arun(self, query: str) -> str:
#         """Async version of the tool."""
#         return self._run(query)