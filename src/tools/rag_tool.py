"""RAG tool for querying unstructured clinical documents."""

import logging, sys
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import Field
from sentence_transformers import SentenceTransformer

from .data_loader import DataLoader
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
                limit=3
            )
            
            if not relevant_docs:
                return "No relevant clinical documents found for your query. Please try rephrasing or provide more specific terms."
            
            # 2) Build grounded context with stable source indexing
            context_parts = []
            sources = []
            for i, doc in enumerate(relevant_docs, start=1):
                # Keep mapping S1, S2, ... to sources for citations
                context_parts.append(f"[S{i}] SOURCE: {doc['source']}\nCONTENT:\n{doc['content']}")
                sources.append(doc['source'])
            context = "\n\n".join(context_parts)
            
            # Create synthesis prompt
            synthesis_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                <ROLE>
                You are a clinical AI assistant. Based only on the following clinical documents, your task is to provide a comprehensive response to the user's query.
                </ROLE>
                
                <Query>
                {query}
                </Query>

                <Clinical Documents>
                {context}
                </Clinical Documents>
                
                <INSTRUCTIONS>
                - Make only statements that are directly supported by the provided clinical documents. Donot hallucinate.
                - Do NOT use any external knowledge, prior training, or assumptions.
                - Prefer clarity and patient safety.
                - If the documents do not contain enough information to answer the query fully, explicitly state: "The documents do not contain sufficient information to answer this."
                - If something is ambiguous or conflicting, say so and cite the relevant sources.
                - Keep the answer concise and structured with short paragraphs or bullet points.
                </INSTRUCTION>
                
                <RESPONSE_5348_TAG>
                """
            )
            
            # Generate synthesized response
            prompt_text = synthesis_prompt.format(query=query, context=context)
            response = self.llm(prompt_text)
            response = response.split("</RESPONSE_5348_TAG>")[0]
            
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


