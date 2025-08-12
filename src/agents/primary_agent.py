"""Primary agent that orchestrates the entire clinical query processing pipeline."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

from .query_router import QueryRouter, QueryType, QueryAnalysis
from ..tools.structured_query_tool import StructuredQueryTool
from ..tools.rag_tool import RAGTool, DomainKnowledgeTool
from ..tools.data_loader import DataLoader
from ..memory.chat_memory import ChatMemoryManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class PrimaryAgent:
    """Main agent that coordinates all clinical data analysis tasks."""
    
    def __init__(self, llm1: BaseLLM, llm2: BaseLLM, session_id: str = "default"):
        self.llm1 = llm1
        self.llm2 = llm2
        self.session_id = session_id
        
        # Initialize components
        self.data_loader = DataLoader()
        self.memory = ChatMemoryManager(llm1, session_id)
        self.query_router = QueryRouter(llm1)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Load data
        self._load_initial_data()
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize all available tools."""
        tools = [
            StructuredQueryTool(self.data_loader, self.llm2),
            RAGTool(self.data_loader, self.llm1),
            DomainKnowledgeTool(self.llm1)
        ]
        
        logger.info(f"Initialized {len(tools)} tools: {[tool.name for tool in tools]}")
        return tools
    
    def _load_initial_data(self) -> None:
        """Load initial data sets."""
        try:
            # Load structured data
            self.data_loader.load_structured_data()
            logger.info("Structured data loaded successfully")
            
            # Load and index unstructured data
            documents = self.data_loader.load_unstructured_data()
            if documents:
                self.data_loader.index_documents(documents)
                logger.info("Unstructured data loaded and indexed successfully")
            else:
                logger.warning("No unstructured documents found")
                
        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
            # Continue without data - will be handled gracefully by tools
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return comprehensive response."""
        start_time = datetime.now()
        
        try:
            # Add user message to memory
            self.memory.add_user_message(query)
            
            # Get memory context
            memory_context = self.memory.get_relevant_context(query)
            
            # Analyze query
            analysis = self.query_router.analyze_query(query, memory_context)
            logger.info(f"Query analyzed as: {analysis.query_type} (confidence: {analysis.confidence})")
            
            # Create execution plan
            execution_plan = self.query_router.create_execution_plan(analysis, query)
            
            # Execute plan
            if self.query_router.should_use_parallel_execution(execution_plan):
                results = await self._execute_parallel(execution_plan, query)
            else:
                results = await self._execute_sequential(execution_plan, query)
            
            # Synthesize final response
            final_response = await self._synthesize_response(query, analysis, results)
            
            # Add to memory
            self.memory.add_ai_message(final_response["answer"])
            
            # Record conversation context
            self.memory.add_conversation_context(
                user_query=query,
                agent_response=final_response["answer"],
                tools_used=[result["tool"] for result in results],
                data_accessed={"results": results},
                confidence_score=analysis.confidence
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "answer": final_response["answer"],
                "analysis": analysis.dict(),
                "execution_plan": execution_plan,
                "tool_results": results,
                "confidence": analysis.confidence,
                "processing_time": processing_time,
                "session_id": self.session_id
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            error_response = self._create_error_response(query, str(e))
            self.memory.add_ai_message(error_response)
            
            return {
                "query": query,
                "answer": error_response,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "session_id": self.session_id
            }
    
    async def _execute_sequential(self, execution_plan: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Execute tools sequentially."""
        results = []
        context_updates = {}
        
        for step in execution_plan:
            tool_name = step["tool"]
            
            if tool_name == "clarification":
                result = self._handle_clarification(query)
            else:
                # Get the tool
                tool = self.tool_map.get(tool_name)
                if not tool:
                    logger.error(f"Tool not found: {tool_name}")
                    continue
                
                # Modify query based on previous results if needed
                modified_query = self._enhance_query_with_context(query, context_updates)
                
                # Execute tool
                try:
                    tool_result = await tool._arun(modified_query)
                except AttributeError:
                    # Fallback to sync execution
                    tool_result = tool._run(modified_query)
                
                result = {
                    "tool": tool_name,
                    "query": modified_query,
                    "result": tool_result,
                    "step_info": step,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update context for next tools
                if tool_name == "domain_knowledge":
                    context_updates["domain_knowledge"] = tool_result
                
                results.append(result)
                logger.info(f"Executed tool: {tool_name}")
        
        return results
    
    async def _execute_parallel(self, execution_plan: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Execute tools in parallel where possible."""
        tasks = []
        
        for step in execution_plan:
            tool_name = step["tool"]
            tool = self.tool_map.get(tool_name)
            
            if tool:
                task = asyncio.create_task(self._execute_single_tool(tool, query, step))
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Tool execution failed: {result}")
                final_results.append({
                    "tool": execution_plan[i]["tool"],
                    "query": query,
                    "result": f"Tool execution failed: {str(result)}",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_single_tool(self, tool: BaseTool, query: str, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool and return formatted result."""
        try:
            result = await tool._arun(query)
        except AttributeError:
            result = tool._run(query)
        
        return {
            "tool": tool.name,
            "query": query,
            "result": result,
            "step_info": step_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def _enhance_query_with_context(self, original_query: str, context_updates: Dict[str, str]) -> str:
        """Enhance query with context from previous tool results."""
        if not context_updates:
            return original_query
        
        enhanced_query = original_query
        
        # Add domain knowledge context if available
        if "domain_knowledge" in context_updates:
            domain_info = context_updates["domain_knowledge"]
            # Extract column suggestions from domain knowledge
            if "Suggested database columns" in domain_info:
                enhanced_query += f"\n\nAdditional context: {domain_info}"
        
        return enhanced_query
    
    async def _synthesize_response(self, query: str, analysis: QueryAnalysis, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final response from all tool results."""
        try:
            # Prepare synthesis prompt
            synthesis_prompt = PromptTemplate(
                input_variables=["query", "analysis", "tool_results", "memory_context"],
                template="""
You are a clinical AI assistant. Synthesize a comprehensive, accurate response based on the query analysis and tool results.

Original Query: {query}

Query Analysis: {analysis}

Tool Results:
{tool_results}

Memory Context: {memory_context}

Instructions:
1. Provide a direct, comprehensive answer to the user's question
2. Integrate information from all relevant tool results
3. Ensure medical accuracy and clinical appropriateness
4. Explain any limitations or uncertainties
5. Use clear, professional language accessible to healthcare professionals
6. If data is incomplete, acknowledge this and suggest next steps
7. Maintain patient privacy and confidentiality

Response:
"""
            )
            
            # Format tool results for synthesis
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Tool: {result['tool']}\n"
                    f"Result: {result['result']}\n"
                    f"---"
                )
            
            results_text = "\n".join(formatted_results)
            memory_context = self.memory.get_memory_variables()
            
            prompt_text = synthesis_prompt.format(
                query=query,
                analysis=analysis.dict(),
                tool_results=results_text,
                memory_context=str(memory_context)
            )
            
            # Generate synthesized response
            synthesized_answer = self.llm1(prompt_text)
            
            return {
                "answer": synthesized_answer,
                "synthesis_successful": True
            }
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            
            # Fallback: combine results directly
            fallback_answer = self._create_fallback_response(query, results)
            
            return {
                "answer": fallback_answer,
                "synthesis_successful": False,
                "synthesis_error": str(e)
            }
    
    def _create_fallback_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Create fallback response when synthesis fails."""
        response_parts = [f"Based on your query: '{query}', here are the findings:\n"]
        
        for result in results:
            tool_name = result["tool"].replace("_", " ").title()
            response_parts.append(f"\n**{tool_name} Results:**")
            response_parts.append(str(result["result"])[:1000])  # Limit length
        
        return "\n".join(response_parts)
    
    def _handle_clarification(self, query: str) -> Dict[str, Any]:
        """Handle queries that need clarification."""
        clarification_response = f"""
I need some clarification to better assist you with your query: "{query}"

Could you please specify:
1. Are you looking for specific patient data from our clinical database?
2. Do you need general clinical information or research findings?
3. Are there specific medical terms or conditions you'd like me to explain?

Available data sources:
- Clinical patient database with records, demographics, outcomes
- Medical literature and research documents
- Clinical terminology and guidelines

Please rephrase your question with more specific details so I can provide the most accurate response.
"""
        
        return {
            "tool": "clarification",
            "query": query,
            "result": clarification_response,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_error_response(self, query: str, error: str) -> str:
        """Create user-friendly error response."""
        return f"""
I encountered an issue while processing your query: "{query}"

Error details: {error}

Please try:
1. Rephrasing your question with more specific terms
2. Checking if you're asking about available data in our system
3. Breaking complex questions into simpler parts

I'm here to help with clinical data analysis and medical information queries.
"""
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "tools_available": [tool.name for tool in self.tools],
            "data_loaded": {
                "structured": self.data_loader.structured_data is not None,
                "unstructured": self.data_loader.weaviate_client is not None
            },
            "memory_stats": self.memory.get_conversation_stats(),
            "data_summary": self.data_loader.get_structured_data_info() if self.data_loader.structured_data is not None else {}
        }
    
    def clear_session(self) -> None:
        """Clear session memory and reset state."""
        self.memory.clear_memory()
        logger.info(f"Session cleared: {self.session_id}")
    
    def get_available_tools_info(self) -> Dict[str, str]:
        """Get information about available tools."""
        return {tool.name: tool.description for tool in self.tools}
    
    def close(self) -> None:
        """Clean up resources."""
        if self.data_loader:
            self.data_loader.close()
        logger.info(f"Primary agent closed for session: {self.session_id}")