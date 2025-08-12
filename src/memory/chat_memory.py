"""Chat memory management for maintaining conversation context."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.llms.base import BaseLLM
from pydantic import BaseModel

from ..config.settings import settings

logger = logging.getLogger(__name__)


class ConversationContext(BaseModel):
    """Structure for storing conversation context."""
    session_id: str
    timestamp: datetime
    user_query: str
    agent_response: str
    tools_used: List[str]
    data_accessed: Dict[str, Any]
    confidence_score: Optional[float] = None


class ChatMemoryManager:
    """Manages chat memory and conversation context."""
    
    def __init__(self, llm: BaseLLM, session_id: str = "default"):
        self.session_id = session_id
        self.llm = llm
        
        # Initialize conversation memory
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=settings.max_tokens // 2,  # Reserve half tokens for memory
            buffer_size=settings.memory_buffer_size,
            return_messages=True
        )
        
        # Store detailed conversation context
        self.conversation_history: List[ConversationContext] = []
        
        # Track domain-specific context
        self.clinical_context = {
            "recent_patients": [],
            "recent_conditions": [],
            "recent_queries": [],
            "column_mappings_used": {},
            "domain_knowledge_accessed": []
        }
    
    def add_user_message(self, message: str) -> None:
        """Add user message to memory."""
        self.memory.chat_memory.add_user_message(message)
        logger.debug(f"Added user message: {message[:100]}...")
    
    def add_ai_message(self, message: str) -> None:
        """Add AI message to memory."""
        self.memory.chat_memory.add_ai_message(message)
        logger.debug(f"Added AI message: {message[:100]}...")
    
    def add_conversation_context(
        self,
        user_query: str,
        agent_response: str,
        tools_used: List[str],
        data_accessed: Dict[str, Any],
        confidence_score: Optional[float] = None
    ) -> None:
        """Add detailed conversation context."""
        context = ConversationContext(
            session_id=self.session_id,
            timestamp=datetime.now(),
            user_query=user_query,
            agent_response=agent_response,
            tools_used=tools_used,
            data_accessed=data_accessed,
            confidence_score=confidence_score
        )
        
        self.conversation_history.append(context)
        
        # Update clinical context
        self._update_clinical_context(context)
        
        # Keep only recent conversations
        if len(self.conversation_history) > settings.memory_buffer_size * 2:
            self.conversation_history = self.conversation_history[-settings.memory_buffer_size:]
    
    def _update_clinical_context(self, context: ConversationContext) -> None:
        """Update clinical domain context based on conversation."""
        query_lower = context.user_query.lower()
        
        # Extract patient IDs mentioned
        if "patient" in query_lower:
            words = context.user_query.split()
            for i, word in enumerate(words):
                if word.lower() == "patient" and i + 1 < len(words):
                    patient_id = words[i + 1]
                    if patient_id not in self.clinical_context["recent_patients"]:
                        self.clinical_context["recent_patients"].append(patient_id)
        
        # Track medical conditions mentioned
        clinical_terms = [
            "stroke", "hemorrhage", "hypodense", "infarct", "taci", "paci",
            "blood pressure", "glucose", "nihss", "rankin"
        ]
        
        for term in clinical_terms:
            if term in query_lower:
                if term not in self.clinical_context["recent_conditions"]:
                    self.clinical_context["recent_conditions"].append(term)
        
        # Track column mappings used
        if "data_accessed" in context.data_accessed:
            columns_used = context.data_accessed.get("columns_used", [])
            for col in columns_used:
                self.clinical_context["column_mappings_used"][col] = (
                    self.clinical_context["column_mappings_used"].get(col, 0) + 1
                )
        
        # Keep recent lists manageable
        for key in ["recent_patients", "recent_conditions"]:
            if len(self.clinical_context[key]) > 10:
                self.clinical_context[key] = self.clinical_context[key][-10:]
    
    def get_relevant_context(self, current_query: str) -> Dict[str, Any]:
        """Get relevant context for the current query."""
        context = {
            "conversation_summary": self._get_conversation_summary(),
            "recent_messages": [],
            "relevant_patients": [],
            "relevant_conditions": [],
            "suggested_columns": []
        }
        
        # Get recent messages
        messages = self.memory.chat_memory.messages[-4:]  # Last 2 exchanges
        for msg in messages:
            context["recent_messages"].append({
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content
            })
        
        # Find relevant patients
        query_lower = current_query.lower()
        for patient in self.clinical_context["recent_patients"]:
            if patient.lower() in query_lower:
                context["relevant_patients"].append(patient)
        
        # Find relevant conditions
        for condition in self.clinical_context["recent_conditions"]:
            if condition in query_lower:
                context["relevant_conditions"].append(condition)
        
        # Suggest relevant columns based on query
        context["suggested_columns"] = self._suggest_columns(current_query)
        
        return context
    
    def _get_conversation_summary(self) -> str:
        """Get conversation summary, handling different LangChain versions."""
        try:
            # Try the newer method first
            if hasattr(self.memory, 'buffer'):
                return str(self.memory.buffer)
            # Try the older method
            elif hasattr(self.memory, 'buffer_as_str'):
                return self.memory.buffer_as_str
            # Fallback: create summary from recent messages
            else:
                recent_messages = self.memory.chat_memory.messages[-6:]  # Last 3 exchanges
                if not recent_messages:
                    return "No conversation history available."
                
                summary_parts = []
                for msg in recent_messages:
                    msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    summary_parts.append(f"{msg_type}: {msg.content[:200]}...")
                
                return "\n".join(summary_parts)
        except Exception as e:
            logger.warning(f"Could not get conversation summary: {e}")
            return "Conversation summary unavailable."
    
    def _suggest_columns(self, query: str) -> List[str]:
        """Suggest relevant columns based on query content."""
        query_lower = query.lower()
        suggested = []
        
        # Use synonym mapping from settings
        for concept, synonyms in settings.clinical_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    # Find corresponding columns
                    for category, columns in settings.column_mappings.items():
                        if any(synonym.lower() in col.lower() for col in columns):
                            suggested.extend(columns)
        
        # Add frequently used columns for similar queries
        for col, count in self.clinical_context["column_mappings_used"].items():
            if count > 2 and any(word in col.lower() for word in query_lower.split()):
                if col not in suggested:
                    suggested.append(col)
        
        return list(set(suggested))[:5]  # Limit to 5 suggestions
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for prompt formatting."""
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            logger.warning(f"Could not load memory variables: {e}")
            return {"history": "Memory variables unavailable"}
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        self.conversation_history.clear()
        self.clinical_context = {
            "recent_patients": [],
            "recent_conditions": [],
            "recent_queries": [],
            "column_mappings_used": {},
            "domain_knowledge_accessed": []
        }
        logger.info(f"Cleared memory for session: {self.session_id}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        if not self.conversation_history:
            return {"total_exchanges": 0}
        
        total_exchanges = len(self.conversation_history)
        tools_usage = {}
        avg_confidence = 0
        
        for context in self.conversation_history:
            for tool in context.tools_used:
                tools_usage[tool] = tools_usage.get(tool, 0) + 1
            
            if context.confidence_score:
                avg_confidence += context.confidence_score
        
        if total_exchanges > 0 and avg_confidence > 0:
            avg_confidence /= total_exchanges
        
        return {
            "total_exchanges": total_exchanges,
            "session_duration": (
                (self.conversation_history[-1].timestamp - 
                 self.conversation_history[0].timestamp).total_seconds() / 60
                if total_exchanges > 1 else 0
            ),
            "tools_usage": tools_usage,
            "average_confidence": avg_confidence,
            "unique_patients_discussed": len(self.clinical_context["recent_patients"]),
            "conditions_discussed": len(self.clinical_context["recent_conditions"])
        }