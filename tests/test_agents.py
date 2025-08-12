"""Unit tests for agent functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import pandas as pd

from src.agents.primary_agent import PrimaryAgent
from src.agents.query_router import QueryRouter, QueryType
from src.tools.data_loader import DataLoader
from src.memory.chat_memory import ChatMemoryManager


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.return_value = "Mock LLM response"
    return llm


@pytest.fixture
def sample_clinical_data():
    """Sample clinical data for testing."""
    return pd.DataFrame({
        'INSPIRE ID': ['INSP_AU010029', 'INSP_AU010030'],
        'Onset Age': [70, 65],
        'Gender': ['Male', 'Male'],
        'Stroke Type': ['R TACI', 'R PACI'],
        'Patient died before discharge': ['No', 'No'],
        'Modified Rankin Score at discharge': [2, 4]
    })


@pytest.fixture
def mock_data_loader(sample_clinical_data):
    """Mock data loader with sample data."""
    loader = Mock(spec=DataLoader)
    loader.structured_data = sample_clinical_data
    loader.load_structured_data.return_value = sample_clinical_data
    loader.get_structured_data_info.return_value = {
        'shape': sample_clinical_data.shape,
        'columns': list(sample_clinical_data.columns),
        'dtypes': sample_clinical_data.dtypes.to_dict(),
        'null_counts': sample_clinical_data.isnull().sum().to_dict()
    }
    loader.search_similar_documents.return_value = [
        {
            'content': 'Stroke is a medical emergency...',
            'source': 'clinical_guide.pdf',
            'document_type': 'pdf',
            'distance': 0.2
        }
    ]
    return loader


class TestQueryRouter:
    """Test cases for QueryRouter."""
    
    def test_init(self, mock_llm):
        """Test router initialization."""
        router = QueryRouter(mock_llm)
        assert router.llm == mock_llm
        assert len(router.structured_patterns) > 0
        assert len(router.unstructured_patterns) > 0
    
    def test_analyze_structured_query(self, mock_llm):
        """Test analysis of structured data query."""
        mock_llm.return_value = """
QUERY_TYPE: structured_only
CONFIDENCE: 0.9
STRUCTURED_COMPONENTS: patient_data, outcomes
UNSTRUCTURED_COMPONENTS: none
DOMAIN_KNOWLEDGE: none
SUGGESTED_TOOLS: structured_data_query
REASONING: Query asks for patient count data from database
"""
        
        router = QueryRouter(mock_llm)
        analysis = router.analyze_query("How many patients died from stroke?")
        
        assert analysis.query_type == QueryType.STRUCTURED_ONLY
        assert analysis.confidence == 0.9
        assert "patient_data" in analysis.structured_components
        assert "structured_data_query" in analysis.suggested_tools
    
    def test_analyze_unstructured_query(self, mock_llm):
        """Test analysis of unstructured data query."""
        mock_llm.return_value = """
QUERY_TYPE: unstructured_only
CONFIDENCE: 0.8
STRUCTURED_COMPONENTS: none
UNSTRUCTURED_COMPONENTS: clinical_knowledge
DOMAIN_KNOWLEDGE: terminology
SUGGESTED_TOOLS: rag_search
REASONING: Query asks for medical definition and explanation
"""
        
        router = QueryRouter(mock_llm)
        analysis = router.analyze_query("What is hypodense stroke?")
        
        assert analysis.query_type == QueryType.UNSTRUCTURED_ONLY
        assert analysis.confidence == 0.8
        assert "rag_search" in analysis.suggested_tools
    
    def test_analyze_hybrid_query(self, mock_llm):
        """Test analysis of hybrid query."""
        mock_llm.return_value = """
QUERY_TYPE: hybrid
CONFIDENCE: 0.85
STRUCTURED_COMPONENTS: patient_data
UNSTRUCTURED_COMPONENTS: clinical_knowledge
DOMAIN_KNOWLEDGE: terminology
SUGGESTED_TOOLS: domain_knowledge, structured_data_query, rag_search
REASONING: Query needs both patient data and clinical explanations
"""
        
        router = QueryRouter(mock_llm)
        analysis = router.analyze_query("How many patients died from hypodense stroke and what are the identification methods?")
        
        assert analysis.query_type == QueryType.HYBRID
        assert len(analysis.suggested_tools) > 1
    
    def test_fallback_analysis(self, mock_llm):
        """Test fallback analysis when LLM fails."""
        mock_llm.side_effect = Exception("LLM error")
        
        router = QueryRouter(mock_llm)
        analysis = router.analyze_query("count patients")
        
        # Should not raise exception and return valid analysis
        assert isinstance(analysis.query_type, QueryType)
        assert 0 <= analysis.confidence <= 1
    
    def test_create_execution_plan(self, mock_llm):
        """Test execution plan creation."""
        router = QueryRouter(mock_llm)
        
        # Mock analysis
        analysis = Mock()
        analysis.query_type = QueryType.STRUCTURED_ONLY
        analysis.confidence = 0.9
        analysis.structured_components = ["patient_data"]
        analysis.unstructured_components = []
        analysis.domain_knowledge_needed = []
        
        plan = router.create_execution_plan(analysis, "test query")
        
        assert len(plan) > 0
        assert plan[0]["tool"] == "structured_data_query"
        assert plan[0]["confidence"] == 0.9


class TestPrimaryAgent:
    """Test cases for PrimaryAgent."""
    
    @patch('src.agents.primary_agent.DataLoader')
    @patch('src.agents.primary_agent.ChatMemoryManager')
    @patch('src.agents.primary_agent.QueryRouter')
    def test_init(self, mock_router, mock_memory, mock_data_loader, mock_llm):
        """Test agent initialization."""
        agent = PrimaryAgent(mock_llm)
        
        assert agent.llm == mock_llm
        assert agent.session_id is not None
        assert len(agent.tools) > 0
    
    @pytest.mark.asyncio
    async def test_process_simple_query(self, mock_llm, mock_data_loader):
        """Test processing a simple query."""
        with patch('src.agents.primary_agent.DataLoader', return_value=mock_data_loader):
            agent = PrimaryAgent(mock_llm)
            
            # Mock components
            agent.query_router.analyze_query = Mock()
            agent.query_router.analyze_query.return_value = Mock(
                query_type=QueryType.STRUCTURED_ONLY,
                confidence=0.9,
                structured_components=["patient_data"],
                unstructured_components=[],
                domain_knowledge_needed=[],
                dict=lambda: {"query_type": "structured_only", "confidence": 0.9}
            )
            
            agent.query_router.create_execution_plan = Mock()
            agent.query_router.create_execution_plan.return_value = [{
                "tool": "structured_data_query",
                "action": "query_database",
                "query": "test query",
                "reason": "test"
            }]
            
            agent.query_router.should_use_parallel_execution = Mock(return_value=False)
            
            # Mock tool execution
            mock_tool = Mock()
            mock_tool._arun = AsyncMock(return_value="Mock tool result")
            mock_tool._run = Mock(return_value="Mock tool result")
            agent.tool_map = {"structured_data_query": mock_tool}
            
            result = await agent.process_query("How many patients?")
            
            assert result["query"] == "How many patients?"
            assert "answer" in result
            assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_process_query_with_error(self, mock_llm, mock_data_loader):
        """Test query processing with error handling."""
        with patch('src.agents.primary_agent.DataLoader', return_value=mock_data_loader):
            agent = PrimaryAgent(mock_llm)
            
            # Mock router to raise exception
            agent.query_router.analyze_query = Mock(side_effect=Exception("Router error"))
            
            result = await agent.process_query("test query")
            
            assert "error" in result
            assert "Router error" in result["error"]
    
    def test_get_session_info(self, mock_llm, mock_data_loader):
        """Test session info retrieval."""
        with patch('src.agents.primary_agent.DataLoader', return_value=mock_data_loader):
            agent = PrimaryAgent(mock_llm)
            
            info = agent.get_session_info()
            
            assert "session_id" in info
            assert "tools_available" in info
            assert "data_loaded" in info
    
    def test_clear_session(self, mock_llm, mock_data_loader):
        """Test session clearing."""
        with patch('src.agents.primary_agent.DataLoader', return_value=mock_data_loader):
            agent = PrimaryAgent(mock_llm)
            
            # Should not raise exception
            agent.clear_session()
            
            # Memory should be cleared
            assert len(agent.memory.conversation_history) == 0


class TestChatMemoryManager:
    """Test cases for ChatMemoryManager."""
    
    def test_init(self, mock_llm):
        """Test memory manager initialization."""
        memory = ChatMemoryManager(mock_llm, "test_session")
        
        assert memory.session_id == "test_session"
        assert memory.llm == mock_llm
        assert len(memory.conversation_history) == 0
    
    def test_add_messages(self, mock_llm):
        """Test adding messages to memory."""
        memory = ChatMemoryManager(mock_llm, "test_session")
        
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi there")
        
        # Check that messages were added
        messages = memory.memory.chat_memory.messages
        assert len(messages) >= 2
    
    def test_add_conversation_context(self, mock_llm):
        """Test adding detailed conversation context."""
        memory = ChatMemoryManager(mock_llm, "test_session")
        
        memory.add_conversation_context(
            user_query="How many patients?",
            agent_response="There are 5 patients.",
            tools_used=["structured_data_query"],
            data_accessed={"columns_used": ["INSPIRE ID"]}
        )
        
        assert len(memory.conversation_history) == 1
        context = memory.conversation_history[0]
        assert context.user_query == "How many patients?"
        assert "structured_data_query" in context.tools_used
    
    def test_get_relevant_context(self, mock_llm):
        """Test retrieving relevant context."""
        memory = ChatMemoryManager(mock_llm, "test_session")
        
        # Add some context
        memory.add_user_message("What is stroke?")
        memory.add_ai_message("Stroke is a medical emergency...")
        
        context = memory.get_relevant_context("Tell me more about stroke")
        
        assert "recent_messages" in context
        assert "suggested_columns" in context
    
    def test_clear_memory(self, mock_llm):
        """Test memory clearing."""
        memory = ChatMemoryManager(mock_llm, "test_session")
        
        # Add some data
        memory.add_user_message("Test")
        memory.add_conversation_context("query", "response", ["tool"], {})
        
        # Clear memory
        memory.clear_memory()
        
        assert len(memory.conversation_history) == 0
        assert len(memory.memory.chat_memory.messages) == 0
    
    def test_conversation_stats(self, mock_llm):
        """Test conversation statistics."""
        memory = ChatMemoryManager(mock_llm, "test_session")
        
        # Add conversation
        memory.add_conversation_context(
            user_query="test",
            agent_response="response",
            tools_used=["tool1", "tool2"],
            data_accessed={},
            confidence_score=0.8
        )
        
        stats = memory.get_conversation_stats()
        
        assert stats["total_exchanges"] == 1
        assert "tool1" in stats["tools_usage"]
        assert stats["average_confidence"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])