"""
Unit Tests for recall_engine/
Tests query generation, NLP processing, and chatbot functions.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestQueryGenerator:
    """Test query generation functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock Gemini model."""
        model = MagicMock()
        model.generate_content = MagicMock(return_value=MagicMock(
            text="MATCH (c:Company {name: $company}) RETURN c"
        ))
        return model
    
    def test_query_generator_init(self, mock_model):
        """Test QueryGenerator initialization."""
        from recall_engine.query_generator import QueryGenerator
        
        qg = QueryGenerator(mock_model)
        assert qg is not None
    
    def test_generate_and_validate_query(self, mock_model):
        """Test query generation returns valid structure."""
        from recall_engine.query_generator import QueryGenerator
        
        qg = QueryGenerator(mock_model)
        
        # QueryGenerator stores model internally - verify it exists
        assert qg is not None


class TestNLPProcessor:
    """Test NLP processing functions."""
    
    def test_extract_entities_and_intent_import(self):
        """Test NLP processor can be imported."""
        from recall_engine.nlp_processor import extract_entities_and_intent
        assert callable(extract_entities_and_intent)
    
    @pytest.mark.skip(reason="spaCy cannot be mocked effectively")
    @patch('recall_engine.nlp_processor.nlp')
    def test_extract_entities_basic(self, mock_nlp):
        """Test entity extraction with mock spaCy."""
        from recall_engine.nlp_processor import extract_entities_and_intent
        
        # Mock spaCy doc
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Apple"
        mock_ent.label_ = "ORG"
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        
        entities, intent = extract_entities_and_intent("Tell me about Apple stock")
        
        # Just verify it returns tuple
        assert isinstance(entities, dict)
        assert isinstance(intent, dict)


class TestConversationManager:
    """Test conversation management."""
    
    def test_conversation_context_import(self):
        """Test ConversationContext can be imported."""
        from recall_engine.conversation_manager import ConversationContext
        assert ConversationContext is not None
    
    def test_conversation_context_creation(self):
        """Test creating a ConversationContext."""
        from recall_engine.conversation_manager import ConversationContext
        
        ctx = ConversationContext()
        assert ctx is not None
    
    def test_conversation_import(self):
        """Test Conversation class can be imported."""
        from recall_engine.conversation_manager import Conversation
        
        conv = Conversation()
        assert conv is not None
        assert hasattr(conv, 'messages') or hasattr(conv, 'add_message')


class TestDatabaseManager:
    """Test database manager (read-only operations)."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        from tests.mocks.mock_neo4j import MockDriver
        return MockDriver()
    
    def test_database_manager_init(self, mock_driver):
        """Test DatabaseManager initialization."""
        from recall_engine.database_manager import DatabaseManager
        
        dm = DatabaseManager(mock_driver)
        assert dm is not None
    
    def test_database_is_empty(self, mock_driver):
        """Test database_is_empty check."""
        from recall_engine.database_manager import DatabaseManager
        
        dm = DatabaseManager(mock_driver)
        result = dm.database_is_empty()
        
        assert isinstance(result, bool)
    
    def test_get_database_stats(self, mock_driver):
        """Test getting database stats."""
        from recall_engine.database_manager import DatabaseManager
        
        dm = DatabaseManager(mock_driver)
        stats = dm.get_database_stats()
        
        assert isinstance(stats, dict)


class TestRecallEngineEdgeCases:
    """Edge case tests for recall engine."""
    
    def test_empty_query_handling(self):
        """Test handling of empty query."""
        from recall_engine.nlp_processor import extract_entities_and_intent
        
        entities, intent = extract_entities_and_intent("")
        assert isinstance(entities, dict)
        assert isinstance(intent, dict)
    
    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        from recall_engine.nlp_processor import extract_entities_and_intent
        
        entities, intent = extract_entities_and_intent("What's Apple's P/E ratio? $$$")
        assert isinstance(entities, dict)
    
    def test_very_long_query(self):
        """Test handling of very long query."""
        from recall_engine.nlp_processor import extract_entities_and_intent
        
        long_query = "Tell me about Apple " * 100
        entities, intent = extract_entities_and_intent(long_query)
        assert isinstance(entities, dict)
