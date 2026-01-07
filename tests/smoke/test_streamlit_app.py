"""
Smoke Tests for Streamlit App
Tests that the app can be imported and initialized without errors.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestStreamlitAppSmoke:
    """Smoke tests for Streamlit application."""
    
    @patch('neo4j.GraphDatabase.driver')
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_streamlit_app_imports(self, mock_model, mock_configure, mock_driver):
        """Test Streamlit app can be imported."""
        # Setup mocks
        mock_driver.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # This should not raise ImportError
        try:
            import app.streamlit_app
            imported = True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
        
        assert imported
    
    @patch('neo4j.GraphDatabase.driver')
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_streamlit_app_has_main_class(self, mock_model, mock_configure, mock_driver):
        """Test Streamlit app has FinWiseApp class."""
        mock_driver.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        from app.streamlit_app import FinWiseApp
        
        assert FinWiseApp is not None
    
    @patch('neo4j.GraphDatabase.driver')
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('streamlit.set_page_config')
    @patch('streamlit.markdown')
    @patch('streamlit.sidebar')
    def test_finwise_app_initialization(
        self, mock_sidebar, mock_markdown, mock_page_config,
        mock_model, mock_configure, mock_driver
    ):
        """Test FinWiseApp can be initialized."""
        mock_driver.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        from app.streamlit_app import FinWiseApp
        
        # Initialize should not raise
        app = FinWiseApp()
        
        assert app is not None


class TestStreamlitDependencies:
    """Test all Streamlit dependencies can be imported."""
    
    def test_streamlit_import(self):
        """Test streamlit can be imported."""
        import streamlit
        assert streamlit is not None
    
    def test_neo4j_import(self):
        """Test neo4j can be imported."""
        import neo4j
        assert neo4j is not None
    
    def test_genai_import(self):
        """Test google.generativeai can be imported."""
        import google.generativeai
        assert google.generativeai is not None
    
    def test_recall_engine_imports(self):
        """Test recall engine modules can be imported."""
        from recall_engine.database_manager import DatabaseManager
        from recall_engine.conversation_manager import Conversation, ConversationContext
        from recall_engine.nlp_processor import extract_entities_and_intent
        from recall_engine.query_generator import QueryGenerator
        from recall_engine.chatbot import chatbot_no_context
        
        assert all([
            DatabaseManager,
            Conversation,
            ConversationContext,
            extract_entities_and_intent,
            QueryGenerator,
            chatbot_no_context
        ])
