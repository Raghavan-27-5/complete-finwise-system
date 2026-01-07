"""
Smoke Tests for Gradio App
Tests that the app can be imported and demo created without errors.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestGradioAppSmoke:
    """Smoke tests for Gradio application."""
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    @patch('neo4j.GraphDatabase.driver')
    def test_gradio_app_imports(self, mock_driver, mock_p2, mock_p1):
        """Test Gradio app can be imported."""
        mock_driver.return_value = MagicMock()
        mock_p1.return_value = {}
        mock_p2.return_value = {}
        
        try:
            import app.gradio_app
            imported = True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
        
        assert imported
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    @patch('neo4j.GraphDatabase.driver')
    def test_gradio_app_has_run_function(self, mock_driver, mock_p2, mock_p1):
        """Test Gradio app has run_finwise function."""
        mock_driver.return_value = MagicMock()
        mock_p1.return_value = {}
        mock_p2.return_value = {}
        
        from app.gradio_app import run_finwise
        
        assert callable(run_finwise)
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    @patch('neo4j.GraphDatabase.driver')
    def test_gradio_demo_creation(self, mock_driver, mock_p2, mock_p1):
        """Test Gradio demo can be created."""
        mock_driver.return_value = MagicMock()
        mock_p1.return_value = {}
        mock_p2.return_value = {}
        
        import app.gradio_app as gradio_app
        
        # Check if demo exists (might be created at module level)
        assert hasattr(gradio_app, 'demo') or hasattr(gradio_app, 'run_finwise')


class TestGradioDependencies:
    """Test all Gradio dependencies can be imported."""
    
    def test_gradio_import(self):
        """Test gradio can be imported."""
        import gradio
        assert gradio is not None
    
    def test_plotly_import(self):
        """Test plotly can be imported."""
        import plotly.graph_objects
        assert plotly is not None
    
    def test_pandas_import(self):
        """Test pandas can be imported."""
        import pandas
        assert pandas is not None
    
    def test_core_orchestrator_import(self):
        """Test core.orchestrator can be imported."""
        from core.orchestrator import compute_state
        assert callable(compute_state)
    
    def test_intelligence_adapters_import(self):
        """Test intelligence adapters can be imported."""
        from intelligence.price.adapter import build_price_intelligence
        from intelligence.sentiment.adapter import build_sentiment_intelligence
        
        assert callable(build_price_intelligence)
        assert callable(build_sentiment_intelligence)


class TestGradioEdgeCases:
    """Edge case tests for Gradio app."""
    
    @patch('core.orchestrator.compute_state')
    @patch('neo4j.GraphDatabase.driver')
    def test_run_finwise_empty_symbol(self, mock_driver, mock_compute):
        """Test run_finwise with empty symbol."""
        mock_driver.return_value = MagicMock()
        mock_compute.return_value = {"symbol": "", "price": {}, "sentiment": {}}
        
        from app.gradio_app import run_finwise
        
        # Should handle gracefully
        result = run_finwise("", 7)
        assert result is not None
    
    @patch('core.orchestrator.compute_state')
    @patch('neo4j.GraphDatabase.driver')
    def test_run_finwise_special_symbols(self, mock_driver, mock_compute):
        """Test run_finwise with special stock symbols."""
        mock_driver.return_value = MagicMock()
        mock_compute.return_value = {"symbol": "BRK.A", "price": {}, "sentiment": {}}
        
        from app.gradio_app import run_finwise
        
        # Should handle symbols with dots/special chars
        result = run_finwise("BRK.A", 7)
        assert result is not None
