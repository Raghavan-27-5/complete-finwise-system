"""
Unit Tests for core/orchestrator.py
Tests state computation and integration of P1/P2 engines.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestOrchestrator:
    """Test orchestrator state computation."""
    
    @pytest.fixture
    def mock_p1_response(self):
        """Mock price intelligence response."""
        return {
            "trend": "Bullish",
            "forecast_slope": 0.05,
            "downside_risk": 0.12,
            "history": {
                "dates": ["2024-01-01", "2024-01-02"],
                "actual": [100, 102],
                "predicted": [99, 101]
            },
            "forecast": {
                "dates": ["2024-01-03", "2024-01-04"],
                "predicted": [103, 105]
            },
            "monte_carlo": {
                "downside_var": 0.08,
                "figure": MagicMock()
            },
            "indicators": {
                "rsi": 55,
                "macd": 1.2,
                "momentum": "Bullish"
            }
        }
    
    @pytest.fixture
    def mock_p2_response(self):
        """Mock sentiment intelligence response."""
        return {
            "global_score": 0.45,
            "label": "Positive",
            "aspects": {
                "earnings": {"score": 0.6},
                "management": {"score": 0.3}
            },
            "impact_articles": [
                {"headline": "Test Article", "impact": 0.5}
            ]
        }
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    def test_compute_state_returns_valid_structure(
        self, mock_p2, mock_p1, mock_p1_response, mock_p2_response
    ):
        """Test compute_state returns expected structure."""
        mock_p1.return_value = mock_p1_response
        mock_p2.return_value = mock_p2_response
        
        from core.orchestrator import compute_state
        
        mock_kg_writer = MagicMock()
        
        state = compute_state("AAPL", 7, mock_kg_writer)
        
        # Verify structure
        assert "symbol" in state
        assert "price" in state
        assert "sentiment" in state
        assert "history" in state
        assert "forecast" in state
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    def test_compute_state_calls_kg_writer(
        self, mock_p2, mock_p1, mock_p1_response, mock_p2_response
    ):
        """Test compute_state calls KG writer."""
        mock_p1.return_value = mock_p1_response
        mock_p2.return_value = mock_p2_response
        
        from core.orchestrator import compute_state
        
        mock_kg_writer = MagicMock()
        
        compute_state("AAPL", 7, mock_kg_writer)
        
        mock_kg_writer.write_snapshot.assert_called_once()
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    def test_compute_state_has_timestamp(
        self, mock_p2, mock_p1, mock_p1_response, mock_p2_response
    ):
        """Test state includes timestamp."""
        mock_p1.return_value = mock_p1_response
        mock_p2.return_value = mock_p2_response
        
        from core.orchestrator import compute_state
        
        mock_kg_writer = MagicMock()
        
        state = compute_state("AAPL", 7, mock_kg_writer)
        
        assert "as_of" in state


class TestOrchestratorEdgeCases:
    """Edge case tests for orchestrator."""
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    def test_compute_state_empty_symbol(self, mock_p2, mock_p1):
        """Test with empty symbol."""
        mock_p1.return_value = {"trend": "Unknown", "history": {}, "forecast": {}, 
                                "monte_carlo": {}, "indicators": {}}
        mock_p2.return_value = {"global_score": 0, "aspects": {}}
        
        from core.orchestrator import compute_state
        
        mock_kg_writer = MagicMock()
        
        # Should handle gracefully
        state = compute_state("", 7, mock_kg_writer)
        assert "symbol" in state
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    def test_compute_state_zero_days(self, mock_p2, mock_p1):
        """Test with zero days horizon."""
        mock_p1.return_value = {"trend": "Neutral", "history": {}, "forecast": {},
                                "monte_carlo": {}, "indicators": {}}
        mock_p2.return_value = {"global_score": 0, "aspects": {}}
        
        from core.orchestrator import compute_state
        
        mock_kg_writer = MagicMock()
        
        state = compute_state("AAPL", 0, mock_kg_writer)
        assert state is not None
    
    @patch('core.orchestrator.build_price_intelligence')
    @patch('core.orchestrator.build_sentiment_intelligence')
    def test_compute_state_large_horizon(self, mock_p2, mock_p1):
        """Test with very large horizon."""
        mock_p1.return_value = {"trend": "Bullish", "history": {}, "forecast": {},
                                "monte_carlo": {}, "indicators": {}}
        mock_p2.return_value = {"global_score": 0.5, "aspects": {}}
        
        from core.orchestrator import compute_state
        
        mock_kg_writer = MagicMock()
        
        state = compute_state("AAPL", 365, mock_kg_writer)
        assert state is not None
