"""
Integration Tests
Tests full pipeline flow with all modules working together.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestFullPipelineIntegration:
    """Test complete pipeline from orchestrator to KG."""
    
    @pytest.fixture
    def mock_yf_data(self):
        """Create mock yfinance data."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B')
        return pd.DataFrame({
            'Open': 100 + np.random.randn(100) * 5,
            'High': 105 + np.random.randn(100) * 5,
            'Low': 95 + np.random.randn(100) * 5,
            'Close': 100 + np.cumsum(np.random.randn(100)),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    @patch('intelligence.price.predictor.safe_download')
    @patch('intelligence.price.predictor.model')
    @patch('intelligence.sentiment.fetchers.aggregate_sources')
    @patch('intelligence.sentiment.models.FinBERT')
    def test_price_adapter_integration(
        self, mock_finbert, mock_sources, mock_model, mock_download, mock_yf_data
    ):
        """Test price adapter integrates correctly."""
        # Setup mocks
        mock_download.return_value = mock_yf_data
        mock_model.predict = MagicMock(return_value=np.array([[0.5]] * 39))
        mock_model.make_predict_function = MagicMock()
        
        from intelligence.price.adapter import build_price_intelligence
        
        result = build_price_intelligence("AAPL", 7)
        
        assert "trend" in result
        assert "history" in result
        assert "forecast" in result
        assert "monte_carlo" in result
        assert "indicators" in result
    
    @patch('intelligence.sentiment.fetchers.aggregate_sources')
    @patch('intelligence.sentiment.pipeline.yf')
    async def test_sentiment_adapter_integration(self, mock_yf, mock_sources):
        """Test sentiment adapter integrates correctly."""
        # Setup mocks
        mock_sources.return_value = [
            {
                'headline': 'Apple Reports Strong Earnings',
                'summary': 'Apple beat estimates.',
                'source': 'Reuters',
                'published': pd.Timestamp.now().isoformat(),
                'engagement': 1000
            }
        ]
        
        mock_ticker = MagicMock()
        mock_ticker.info = {'longName': 'Apple Inc'}
        mock_yf.Ticker.return_value = mock_ticker
        
        from intelligence.sentiment.adapter import build_sentiment_intelligence
        
        result = build_sentiment_intelligence("AAPL", 7)
        
        assert "global_score" in result


class TestDataFlowIntegration:
    """Test data flows correctly between modules."""
    
    def test_config_to_sentiment_flow(self):
        """Test config values flow to sentiment module."""
        from core.config import ASPECT_KEYWORDS, BASE_SOURCE_WEIGHTS
        
        # Verify config is accessible
        assert len(ASPECT_KEYWORDS) > 0
        assert len(BASE_SOURCE_WEIGHTS) > 0
    
    def test_orchestrator_to_kg_adapter_flow(self):
        """Test orchestrator output format matches KG adapter input."""
        # Create sample orchestrator output
        raw_state = {
            "symbol": "AAPL",
            "as_of": "2024-01-01T00:00:00Z",
            "price": {"trend": "Bullish"},
            "sentiment": {"global_score": 0.5},
            "history": {},
            "forecast": {},
            "monte_carlo": {},
            "indicators": {},
            "articles": []
        }
        
        from kg.state_adapter import adapt_state
        
        adapted = adapt_state(raw_state)
        
        # Verify adapted state is valid
        assert adapted is not None


class TestStressTestIntegration:
    """Stress tests for the integration."""
    
    def test_large_article_batch(self):
        """Test processing large batch of articles."""
        from intelligence.sentiment.pipeline import calculate_institutional_dsp
        
        # Create large DataFrame
        n = 1000
        df = pd.DataFrame({
            'sentiment_score': np.random.randn(n) * 0.5,
            'weight': np.random.uniform(0.1, 2.0, n),
            'age_days': np.random.uniform(0, 30, n)
        })
        
        dsp = calculate_institutional_dsp(df)
        
        assert isinstance(dsp, float)
        assert -3 <= dsp <= 3
    
    def test_concurrent_symbol_processing(self):
        """Test system can handle multiple symbols conceptually."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        from intelligence.sentiment.pipeline import generate_synonyms_for_company
        
        for symbol in symbols:
            syns = generate_synonyms_for_company(f"{symbol} Inc", symbol)
            assert len(syns) > 0
    
    def test_memory_efficiency_large_history(self):
        """Test memory handling with large price history."""
        from intelligence.sentiment.pipeline import compute_age_days
        
        # Create large DataFrame
        n = 10000
        df = pd.DataFrame({
            'published': pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=n, freq='H'),
            'sentiment_score': np.random.randn(n)
        })
        
        result = compute_age_days(df, 'published')
        
        assert len(result) == n
        assert 'age_days' in result.columns
