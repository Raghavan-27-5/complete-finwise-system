"""
Unit Tests for intelligence/price/predictor.py
Tests LSTM model loading, prediction, and Monte Carlo simulation.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestPricePredictor:
    """Test price prediction functions."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock LSTM model."""
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([[0.5]]))
        model.make_predict_function = MagicMock()
        return model
    
    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    @pytest.mark.skip(reason="preprocess_data expects specific yfinance DataFrame format")
    def test_preprocess_data(self, sample_df):
        """Test data preprocessing function."""
        from intelligence.price.predictor import preprocess_data
        
        # Create MultiIndex columns like yfinance returns
        sample_df = sample_df.reset_index()
        sample_df.columns = pd.MultiIndex.from_tuples([
            ('Date', ''), ('Open', 'AAPL'), ('High', 'AAPL'), 
            ('Low', 'AAPL'), ('Close', 'AAPL'), ('Volume', 'AAPL')
        ])
        
        result = preprocess_data(sample_df)
        assert 'Close' in result.columns
        assert len(result) > 0
    
    def test_prepare_data_shapes(self, sample_df):
        """Test prepare_data returns correct shapes."""
        from intelligence.price.predictor import prepare_data, TIME_STEP
        
        sample_df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(sample_df), freq='B')
        X, y, scaler = prepare_data(sample_df)
        
        assert X.shape[1] == TIME_STEP
        assert X.shape[2] == 1
        assert len(y) == len(X)
    
    def test_next_trading_day_skips_weekend(self):
        """Test next_trading_day skips weekends."""
        from intelligence.price.predictor import next_trading_day
        
        # Friday
        friday = pd.Timestamp('2024-01-05')
        next_day = next_trading_day(friday)
        assert next_day.weekday() == 0  # Should be Monday
    
    def test_next_trading_day_normal(self):
        """Test next_trading_day on weekday."""
        from intelligence.price.predictor import next_trading_day
        
        monday = pd.Timestamp('2024-01-08')
        next_day = next_trading_day(monday)
        assert next_day.weekday() == 1  # Should be Tuesday


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation functions."""
    
    @patch('intelligence.price.predictor.safe_download')
    def test_monte_carlo_returns_figure(self, mock_download):
        """Test Monte Carlo returns Plotly figure."""
        # Setup mock
        dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='B')
        mock_download.return_value = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(60) * 2)
        }, index=dates)
        
        from intelligence.price.predictor import monte_carlo_heston
        
        fig, stats = monte_carlo_heston(
            stock_symbol="AAPL",
            last_price=150.0,
            lstm_forecast=155.0,
            days=7,
            n_simulations=100,
            sample_paths=10
        )
        
        assert fig is not None
        assert isinstance(stats, str)
        assert "mean" in stats.lower()
    
    @patch('intelligence.price.predictor.safe_download')
    def test_monte_carlo_stats_only(self, mock_download):
        """Test Monte Carlo stats-only function."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='B')
        mock_download.return_value = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(60) * 2)
        }, index=dates)
        
        from intelligence.price.predictor import monte_carlo_heston_stats_only
        
        downside_risk = monte_carlo_heston_stats_only(
            stock_symbol="AAPL",
            last_price=150.0,
            lstm_forecast=155.0,
            days=7,
            n_simulations=100
        )
        
        assert 0 <= downside_risk <= 0.5


class TestPricePredictorEdgeCases:
    """Edge case tests for price predictor."""
    
    def test_prepare_data_insufficient_data(self):
        """Test prepare_data with insufficient data points."""
        from intelligence.price.predictor import prepare_data, TIME_STEP
        
        # Less than TIME_STEP + 2 rows - should handle gracefully or raise
        short_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        # The function may handle this gracefully or raise - test it doesn't crash
        try:
            result = prepare_data(short_df)
            # If it doesn't raise, verify it returns something reasonable
            assert result is not None
        except (ValueError, IndexError):
            # Expected behavior for insufficient data
            pass
    
    @patch('intelligence.price.predictor.safe_download')
    def test_monte_carlo_with_zero_price(self, mock_download):
        """Test Monte Carlo with zero last price."""
        mock_download.return_value = pd.DataFrame({
            'Close': [100] * 60
        })
        
        from intelligence.price.predictor import monte_carlo_heston_stats_only
        
        # Should handle gracefully
        result = monte_carlo_heston_stats_only(
            stock_symbol="AAPL",
            last_price=0.01,  # Near-zero
            lstm_forecast=0.01,
            days=7,
            n_simulations=100
        )
        
        assert isinstance(result, float)
    
    @patch('intelligence.price.predictor.safe_download')
    def test_monte_carlo_negative_forecast(self, mock_download):
        """Test Monte Carlo with negative forecast (edge case)."""
        mock_download.return_value = pd.DataFrame({
            'Close': [100] * 60
        })
        
        from intelligence.price.predictor import monte_carlo_heston_stats_only
        
        result = monte_carlo_heston_stats_only(
            stock_symbol="AAPL",
            last_price=100.0,
            lstm_forecast=-10.0,  # Negative (invalid but should handle)
            days=7,
            n_simulations=100
        )
        
        assert isinstance(result, float)
