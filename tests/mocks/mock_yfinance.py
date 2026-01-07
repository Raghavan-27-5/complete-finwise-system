"""
Mock yfinance
Returns synthetic OHLCV data instead of calling Yahoo Finance.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import MagicMock


def generate_synthetic_ohlcv(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate realistic synthetic stock data."""
    np.random.seed(hash(symbol) % 2**32)
    
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='B')
    
    # Start price based on symbol hash
    base_price = 100 + (hash(symbol) % 400)
    
    # Generate random walk with drift
    returns = np.random.normal(0.0005, 0.02, days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = {
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': prices * (1 + np.random.uniform(0.005, 0.025, days)),
        'Low': prices * (1 - np.random.uniform(0.005, 0.025, days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, days)
    }
    
    df = pd.DataFrame(data, index=dates)
    df['Adj Close'] = df['Close']
    return df


class MockTicker:
    """Mock yfinance Ticker."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._info = {
            'longName': f'{symbol} Test Corporation',
            'symbol': symbol,
            'sector': 'Technology',
            'industry': 'Software',
            'currentPrice': 150.00,
            'marketCap': 1000000000,
        }
    
    @property
    def info(self) -> Dict[str, Any]:
        return self._info
    
    def history(self, period: str = "1mo", **kwargs) -> pd.DataFrame:
        days = {'1d': 1, '5d': 5, '1mo': 21, '3mo': 63, '6mo': 126, '1y': 252}.get(period, 60)
        return generate_synthetic_ohlcv(self.symbol, days)


def mock_download(
    tickers: str,
    start=None,
    end=None,
    period=None,
    auto_adjust: bool = True,
    threads: bool = False,
    progress: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Mock yf.download()"""
    symbol = tickers if isinstance(tickers, str) else tickers[0]
    
    if period:
        days = {'1d': 1, '5d': 5, '1mo': 21, '3mo': 63, '6mo': 126, '1y': 252, '60d': 60}.get(period, 60)
    elif start and end:
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    else:
        days = 60
    
    return generate_synthetic_ohlcv(symbol, max(days, 60))


# Pre-configured mocks
mock_yf = MagicMock()
mock_yf.download = mock_download
mock_yf.Ticker = MockTicker
