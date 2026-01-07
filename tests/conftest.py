"""
Pytest Configuration and Shared Fixtures
Central place for all test fixtures and configuration.
"""
import os
import sys
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mocks
from tests.mocks.mock_neo4j import MockDriver, create_mock_driver
from tests.mocks.mock_gemini import MockGenerativeModel, mock_configure
from tests.mocks.mock_yfinance import mock_download, MockTicker
from tests.mocks.mock_fetchers import mock_aggregate_sources, generate_sample_articles


# ============================================================
# Environment Fixtures
# ============================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up environment variables for testing."""
    os.environ["AURA_CONNECTION_URI"] = "neo4j+s://test.databases.neo4j.io"
    os.environ["AURA_USERNAME"] = "test_user"
    os.environ["AURA_PASSWORD"] = "test_password"
    os.environ["GEMINI_API_KEY"] = "test_gemini_key"
    os.environ["NEWS_API_KEY"] = "test_news_key"
    os.environ["PIPELINE_DEBUG"] = "0"
    yield
    

# ============================================================
# Neo4j Fixtures
# ============================================================

@pytest.fixture
def mock_neo4j_driver():
    """Provide mock Neo4j driver."""
    return MockDriver()


@pytest.fixture
def patch_neo4j():
    """Patch Neo4j GraphDatabase.driver."""
    with patch('neo4j.GraphDatabase.driver', side_effect=create_mock_driver):
        yield


# ============================================================
# Gemini Fixtures
# ============================================================

@pytest.fixture
def mock_gemini_model():
    """Provide mock Gemini model."""
    return MockGenerativeModel()


@pytest.fixture
def patch_gemini():
    """Patch google.generativeai."""
    with patch.dict('sys.modules', {'google.generativeai': MagicMock()}):
        import google.generativeai as genai
        genai.configure = mock_configure
        genai.GenerativeModel = MockGenerativeModel
        yield genai


# ============================================================
# yfinance Fixtures
# ============================================================

@pytest.fixture
def patch_yfinance():
    """Patch yfinance module."""
    with patch('yfinance.download', side_effect=mock_download):
        with patch('yfinance.Ticker', MockTicker):
            yield


@pytest.fixture
def sample_stock_data():
    """Provide sample stock DataFrame."""
    return mock_download("AAPL", period="60d")


# ============================================================
# News/Fetcher Fixtures
# ============================================================

@pytest.fixture
def sample_articles():
    """Provide sample news articles."""
    return generate_sample_articles("AAPL", count=10)


@pytest.fixture
def patch_fetchers():
    """Patch news fetchers."""
    with patch('intelligence.sentiment.fetchers.aggregate_sources', mock_aggregate_sources):
        yield


# ============================================================
# Combined Fixtures for Integration Tests
# ============================================================

@pytest.fixture
def full_mock_environment(patch_neo4j, patch_gemini, patch_yfinance, patch_fetchers):
    """Combine all mocks for full integration testing."""
    yield


# ============================================================
# Edge Case Data Fixtures
# ============================================================

@pytest.fixture
def edge_case_empty_data():
    """Empty DataFrame for edge case testing."""
    import pandas as pd
    return pd.DataFrame()


@pytest.fixture
def edge_case_null_values():
    """Data with null values."""
    import pandas as pd
    import numpy as np
    return pd.DataFrame({
        'Close': [100, np.nan, 102, None, 104],
        'Volume': [1000, 0, np.nan, 1500, 2000]
    })


@pytest.fixture
def edge_case_extreme_values():
    """Data with extreme values for stress testing."""
    import pandas as pd
    import numpy as np
    return pd.DataFrame({
        'Close': [0.001, 1e10, -100, float('inf'), 100],
        'Volume': [0, 1e15, -1, 1000000, 500]
    })


@pytest.fixture
def edge_case_single_row():
    """Single row DataFrame."""
    import pandas as pd
    return pd.DataFrame({'Close': [100], 'Volume': [1000]})
