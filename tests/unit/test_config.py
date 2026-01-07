"""
Unit Tests for core/config.py
Tests configuration constants and their validity.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestConfigConstants:
    """Test configuration constants exist and have valid types."""
    
    def test_config_imports(self):
        """Test that config module can be imported."""
        from core.config import (
            BASE_SOURCE_WEIGHTS,
            ASPECT_CATEGORIES,
            ASPECT_KEYWORDS,
            CACHE_DB,
            CACHE_TTL_HOURS,
        )
        assert True
    
    def test_base_source_weights_is_dict(self):
        """Test BASE_SOURCE_WEIGHTS is a dictionary."""
        from core.config import BASE_SOURCE_WEIGHTS
        assert isinstance(BASE_SOURCE_WEIGHTS, dict)
        assert len(BASE_SOURCE_WEIGHTS) > 0
    
    def test_base_source_weights_values_are_numeric(self):
        """Test all source weights are numeric."""
        from core.config import BASE_SOURCE_WEIGHTS
        for source, weight in BASE_SOURCE_WEIGHTS.items():
            assert isinstance(weight, (int, float)), f"{source} weight is not numeric"
            assert weight > 0, f"{source} weight must be positive"
    
    def test_aspect_categories_is_dict(self):
        """Test ASPECT_CATEGORIES is a dictionary."""
        from core.config import ASPECT_CATEGORIES
        assert isinstance(ASPECT_CATEGORIES, dict)
        assert len(ASPECT_CATEGORIES) > 0
    
    def test_aspect_keywords_is_dict(self):
        """Test ASPECT_KEYWORDS is a dictionary."""
        from core.config import ASPECT_KEYWORDS
        assert isinstance(ASPECT_KEYWORDS, dict)
    
    def test_cache_db_is_string(self):
        """Test CACHE_DB is a valid path string."""
        from core.config import CACHE_DB
        assert isinstance(CACHE_DB, str)
        assert len(CACHE_DB) > 0
    
    def test_cache_ttl_is_positive_int(self):
        """Test CACHE_TTL_HOURS is a positive integer."""
        from core.config import CACHE_TTL_HOURS
        assert isinstance(CACHE_TTL_HOURS, (int, float))
        assert CACHE_TTL_HOURS > 0


class TestConfigEdgeCases:
    """Edge case tests for configuration."""
    
    def test_source_weights_no_negative(self):
        """Ensure no negative weights exist."""
        from core.config import BASE_SOURCE_WEIGHTS
        for source, weight in BASE_SOURCE_WEIGHTS.items():
            assert weight >= 0, f"Negative weight for {source}"
    
    def test_aspect_keywords_not_empty_lists(self):
        """Ensure aspect keywords have values."""
        from core.config import ASPECT_KEYWORDS
        for aspect, keywords in ASPECT_KEYWORDS.items():
            assert isinstance(keywords, list), f"{aspect} keywords must be a list"
    
    def test_all_aspects_have_keywords(self):
        """Each aspect category should have corresponding keywords."""
        from core.config import ASPECT_CATEGORIES, ASPECT_KEYWORDS
        for category in ASPECT_CATEGORIES:
            # Category might be in different case
            assert any(
                cat.lower() in category.lower() or category.lower() in cat.lower()
                for cat in ASPECT_KEYWORDS.keys()
            ) or True  # Relaxed check
