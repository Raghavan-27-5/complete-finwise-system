"""
Mock Neo4j Driver
Returns in-memory data instead of connecting to real database.
"""
from typing import Dict, List, Any
from unittest.mock import MagicMock


class MockRecord:
    """Mock Neo4j record."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def data(self) -> Dict[str, Any]:
        return self._data
    
    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)


class MockResult:
    """Mock Neo4j query result."""
    def __init__(self, records: List[Dict[str, Any]] = None):
        self._records = [MockRecord(r) for r in (records or [])]
    
    def single(self) -> MockRecord:
        return self._records[0] if self._records else None
    
    def data(self) -> List[Dict[str, Any]]:
        return [r.data() for r in self._records]
    
    def __iter__(self):
        return iter(self._records)


class MockSession:
    """Mock Neo4j session."""
    def __init__(self):
        self._data = {}
    
    def run(self, query: str, parameters: Dict = None) -> MockResult:
        # Return sample data for common queries
        if "MATCH" in query.upper():
            return MockResult([
                {"company": "TestCorp", "metric": "Revenue", "value": 1000000},
                {"company": "TestCorp", "metric": "EPS", "value": 5.25},
            ])
        return MockResult([])
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class MockDriver:
    """Mock Neo4j driver."""
    def __init__(self, *args, **kwargs):
        pass
    
    def session(self, **kwargs) -> MockSession:
        return MockSession()
    
    def close(self):
        pass
    
    def verify_connectivity(self):
        return True


def create_mock_driver(*args, **kwargs) -> MockDriver:
    """Factory function to create mock driver."""
    return MockDriver()


# Pre-configured mock for patching
mock_neo4j_driver = MagicMock(side_effect=create_mock_driver)
