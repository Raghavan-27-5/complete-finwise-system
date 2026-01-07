"""
Mock Gemini API
Returns static responses instead of calling real API.
"""
from typing import Dict, Any, List
from unittest.mock import MagicMock


class MockGenerationResponse:
    """Mock Gemini generation response."""
    def __init__(self, text: str):
        self.text = text
        self.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text=text)]))]


class MockGenerativeModel:
    """Mock Gemini GenerativeModel."""
    def __init__(self, model_name: str = "gemini-1.0-pro"):
        self.model_name = model_name
    
    def generate_content(self, prompt: str, **kwargs) -> MockGenerationResponse:
        # Return contextual mock responses
        if "cypher" in prompt.lower() or "query" in prompt.lower():
            return MockGenerationResponse(
                "MATCH (c:Company {name: $company}) RETURN c.name, c.revenue"
            )
        elif "sentiment" in prompt.lower():
            return MockGenerationResponse(
                "The market sentiment appears neutral with slight bullish undertones."
            )
        elif "insight" in prompt.lower():
            return MockGenerationResponse(
                "TestCorp shows strong Q4 performance with 15% revenue growth."
            )
        else:
            return MockGenerationResponse(
                "This is a mock AI response for testing purposes."
            )


def mock_configure(api_key: str = None):
    """Mock genai.configure()"""
    pass


def chatbot_with_context_mock(prompt: str, context: Any, model: Any) -> str:
    """Mock chatbot function."""
    return f"Mock response to: {prompt[:50]}..."


def chatbot_no_context_mock(prompt: str, model: Any) -> str:
    """Mock chatbot without context."""
    return f"Mock response: {prompt[:30]}..."


# Pre-configured mocks
mock_genai = MagicMock()
mock_genai.configure = mock_configure
mock_genai.GenerativeModel = MockGenerativeModel
