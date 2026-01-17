"""
LLM Provider Abstraction Layer for FinWise Core.

Supports multiple LLM backends with automatic fallback:
1. OpenRouter (Recommended) - Access to DeepSeek, Llama, and 300+ models
2. Gemini (Legacy) - Google's deprecated generativeai package

Usage:
    from core.llm_provider import get_llm_client
    
    client = get_llm_client()
    response = client.generate("Your prompt here")
"""

import os
import logging
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_content(self, prompt: str) -> object:
        """Generate content (for backwards compatibility with Gemini API)."""
        pass


class OpenRouterClient(LLMClient):
    """OpenRouter API client using OpenAI-compatible interface."""
    
    def __init__(self, api_key: str, model: str = "deepseek/deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Import here to avoid dependency issues
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            logger.info(f"OpenRouter client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """Generate a response from OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    def generate_content(self, prompt: str) -> object:
        """
        Generate content with Gemini-compatible response object.
        This provides backwards compatibility with existing code that uses model.generate_content().
        """
        class GeminiCompatResponse:
            def __init__(self, text: str):
                self.text = text
        
        response_text = self.generate(prompt)
        return GeminiCompatResponse(response_text)


class GeminiClient(LLMClient):
    """Legacy Gemini client for backwards compatibility."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("google-generativeai package required")
        except Exception as e:
            logger.warning(f"Gemini initialization failed: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def generate_content(self, prompt: str) -> object:
        """Native Gemini generate_content for full compatibility."""
        return self.model.generate_content(prompt)


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClient:
    """
    Get an LLM client instance.
    
    Priority order:
    1. OpenRouter (if OPENROUTER_API_KEY is set)
    2. Gemini (if GEMINI_API_KEY is set)
    
    Args:
        provider: Force a specific provider ("openrouter" or "gemini")
        api_key: Override the API key
        model: Override the model name
    
    Returns:
        LLMClient instance
    
    Raises:
        ValueError: If no API keys are configured
    """
    
    # Check for OpenRouter first (preferred)
    openrouter_key = api_key if provider == "openrouter" else os.getenv("OPENROUTER_API_KEY")
    openrouter_model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    
    # Check for Gemini (fallback)
    gemini_key = api_key if provider == "gemini" else os.getenv("GEMINI_API_KEY")
    
    # Provider selection logic
    if provider == "openrouter" or (provider is None and openrouter_key):
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not configured")
        try:
            return OpenRouterClient(api_key=openrouter_key, model=openrouter_model)
        except ImportError:
            logger.warning("OpenAI package not installed, falling back to Gemini")
            if gemini_key:
                return GeminiClient(api_key=gemini_key)
            raise
    
    elif provider == "gemini" or gemini_key:
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not configured")
        return GeminiClient(api_key=gemini_key, model=model or "gemini-1.5-flash")
    
    else:
        raise ValueError(
            "No LLM API key configured. Set OPENROUTER_API_KEY (recommended) or GEMINI_API_KEY in .env"
        )


# Convenience function for simple usage
def generate_text(prompt: str) -> str:
    """Quick helper to generate text without managing client lifecycle."""
    client = get_llm_client()
    return client.generate(prompt)
