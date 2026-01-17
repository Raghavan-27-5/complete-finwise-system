"""
Quick test script to verify all API connections are working.
Run this to ensure your .env is configured correctly.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_neo4j():
    """Test Neo4j Aura connection"""
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("AURA_CONNECTION_URI")
        username = os.getenv("AURA_USERNAME")
        password = os.getenv("AURA_PASSWORD")
        
        if not all([uri, username, password]):
            print("[FAIL] Neo4j: Missing credentials in .env")
            return False
        
        # neo4j+s:// scheme already handles SSL, no extra config needed
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()
        print("[OK] Neo4j Aura: Connected successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Neo4j Aura: {str(e)[:100]}")
        return False

def test_openrouter():
    """Test OpenRouter API (replaces Gemini)"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        
        if not api_key:
            print("[SKIP] OpenRouter: Missing API key (add OPENROUTER_API_KEY to .env)")
            return None
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        
        result = response.choices[0].message.content
        # Handle unicode for Windows console
        result_safe = result.encode('ascii', 'replace').decode('ascii')[:30]
        print(f"[OK] OpenRouter API: Connected ({model}) - Response: {result_safe}")
        return True
    except ImportError:
        print("[FAIL] OpenRouter: openai package not installed (pip install openai)")
        return False
    except Exception as e:
        print(f"[FAIL] OpenRouter API: {str(e)[:80]}")
        return False

def test_newsapi():
    """Test NewsAPI"""
    try:
        import requests
        api_key = os.getenv("NEWS_API_KEY")
        
        if not api_key:
            print("[SKIP] NewsAPI: Missing API key (optional)")
            return None
            
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("[OK] NewsAPI: Connected successfully")
            return True
        else:
            print(f"[FAIL] NewsAPI: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] NewsAPI: {e}")
        return False

def test_finnhub():
    """Test Finnhub API"""
    try:
        import requests
        api_key = os.getenv("FINNHUB_API_KEY")
        
        if not api_key:
            print("[SKIP] Finnhub: Missing API key (recommended)")
            return None
            
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'c' in data:  # 'c' is current price
                print(f"[OK] Finnhub API: Connected successfully - AAPL: ${data['c']}")
                return True
        print(f"[FAIL] Finnhub: HTTP {response.status_code}")
        return False
    except Exception as e:
        print(f"[FAIL] Finnhub: {e}")
        return False

def test_alpha_vantage():
    """Test Alpha Vantage API"""
    try:
        import requests
        api_key = os.getenv("ALPHA_VANTAGE_KEY")
        
        if not api_key:
            print("[SKIP] Alpha Vantage: Missing API key (optional)")
            return None
            
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'Symbol' in data:
                print("[OK] Alpha Vantage: Connected successfully")
                return True
        print(f"[FAIL] Alpha Vantage: HTTP {response.status_code}")
        return False
    except Exception as e:
        print(f"[FAIL] Alpha Vantage: {e}")
        return False

def test_reddit():
    """Test Reddit API"""
    try:
        import praw
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT")
        
        if not all([client_id, client_secret, user_agent]):
            print("[SKIP] Reddit: Missing credentials (optional)")
            return None
            
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Test by accessing read-only endpoint
        reddit.subreddit("wallstreetbets").hot(limit=1)
        print("[OK] Reddit API: Connected successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Reddit API: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FinWise Core - API Connection Test")
    print("="*60 + "\n")
    
    results = {
        "Neo4j Aura (Required)": test_neo4j(),
        "OpenRouter LLM (Required)": test_openrouter(),
        "NewsAPI": test_newsapi(),
        "Finnhub": test_finnhub(),
        "Alpha Vantage": test_alpha_vantage(),
        "Reddit": test_reddit(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    required_ok = results["Neo4j Aura (Required)"] and results["OpenRouter LLM (Required)"]
    optional_count = sum(1 for v in results.values() if v is True)
    
    print(f"\nRequired APIs: {'OK - All working' if required_ok else 'FAILED - Some failed'}")
    print(f"Optional APIs: {optional_count}/6 configured")
    
    if required_ok:
        print("\nSUCCESS: Your FinWise Core is ready to run!")
    else:
        print("\nWARNING: Fix required API connections before proceeding.")
    
    print("\n")
