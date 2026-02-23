import httpx
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SentimentAgent:
    """Analyzes crypto news and social media sentiment"""
    
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def analyze_news_sentiment(self, symbols):
        """Fetch and analyze recent crypto news"""
        cache_key = f"news_{datetime.now().strftime('%Y%m%d%H')}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Fetch recent news (you'd integrate with NewsAPI, CryptoPanic, etc.)
            news_headlines = await self._fetch_crypto_news(symbols)
            
            # Use LLM to analyze sentiment
            prompt = f"""Analyze the sentiment of these crypto news headlines. 
            Return a JSON with:
            - overall_sentiment: (bullish/bearish/neutral)
            - confidence: 0-100
            - key_drivers: main factors driving sentiment
            
            Headlines: {news_headlines[:10]}"""
            
            sentiment = await self._call_llm(prompt)
            self.cache[cache_key] = sentiment
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'overall_sentiment': 'neutral', 'confidence': 50}
    
    async def _call_llm(self, prompt):
        """Call LLM API (OpenRouter, OpenAI, etc.)"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=10
            )
            return response.json()