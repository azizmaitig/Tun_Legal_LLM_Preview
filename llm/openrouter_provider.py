"""v7 OpenRouter Provider - Unified API via openrouter.ai."""
import requests
import json
from typing import Dict, Any

from jort_parser.v7_1.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, LLM_PROMPT
from jort_parser.v7_1.utils import repair_json
from jort_parser.v7_1.llm.base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter Provider - routes to multiple LLM providers."""
    
    def __init__(self, api_key: str = None):
        # Remove :fastest suffix
        self.model = OPENROUTER_MODEL.split(':')[0]
        super().__init__(model=self.model)
        self.available = False
        
        self.token = api_key or OPENROUTER_API_KEY
        if not self.token:
            return
        
        # OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.available = True
    
    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Extract data using OpenRouter API.
        Only handles API call, NO routing logic.
        """
        if not self.available:
            return {"error": "OpenRouter provider not available"}
        
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": LLM_PROMPT},
                        {"role": "user", "content": prompt[:2000]}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 800
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse chat completions response
                    if "choices" in result and len(result["choices"]) > 0:
                        generated = result["choices"][0]["message"]["content"]
                    else:
                        generated = str(result)
                    
                    self._reset_429_counter()
                    
                    # Parse JSON from generated text
                    parsed = repair_json(generated)
                    if isinstance(parsed, list):
                        parsed = parsed[0] if parsed else {}
                    elif not isinstance(parsed, dict):
                        parsed = {}
                    
                    return parsed
                    
                elif response.status_code == 429:
                    should_retry = self._handle_429(attempt)
                    if not should_retry:
                        return {"error": "429_rate_limit"}
                    continue
                    
                else:
                    return {"error": f"HTTP {response.status_code}: {response.text[:150]}"}
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    should_retry = self._handle_429(attempt)
                    if not should_retry:
                        return {"error": "429_rate_limit"}
                    continue
                else:
                    return {"error": error_msg[:150]}
        
        return {"error": "max_retries_exceeded"}
