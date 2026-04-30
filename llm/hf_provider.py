"""v7 HF Inference Provider - uses requests directly to avoid Groq routing."""
 
from typing import Dict, Any
from pathlib import Path
import requests
import json
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
 
from jort_parser.v7_1.config import HF_MODEL, HF_TOKEN, LLM_PROMPT
from jort_parser.v7_1.utils import repair_json
from jort_parser.v7_1.llm.base import BaseLLMProvider
 
 
class HFProvider(BaseLLMProvider):
    """HuggingFace Inference Provider using direct HTTP requests."""
    
    def __init__(self, api_token: str = None):
        # Remove :fastest suffix
        self.model = HF_MODEL.split(':')[0]
        super().__init__(model=self.model)
        self.available = False
        
        self.token = api_token or HF_TOKEN
        if not self.token:
            return
        
        # Strip any whitespace/newlines from token
        self.token = self.token.strip()
        
        # Use HF Router API - correct endpoint that avoids auto-routing to Groq
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        self.available = True
    
    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Extract data using HF Inference via direct HTTP request.
        Only handles API call, NO routing logic.
        """
        if not self.available:
            return {"error": "HF provider not available"}
        
        for attempt in range(self.max_retries):
            try:
                # Use chat completions format (correct for router.huggingface.co)
                payload = {
                    "model": "meta-llama/Llama-3.3-70B-Instruct",
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
