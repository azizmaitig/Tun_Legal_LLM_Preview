"""v7 Groq Provider - llama-3.3-70b-versatile."""

from typing import Dict, Any
from pathlib import Path

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jort_parser.v7_1.config import GROQ_MODEL, GROQ_API_KEY, LLM_PROMPT
from jort_parser.v7_1.utils import repair_json
from jort_parser.v7_1.llm.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider."""
    
    def __init__(self, api_key: str = None):
        super().__init__(model=GROQ_MODEL)
        self.available = False
        
        if not GROQ_AVAILABLE:
            return
        
        key = api_key or GROQ_API_KEY
        if not key:
            return
        
        try:
            self.client = Groq(api_key=key)
            self.available = True
        except Exception as e:
            print(f"Groq init failed: {e}")
    
    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Extract data using Groq.
        Only handles API call, NO routing logic.
        """
        if not self.available:
            return {"error": "Groq provider not available"}
        
        for attempt in range(self.max_retries):
            try:
                chat = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": LLM_PROMPT},
                        {"role": "user", "content": prompt[:1500]}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                result = chat.choices[0].message.content.strip()
                self._reset_429_counter()
                
                # Parse JSON
                parsed = repair_json(result)
                
                # Handle list response
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else {}
                elif not isinstance(parsed, dict):
                    parsed = {}
                
                return parsed
                
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    should_retry = self._handle_429(attempt)
                    if not should_retry:
                        return {"error": "429_rate_limit", "switch_provider": True}
                    continue
                else:
                    return {"error": error_msg[:150]}
        
        return {"error": "max_retries_exceeded"}
