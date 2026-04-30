"""v7 LLM Base - Retry logic, exponential backoff, provider switching."""

import time
from typing import Optional, Dict, Any

class BaseLLMProvider:
    """Base class for LLM providers with retry logic."""
    
    def __init__(self, model: str, delay: int = 15, max_delay: int = 60, max_retries: int = 5):
        self.model = model
        self.delay = delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.consecutive_429s = 0
    
    def _calculate_delay(self, attempt: int) -> int:
        """Exponential backoff: 15s -> 30s -> 60s (capped)."""
        delay = min(self.delay * (2 ** attempt), self.max_delay)
        return int(delay)
    
    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Extract data using LLM. To be implemented by subclasses.
        Returns dict with extracted fields or error.
        """
        raise NotImplementedError
    
    def _handle_429(self, attempt: int) -> bool:
        """
        Handle 429 rate limit error.
        Returns True if should retry, False if should switch provider.
        """
        self.consecutive_429s += 1
        
        if self.consecutive_429s >= 3:
            # Too many consecutive 429s, signal provider switch
            return False
        
        delay = self._calculate_delay(attempt)
        print(f"  429 rate limit, waiting {delay}s...")
        time.sleep(delay)
        return True
    
    def _reset_429_counter(self):
        """Reset consecutive 429 counter on successful call."""
        self.consecutive_429s = 0
