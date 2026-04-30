"""v7 Processor - Centralized field routing, cost-aware, resume support."""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# v7_1 imports
from jort_parser.v7_1.config import (
    V7_OUTPUT_DIR, PROGRESS_DIR, PROGRESS_SAVE_INTERVAL,
    FIELD_ROUTING, LLM_SCHEMA, LLM_PROMPT
)
from jort_parser.v7_1.utils import save_progress, load_progress, repair_json
from jort_parser.v7_1.fallback import (
    get_arabert_classifier, get_regex_detector, get_camel_ner
)
from jort_parser.v7_1.llm.hf_provider import HFProvider
from jort_parser.v7_1.llm.groq_provider import GroqProvider


class ProcessingStats:
    """Track processing statistics."""
    
    def __init__(self):
        self.total = 0
        self.llm_calls = 0
        self.fallback_used = 0
        self.errors = 0
        self.skipped = 0


class LLMProcessor:
    """Process articles with cost-aware field routing."""
    
    def __init__(self, hf_token: str = None, groq_key: str = None):
        self.hf_provider = HFProvider(hf_token)
        self.groq_provider = GroqProvider(groq_key)
        self.stats = ProcessingStats()
        
        # Lazy-load fallback tools (avoid hanging at init)
        self._arabert = None
        self._regex_detector = None
        self._camel_ner = None
    
    @property
    def arabert(self):
        if self._arabert is None:
            self._arabert = get_arabert_classifier()
        return self._arabert
    
    @property
    def regex_detector(self):
        if self._regex_detector is None:
            self._regex_detector = get_regex_detector()
        return self._regex_detector
    
    @property
    def camel_ner(self):
        if self._camel_ner is None:
            self._camel_ner = get_camel_ner()
        return self._camel_ner
    
    def _get_llm_provider(self, prefer: str = "hf"):
        """Get LLM provider based on preference and availability."""
        if prefer == "hf" and self.hf_provider.available:
            return self.hf_provider, "hf"
        elif self.groq_provider.available:
            return self.groq_provider, "groq"
        elif self.hf_provider.available:
            return self.hf_provider, "hf"
        else:
            return None, None
    
    def _process_field_with_tool(self, field: str, article: Dict, routing: Dict) -> Any:
        """
        Process a single field using the appropriate tool.
        Returns (value, used_llm: bool).
        """
        tool = routing.get("tool")
        fallback_llm = routing.get("fallback_llm", False)
        threshold = routing.get("conf_threshold")
        
        # Route to appropriate tool
        if tool == "arabert":
            if not self.arabert.available:
                return None, False
            
            text = article.get("content", "")
            result = self.arabert.predict(text, field)
            
            if result.get("confidence", 0) >= (threshold or 0.7):
                return result.get("prediction"), False
            elif fallback_llm:
                self.stats.fallback_used += 1
                return None, True  # Signal to use LLM
            else:
                return result.get("prediction"), False
        
        elif tool == "regex":
            text = article.get("content", "")
            if field == "sanctions":
                sanctions = self.regex_detector.detect(text)
                if sanctions or not fallback_llm:
                    return sanctions, False
                else:
                    return None, True  # Signal to use LLM
            return None, False
        
        elif tool == "camel_ner":
            if not self.camel_ner.available:
                return [], False
            
            text = article.get("content", "")
            concepts = self.camel_ner.extract_key_concepts(text)
            
            if len(concepts) >= 3 or not fallback_llm:
                return concepts, False
            else:
                self.stats.fallback_used += 1
                return None, True  # Signal to use LLM
        
        elif tool == "keyword":
            # Simple keyword detection for intent
            text = article.get("content", "")
            keywords = ["عمدا", "علم", "قصد", "نية", "ترصد"]
            for kw in keywords:
                if kw in text:
                    return "explicit", False
            return None, fallback_llm
        
        elif tool == "llm":
            return None, True  # Always use LLM
        
        return None, False
    
    def process_article(self, article: Dict) -> Dict:
        """Process a single article with cost-aware routing."""
        result = dict(LLM_SCHEMA)
        llm_fields_needed = []
        
        # First pass: try deterministic tools
        for field, routing in FIELD_ROUTING.items():
            value, needs_llm = self._process_field_with_tool(field, article, routing)
            
            if needs_llm:
                llm_fields_needed.append(field)
            else:
                result[field] = value
        
        # Second pass: call LLM only for fields that need it
        if llm_fields_needed:
            self.stats.llm_calls += 1
            llm_result = self._call_llm_for_fields(article, llm_fields_needed)
            
            for field in llm_fields_needed:
                if field in llm_result:
                    result[field] = llm_result[field]
        
        # Add llm_ prefix for output
        output = {}
        for key, value in result.items():
            output[f"llm_{key}"] = value
        
        return output
    
    def _call_llm_for_fields(self, article: Dict, fields: List[str]) -> Dict:
        """Call LLM with prompt containing only needed fields."""
        provider, provider_name = self._get_llm_provider("hf")
        
        if not provider:
            return {}
        
        # Build prompt mentioning only needed fields
        fields_str = ", ".join(fields)
        prompt = f"{LLM_PROMPT}\n\nFocus on these fields: {fields_str}\n\nArticle content: {article.get('content', '')[:1500]}"
        
        result = provider.extract(prompt)
        
        # Check if provider switch needed
        if result.get("switch_provider"):
            provider, provider_name = self._get_llm_provider("groq")
            if provider:
                result = provider.extract(prompt)
        
        return result if isinstance(result, dict) else {}
    
    def process_file(
        self,
        input_file: Path,
        output_file: Path = None,
        start_idx: int = 0,
        end_idx: int = None,
        resume: bool = True
    ) -> Dict:
        """Process JSON file with resume support."""
        
        print(f"Loading: {input_file}")
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)
        
        articles = data.get("articles", [])
        self.stats.total = len(articles)
        
        # Load progress if resuming
        progress = None
        if resume:
            progress = load_progress(PROGRESS_DIR, data.get("code_name", ""))
            if progress:
                start_idx = max(start_idx, progress.get("last_index", 0))
                print(f"Resuming from article index {start_idx}")
        
        # Determine end index
        if end_idx is None:
            end_idx = len(articles)
        
        output_file = output_file or (V7_OUTPUT_DIR / f"{data['code_name']}_enhanced.json")
        
        print(f"Processing articles {start_idx+1} to {end_idx} of {len(articles)}...")
        print(f"LLM Provider: {'HF + Groq fallback' if self.hf_provider.available else 'Groq' if self.groq_provider.available else 'None'}")
        print("=" * 50)
        
        # Process articles
        for idx in range(start_idx, min(end_idx, len(articles))):
            article = articles[idx]
            num = article.get("article_number", "?")
            
            print(f"  [{idx+1}/{len(articles)}] Article #{num}...", end=" ", flush=True)
            
            try:
                llm_data = self.process_article(article)
                
                # Update article with LLM data
                for key, value in llm_data.items():
                    article[key] = value
                
                print("OK")
                
            except Exception as e:
                print(f"Error: {str(e)[:50]}")
                self.stats.errors += 1
            
            # Save progress periodically
            if (idx + 1) % PROGRESS_SAVE_INTERVAL == 0:
                save_progress(
                    PROGRESS_DIR,
                    data.get("code_name", ""),
                    idx + 1,
                    [a.get("article_number") for a in articles[:idx+1]]
                )
                print(f"  Progress saved at article {idx+1}")
            
            # Delay between articles (avoid rate limits)
            if idx < min(end_idx, len(articles)) - 1:
                time.sleep(3)
        
        # Save final output
        print(f"\nSaving to: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        print("\n=== Processing Complete ===")
        print(f"Total articles: {self.stats.total}")
        print(f"LLM calls: {self.stats.llm_calls}")
        print(f"Fallback used: {self.stats.fallback_used}")
        print(f"Errors: {self.stats.errors}")
        
        return data
