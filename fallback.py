"""v7 Fallback Tools - AraBERT, Regex Sanctions, CAMeL Tools NER."""

import re
from pathlib import Path

# Try importing dependencies
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from camel_tools.ner import NERecognizer
    CAMEL_NER_AVAILABLE = True
except ImportError:
    CAMEL_NER_AVAILABLE = False

# v7_1 imports
from jort_parser.v7_1.config import (
    ARABERT_MODEL, CAMELBERT_NER_MODEL, CAMELBERT_DA_NER_MODEL,
    ARABERT_CONFIDENCE_THRESHOLD, NER_CONFIDENCE_THRESHOLD,
    SANCTIONS_PATTERNS
)


class AraBERTClassifier:
    """AraBERT classifier for legal_domain and legal_type."""
    
    def __init__(self):
        self.classifier = None
        self.available = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model=ARABERT_MODEL,
                    tokenizer=ARABERT_MODEL
                )
                self.available = True
            except Exception as e:
                print(f"AraBERT init failed: {e}")
    
    def predict(self, text: str, task: str) -> dict:
        """
        Predict legal_domain or legal_type.
        Returns dict with 'prediction', 'confidence', 'all_scores'.
        """
        if not self.available:
            return {"prediction": None, "confidence": 0.0, "error": "AraBERT not available"}
        
        try:
            # Truncate text for classifier
            truncated = text[:512]
            results = self.classifier(truncated)
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    "prediction": result.get("label"),
                    "confidence": result.get("score", 0.0),
                    "all_scores": results
                }
        except Exception as e:
            return {"prediction": None, "confidence": 0.0, "error": str(e)}
        
        return {"prediction": None, "confidence": 0.0}


class RegexSanctionsDetector:
    """Regex-based sanctions detection (owned by regex, NOT NER)."""
    
    def __init__(self):
        self.patterns = [re.compile(p) for p in SANCTIONS_PATTERNS]
    
    def detect(self, text: str) -> list:
        """
        Detect sanctions in text using regex patterns.
        Returns list of {"type", "condition", "value"} dicts.
        """
        sanctions = []
        
        for pattern in self.patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Basic extraction - can be enhanced
                sanction = {
                    "type": match if isinstance(match, str) else " ".join(match),
                    "condition": "",
                    "value": ""
                }
                sanctions.append(sanction)
        
        return sanctions
    
    def has_sanctions(self, text: str) -> bool:
        """Quick check if text contains any sanction patterns."""
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False


class CamelNERecognizer:
    """CAMeL Tools NERecognizer for key_concepts extraction."""
    
    def __init__(self):
        self.recognizer = None
        self.available = False
        
        if CAMEL_NER_AVAILABLE:
            try:
                self.recognizer = NERecognizer(CAMELBERT_NER_MODEL)
                self.available = True
            except Exception as e:
                print(f"CAMeL NER init failed: {e}")
                # Try fallback to DA model
                try:
                    self.recognizer = NERecognizer(CAMELBERT_DA_NER_MODEL)
                    self.available = True
                    print(f"Fallback to DA NER model")
                except Exception as e2:
                    print(f"DA NER fallback failed: {e2}")
    
    def extract_key_concepts(self, text: str, min_confidence: float = NER_CONFIDENCE_THRESHOLD) -> list:
        """
        Extract key legal concepts using NER.
        Returns list of concept strings (5-8 concepts).
        """
        if not self.available:
            return []
        
        try:
            # Tokenize text for CAMeL Tools
            from camel_tools.tokenizers.word import simple_word_tokenize
            tokens = simple_word_tokenize(text)
            
            # Get NER predictions
            predictions = self.recognizer.predict_sentence(tokens)
            
            # Extract entities (filter by confidence if available)
            concepts = []
            for token, tag in predictions:
                if tag != 'O':  # Not 'Other' - it's a named entity
                    if token not in concepts:
                        concepts.append(token)
            
            return concepts[:8]  # Limit to 8 concepts
            
        except Exception as e:
            print(f"NER extraction failed: {e}")
            return []
    
    def get_entities_with_confidence(self, text: str) -> list:
        """Get entities with confidence scores if available."""
        if not self.available:
            return []
        
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            tokens = simple_word_tokenize(text)
            predictions = self.recognizer.predict_sentence(tokens)
            return predictions
        except Exception:
            return []


# Singleton instances for reuse
_arabert_classifier = None
_regex_detector = None
_camel_ner = None

def get_arabert_classifier():
    global _arabert_classifier
    if _arabert_classifier is None:
        _arabert_classifier = AraBERTClassifier()
    return _arabert_classifier

def get_regex_detector():
    global _regex_detector
    if _regex_detector is None:
        _regex_detector = RegexSanctionsDetector()
    return _regex_detector

def get_camel_ner():
    global _camel_ner
    if _camel_ner is None:
        _camel_ner = CamelNERecognizer()
    return _camel_ner
