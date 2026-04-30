"""v7_1 Configuration - Centralized settings for JORT Parser v7.1."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from v7_1 directory
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# =============================================================================
# PATHS (WSL)
# =============================================================================

# Base directory: .../jort/parser/src/jort_parser/v7
V7_DIR = Path(__file__).resolve().parent  # jort_parser/v7
INPUT_DIR = V7_DIR / "input"   # v7/input/
OUTPUT_DIR = V7_DIR / "output"  # v7/output/

MD_INPUT_DIR = INPUT_DIR
PDF_INPUT_DIR = INPUT_DIR  # PDFs also in input dir
V7_OUTPUT_DIR = OUTPUT_DIR
PROGRESS_DIR = OUTPUT_DIR / "progress"

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API KEYS (loaded from environment variables)
# =============================================================================

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =============================================================================
# MODELS
# =============================================================================

# LLM Providers (fixed: meta-llama, not meta-llama)
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # Removed :fastest suffix
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"  # OpenRouter model ID
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Fallback Models
ARABERT_MODEL = "aubmindlab/bert-base-arabertv2"
CAMELBERT_NER_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner"
CAMELBERT_DA_NER_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-ner"

# =============================================================================
# THRESHOLDS
# =============================================================================

# Confidence thresholds
ARABERT_CONFIDENCE_THRESHOLD = 0.70
NER_CONFIDENCE_THRESHOLD = 0.70

# Progress save settings
PROGRESS_SAVE_INTERVAL = 10  # Save every N articles

# LLM retry settings
LLM_BASE_DELAY = 15  # seconds
LLM_MAX_DELAY = 60   # seconds
LLM_MAX_RETRIES = 5

# =============================================================================
# LLM PROVIDER ORDER
# =============================================================================

LLM_PROVIDER_ORDER = ["hf", "groq"]  # Primary -> Fallback

# =============================================================================
# FIELD ROUTING TABLE (only in processor.py, not in providers)
# =============================================================================

FIELD_ROUTING = {
    "legal_domain": {"tool": "arabert", "fallback_llm": True, "conf_threshold": 0.70},
    "legal_type": {"tool": "arabert", "fallback_llm": True, "conf_threshold": 0.70},
    "sanctions": {"tool": "regex", "fallback_llm": True, "conf_threshold": None},
    "key_concepts": {"tool": "camel_ner", "fallback_llm": True, "conf_threshold": 0.70},
    "crime_type": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "legal_action": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "acts": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "objects": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "means": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "intent": {"tool": "keyword", "fallback_llm": True, "conf_threshold": None},
    "procedures": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "rights": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "obligations": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
    "banking_related": {"tool": "llm", "fallback_llm": False, "conf_threshold": None},
}

# =============================================================================
# LLM SCHEMA (10 fields from SESSION_SUMMARY)
# =============================================================================

LLM_SCHEMA = {
    "legal_domain": None,
    "legal_type": None,
    "legal_action": None,
    "crime_type": None,
    "sanctions": [],
    "procedures": [],
    "rights": [],
    "obligations": [],
    "banking_related": False,
    "key_concepts": []
}

# =============================================================================
# LLM PROMPT (from SESSION_SUMMARY)
# =============================================================================

LLM_PROMPT = """You are a legal information extraction system specialized in Tunisian law.

TASK: Extract structured metadata from the Arabic legal article below.

RULES:
- Do NOT invent information.
- Extract only what is explicitly stated or clearly implied.
- Keep legal terminology in Arabic.
- Output ONLY valid JSON (no markdown, no explanation).

FIELDS TO EXTRACT:
- legal_action: Type of legal action (e.g., "تجريم", "عقوبة", "إجراء", null if not applicable)
- crime_type: Type of crime if specified (e.g., "سرقة", "احتيال", null if not a crime article)
- banking_related: Is this related to banking/financial sector? (true/false)
- key_concepts: Array of key legal concepts (e.g., ["الدعوى العمومية", "العقوبة"])
- sanctions: Array of sanctions mentioned (e.g., ["سجن", "غرامة"], empty if none)

ARTICLE TEXT:
{article_text}

OUTPUT JSON ONLY:"""

# =============================================================================
# REGEX PATTERNS (from v6.1, preserved)
# =============================================================================

ARTICLE_PATTERNS = [
    r"\*\*الفصل\s+\d+\s*-\*\*",     # **الفصل 2-** (bold with closing)
    r"\*\*الفصل\s+الأول\s*-\*\*",   # **الفصل الأول-** (ordinal bold)
    r"\*\*الفصل\s+الوحيد\s*-\*\*",  # **الفصل الوحيد-** (unique)
    r"^##\s*الفصل\s+\d+",          # ## الفصل 2 (markdown header)
    r"^##\s*الفصل\s+الأول",        # ## الفصل الأول (markdown ordinal)
]

# Sanctions regex patterns (owned by regex, NOT NER)
SANCTIONS_PATTERNS = [
    r"يغرم",
    r"سجن",
    r"غرامة",
    r"عقوبة",
    r"خطية",
    r"ايداع",
]
