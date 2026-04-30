"""v7 Utilities - Shared helpers for PyArabic, progress I/O, normalization."""

import json
import re
from pathlib import Path
import sys

# Try importing PyArabic
try:
    import pyarabic.araby as araby
    PYARABIC_AVAILABLE = True
except ImportError:
    PYARABIC_AVAILABLE = False
    araby = None

# Try importing json_repair
try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False


# =============================================================================
# PYARABIC WRAPPERS (replaces v6.1 ARABIC_NUMBERS dict)
# =============================================================================

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text: remove diacritics, standardize chars."""
    if not PYARABIC_AVAILABLE:
        # Fallback: basic normalization without PyArabic
        text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove tashkeel
        text = text.replace('\u0640', '')  # Remove tatweel
        return text.strip()
    
    # Use PyArabic for proper normalization
    text = araby.strip_tashkeel(text)  # Remove diacritics
    text = araby.strip_tatweel(text)    # Remove tatweel (kashida)
    text = araby.normalize_hamza(text)  # Standardize hamza variants
    return text.strip()


def convert_arabic_number(text: str) -> str:
    """
    Convert Arabic/Hindi numerals to Western numerals.
    Also handles Arabic ordinal numbers (الأول، الثاني، etc.)
    Replaces v6.1 ARABIC_NUMBERS dict with PyArabic's comprehensive handling.
    """
    if not PYARABIC_AVAILABLE:
        # Fallback to v6.1 logic if PyArabic not available
        ARABIC_NUMBERS = {
            "الأول": "1", "الثاني": "2", "الثالث": "3", "الرابع": "4", "الخامس": "5",
            "السادس": "6", "السابع": "7", "الثامن": "8", "التاسع": "9", "العاشر": "10",
            "الثاني عشر": "12", "الثالث عشر": "13", "الرابع عشر": "14", "الخامس عشر": "15",
            "السادس عشر": "16", "السابع عشر": "17", "الثامن عشر": "18", "التاسع عشر": "19",
            "العشرون": "20", "وحيد": "1"
        }
        for ar, num in ARABIC_NUMBERS.items():
            if ar in text:
                return num
        # Try digits
        m = re.search(r"الفصل\s+(\d+)", text)
        if m:
            return m.group(1)
        return "?"
    
    # Use PyArabic's number conversion
    # First try ordinal words (الأول، الثاني، etc.)
    ordinal_map = {
        "الأول": "1", "الثاني": "2", "الثالث": "3", "الرابع": "4", "الخامس": "5",
        "السادس": "6", "السابع": "7", "الثامن": "8", "التاسع": "9", "العاشر": "10",
        "الحادي عشر": "11", "الثاني عشر": "12", "الثالث عشر": "13", "الرابع عشر": "14",
        "الخامس عشر": "15", "السادس عشر": "16", "السابع عشر": "17", "الثامن عشر": "18",
        "التاسع عشر": "19", "العشرون": "20", "الوحيد": "1"
    }
    
    for ar, num in ordinal_map.items():
        if ar in text:
            return num
    
    # Try to extract digit numbers from text
    m = re.search(r"الفصل\s+(\d+)", text)
    if m:
        return m.group(1)
    
    # Try converting Hindi numerals if present
    hindi_digits = araby.arabic2roman(text)
    if hindi_digits and hindi_digits != text:
        return hindi_digits
    
    return "?"


# =============================================================================
# JSON REPAIR HELPER
# =============================================================================

def repair_json(llm_output: str) -> dict:
    """
    Parse potentially malformed JSON from LLM output.
    Uses json_repair if available, otherwise basic cleanup.
    """
    if JSON_REPAIR_AVAILABLE:
        try:
            return json_repair.loads(llm_output)
        except Exception:
            pass
    
    # Fallback: basic JSON cleanup
    text = llm_output.strip()
    
    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    
    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to fix common issues
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)
        try:
            return json.loads(text)
        except:
            return {"error": "parse_failed", "raw": llm_output[:300]}


# =============================================================================
# PROGRESS SAVE/LOAD (for resume support)
# =============================================================================

def save_progress(progress_dir: Path, code_name: str, last_index: int, processed_articles: list):
    """Save processing progress to JSON file."""
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"{code_name}_progress.json"
    
    progress_data = {
        "last_index": last_index,
        "processed_articles": processed_articles,  # List of article_numbers already done
        "timestamp": str(Path(__file__).stat().st_mtime if Path(__file__).exists() else None)
    }
    
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


def load_progress(progress_dir: Path, code_name: str) -> dict:
    """
    Load processing progress from JSON file.
    Returns dict with 'last_index' and 'processed_articles', or None if no progress file.
    """
    progress_file = progress_dir / f"{code_name}_progress.json"
    
    if not progress_file.exists():
        return None
    
    try:
        with open(progress_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_progress(progress_dir: Path, code_name: str):
    """Clear progress file for a given code."""
    progress_file = progress_dir / f"{code_name}_progress.json"
    if progress_file.exists():
        progress_file.unlink()


# =============================================================================
# CHARACTER NORMALIZATION
# =============================================================================

def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for comparison purposes.
    Removes diacritics, standardizes spaces, lowercases if applicable.
    """
    if not text:
        return ""
    
    text = normalize_arabic_text(text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


# =============================================================================
# CHECK DEPENDENCIES
# =============================================================================

def check_dependencies() -> dict:
    """Check which dependencies are available."""
    return {
        "pyarabic": PYARABIC_AVAILABLE,
        "json_repair": JSON_REPAIR_AVAILABLE,
        "camel_tools": False,  # Will be updated when camel_tools is imported elsewhere
        "groq": False,  # Will be updated in llm module
        "huggingface_hub": False,  # Will be updated in llm module
    }
