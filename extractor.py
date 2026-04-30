#!/usr/bin/env python3
"""v7 Article Extractor - PyArabic + regex + CAMeL POS validation."""

import sys
import json
import re
from pathlib import Path
from typing import Optional

# Dependencies
try:
    import fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from camel_tools.tagger import DefaultTagger
    from camel_tools.tokenizers.word import simple_word_tokenize
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:
    CAMEL_TOOLS_AVAILABLE = False

# v7 imports
from jort_parser.v7.config import V7_OUTPUT_DIR, MD_INPUT_DIR, PDF_INPUT_DIR, ARTICLE_PATTERNS
from jort_parser.v7.utils import convert_arabic_number

# Initialize CAMeL Tools tagger
_tagger = None
if CAMEL_TOOLS_AVAILABLE:
    try:
        _tagger = DefaultTagger()
    except Exception:
        _tagger = None


class ArticleExtractor:
    """Extract articles from Arabic legal codes."""
    
    def __init__(self, code_file: str, pdf_file: str = None):
        self.file_path = Path(code_file)
        self.content = self.file_path.read_text(encoding="utf-8")
        self.code_name = self.file_path.stem
        self.tagger = _tagger
        
        self.char_to_page = {}
        if pdf_file and PDF_AVAILABLE and Path(pdf_file).exists():
            self._build_page_map(Path(pdf_file))
        else:
            self._estimate_page_map()
    
    def _build_page_map(self, pdf_path: Path):
        """Build char position to page number map from PDF."""
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                idx = self.content.find(text.strip()[:80])
                if idx >= 0:
                    self.char_to_page[idx] = page_num
        doc.close()
        if not self.char_to_page:
            self._estimate_page_map()
    
    def _estimate_page_map(self):
        """Estimate pages from content length."""
        total = len(self.content)
        for i in range(max(1, total // 3500)):
            self.char_to_page[(i * total) // max(1, total // 3500)] = i + 1
    
    def _find_page(self, char_pos: int) -> int:
        page = 1
        for cp, p in sorted(self.char_to_page.items()):
            if cp <= char_pos:
                page = p
        return page
    
    def _normalize_article_num(self, text: str) -> str:
        """Normalize article number using PyArabic."""
        return convert_arabic_number(text)
    
    def _extract_content(self, start: int, end: int) -> str:
        """Extract and clean article content."""
        text = self.content[start:end].strip()
        text = re.sub(r"\*\*", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        first_dash = text.find("-")
        if first_dash > 0 and first_dash < 30:
            text = text[first_dash + 1:].strip()
        
        # Split into lines and stop at structural headings
        lines = text.split("\n")
        content_lines = []
        for line in lines:
            stripped = line.strip()
            # Stop if we hit a structural heading (باب، قسم، كتاب، with or without ال)
            if re.match(r"^#+\s*(ال)?(باب|قسم|كتاب)\s+", stripped):
                break
            # Also stop at any markdown header that isn't an article header
            if re.match(r"^##+\s+", stripped) and not re.match(r"الفصل\s+", stripped):
                break
            content_lines.append(line)
        
        text = "\n".join(content_lines)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000]
    
    def _validate_header_pos(self, text: str) -> bool:
        """Validate header using CAMeL POS tagging."""
        if not self.tagger:
            return True
        try:
            tokens = simple_word_tokenize(text)
            tagged = self.tagger.tag(tokens)
            for word, tag in tagged:
                if word in ["كتب", "باب", "قسم", "فصل"]:
                    return tag.startswith('N') if tag else True
        except Exception:
            pass
        return True
    
    def _find_hierarchy(self, text_before: str) -> dict:
        """Find hierarchy (book/title/section) from text before article."""
        lines = text_before.split("\n")
        hierarchy = {"book": None, "title": None, "section": None}
        
        # Track which levels we've found (search from closest to farthest)
        found_book = False
        found_title = False
        found_section = False
        
        for line in reversed(lines):
            if found_book and found_title and found_section:
                break
            
            l = line.strip()
            
            # Section: #### header (level 4)
            if not found_section and l.startswith("#### "):
                if self._validate_header_pos(l[5:]):
                    hierarchy["section"] = re.sub(r"\*+", "", l[5:]).strip()
                    found_section = True
            
            # Title: ### header (level 3)
            elif not found_title and l.startswith("### ") and not l.startswith("####"):
                if self._validate_header_pos(l[4:]):
                    hierarchy["title"] = re.sub(r"\*+", "", l[4:]).strip()
                    found_title = True
            
            # Book: ## header (level 2)
            elif not found_book and l.startswith("## ") and not l.startswith("###"):
                if self._validate_header_pos(l[3:]):
                    hierarchy["book"] = re.sub(r"\*+", "", l[3:]).strip()
                    found_book = True
        
        return hierarchy
    
    def extract_all(self) -> dict:
        """Extract all articles with regex boundaries + CAMeL POS validation."""
        articles = []
        positions = []
        seen_positions = set()
        
        for pattern in ARTICLE_PATTERNS:
            for m in re.finditer(pattern, self.content, re.MULTILINE):
                pos = m.start()
                # Skip overlapping matches (within 5 chars of existing match)
                if any(abs(pos - p) < 5 for p in seen_positions):
                    continue
                if pos not in [p[0] for p in positions]:
                    positions.append((pos, m.group(0)))
                    seen_positions.add(pos)
        
        positions.sort(key=lambda x: x[0])
        
        for idx, (start, pat) in enumerate(positions):
            article_num = self._normalize_article_num(self.content[start:start + 30])
            end = positions[idx + 1][0] if idx + 1 < len(positions) else len(self.content)
            content = self._extract_content(start, end)
            hier = self._find_hierarchy(self.content[:start])
            
            articles.append({
                "article_number": article_num,
                "content": content,
                "page_num": self._find_page(start),
                "book": hier.get("book"),
                "title": hier.get("title"),
                "section": hier.get("section")
            })
        
        return {
            "code_name": self.code_name,
            "total_articles": len(articles),
            "articles": articles
        }
    
    def save(self, output_path: Optional[Path] = None):
        """Save extracted articles to v7_ful_json/ directory."""
        output_path = output_path or (V7_OUTPUT_DIR / f"{self.code_name}_full.json")
        data = self.extract_all()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {self.code_name}: {data['total_articles']} articles saved to {output_path}")


def main():
    """Test extraction on all MD files."""
    for md_path in MD_INPUT_DIR.glob("*.md"):
        pdf_file = str(PDF_INPUT_DIR / f"{md_path.stem}.pdf") if (PDF_INPUT_DIR / f"{md_path.stem}.pdf").exists() else None
        print(f"Processing: {md_path.name}")
        ArticleExtractor(str(md_path), pdf_file).save()


if __name__ == "__main__":
    main()
