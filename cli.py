#!/usr/bin/env python3
"""v7 CLI - Single entry point for JORT Parser v7."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from jort_parser.v7.extractor import ArticleExtractor
    from jort_parser.v7.llm.processor import LLMProcessor
    from jort_parser.v7.config import (
        MD_INPUT_DIR, PDF_INPUT_DIR, V7_OUTPUT_DIR,
        PROGRESS_DIR, HF_TOKEN, GROQ_API_KEY
    )
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory.")
    sys.exit(1)


def extract_command(args):
    """Run article extraction."""
    if args.file:
        # Process single file
        md_path = Path(args.file)
        if not md_path.exists():
            print(f"File not found: {md_path}")
            return
        
        pdf_file = None
        if args.pdf:
            pdf_file = args.pdf
        else:
            pdf_path = PDF_INPUT_DIR / f"{md_path.stem}.pdf"
            pdf_file = str(pdf_path) if pdf_path.exists() else None
        
        print(f"Extracting from: {md_path.name}")
        extractor = ArticleExtractor(str(md_path), pdf_file)
        extractor.save()
    else:
        # Process all files in input directory
        print(f"Processing all MD files in: {MD_INPUT_DIR}")
        for md_path in MD_INPUT_DIR.glob("*.md"):
            pdf_file = None
            pdf_path = PDF_INPUT_DIR / f"{md_path.stem}.pdf"
            if pdf_path.exists():
                pdf_file = str(pdf_path)
            
            print(f"\nProcessing: {md_path.name}")
            try:
                extractor = ArticleExtractor(str(md_path), pdf_file)
                extractor.save()
            except Exception as e:
                print(f"Error: {e}")


def enhance_command(args):
    """Run LLM enhancement."""
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    output_file = Path(args.output) if args.output else None
    
    print(f"Enhancing: {input_file.name}")
    print(f"HF Token: {'set' if HF_TOKEN else 'NOT SET'}")
    print(f"Groq Key: {'set' if GROQ_API_KEY else 'NOT SET'}")
    
    processor = LLMProcessor(
        hf_token=args.hf_token or HF_TOKEN,
        groq_key=args.groq_key or GROQ_API_KEY
    )
    
    try:
        result = processor.process_file(
            input_file=input_file,
            output_file=output_file,
            start_idx=args.start,
            end_idx=args.end,
            resume=not args.no_resume
        )
        print(f"\nDone! Enhanced {result.get('total_articles', 0)} articles.")
    except Exception as e:
        print(f"Error during enhancement: {e}")


def resume_command(args):
    """Resume interrupted processing."""
    code_name = args.code_name
    progress_file = PROGRESS_DIR / f"{code_name}_progress.json"
    
    if not progress_file.exists():
        print(f"No progress file found for: {code_name}")
        return
    
    print(f"Resuming processing for: {code_name}")
    # This will be handled by enhance with resume=True
    input_file = V7_OUTPUT_DIR / f"{code_name}_full.json"
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    enhance_command(argparse.Namespace(
        input=str(input_file),
        output=args.output,
        hf_token=args.hf_token,
        groq_key=args.groq_key,
        start=0,
        end=None,
        no_resume=False
    ))


def main():
    parser = argparse.ArgumentParser(
        description="JORT Parser v7 - Arabic Legal Code Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract articles from all MD files
  python cli.py extract
  
  # Extract from single file
  python cli.py extract --file "مجلة الإجراءات الجزائية.md"
  
  # Enhance articles with LLM
  python cli.py enhance --input "v7_ful_json/مجلة الإجراءات الجزائية_full.json"
  
  # Enhance specific article range
  python cli.py enhance --input "..." --start 50 --end 112
  
  # Resume interrupted processing
  python cli.py resume --code-name "مجلة الإجراءات الجزائية"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract articles from MD files")
    extract_parser.add_argument("--file", help="Process single MD file instead of all files")
    extract_parser.add_argument("--pdf", help="PDF file to use for page mapping")
    
    # Enhance command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance articles with LLM")
    enhance_parser.add_argument("--input", required=True, help="Input JSON file")
    enhance_parser.add_argument("--output", help="Output JSON file (default: auto-generated)")
    enhance_parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    enhance_parser.add_argument("--end", type=int, help="End index (default: all)")
    enhance_parser.add_argument("--hf-token", help="HuggingFace API token (overrides env var)")
    enhance_parser.add_argument("--groq-key", help="Groq API key (overrides env var)")
    enhance_parser.add_argument("--no-resume", action="store_true", help="Don't resume from progress")
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume interrupted processing")
    resume_parser.add_argument("--code-name", required=True, help="Code name (e.g., 'مجلة الإجراءات الجزائية')")
    resume_parser.add_argument("--output", help="Output JSON file")
    resume_parser.add_argument("--hf-token", help="HuggingFace API token")
    resume_parser.add_argument("--groq-key", help="Groq API key")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "extract":
        extract_command(args)
    elif args.command == "enhance":
        enhance_command(args)
    elif args.command == "resume":
        resume_command(args)


if __name__ == "__main__":
    main()
