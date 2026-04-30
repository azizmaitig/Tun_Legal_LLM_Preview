#!/usr/bin/env python3
"""
v7.1 Q&A API - LLM-powered Legal Assistant for Tunisian Law
Provides GPT-style Q&A over legal articles using hybrid search + LLM grounding.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import sys
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from v7_1 directory
V7_1_PATH = Path(__file__).parent
load_dotenv(dotenv_path=V7_1_PATH / '.env')

# Add parent directories to path for imports
sys.path.insert(0, str(V7_1_PATH.parent.parent))  # src/
sys.path.insert(0, str(V7_1_PATH))  # v7_1/

# Import from search_api for hybrid search logic
from search_api import (
    get_qdrant_client, get_model, get_bm25_vectorizer,
    embed_text, models
)

# v7.1 collection (NEW - only use this)
COLLECTION_NAME = "v7.1"
MODEL_NAME = "aubmindlab/bert-base-arabertv2"  # AraBERT for Arabic

# Import LLM providers from v7_1
from jort_parser.v7_1.llm.hf_provider import HFProvider
from jort_parser.v7_1.llm.groq_provider import GroqProvider

app = FastAPI(title="JORT Legal Q&A API - v7.1")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
_qa_sessions: Dict[str, List[Dict]] = {}
_qa_llm = None


# ===============
# LLM Initialization
# ===============

def get_qa_llm():
    """Initialize LLM provider (OpenRouter primary, HF/Groq fallback)."""
    global _qa_llm
    if _qa_llm is not None:
        return _qa_llm
    
    # Try OpenRouter first (user requested)
    try:
        from jort_parser.v7_1.llm.openrouter_provider import OpenRouterProvider
        openrouter = OpenRouterProvider()
        if openrouter.available:
            _qa_llm = openrouter
            print("✓ Using OpenRouter LLM (primary)")
            return _qa_llm
    except Exception as e:
        print(f"OpenRouter init failed: {e}")
    
    # Fallback to HF
    try:
        hf = HFProvider()
        if hf.available:
            _qa_llm = hf
            print("✓ Using HuggingFace LLM (fallback)")
            return _qa_llm
    except Exception as e:
        print(f"HF init failed: {e}")
    
    # Fallback to Groq
    try:
        groq = GroqProvider()
        if groq.available:
            _qa_llm = groq
            print("✓ Using Groq LLM (fallback)")
            return _qa_llm
    except Exception as e:
        print(f"Groq init failed: {e}")
    
    return None


# ===============
# Q&A Prompt Template
# ===============

QA_SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في القانون التونسي.

منهجية الإجابة:
1. حدد المفهوم القانوني للسؤال
2. اربط السؤال بالفصول المناسبة
3. اشرح القاعدة القانونية
4. وضّح الشروط والاستثناءات إن وجدت
5. قدّم خلاصة واضحة

القواعد:
- استخدم فقط النصوص المقدمة
- اذكر الفصل واسم المجلة
- إذا كانت الإجابة غير مباشرة، فسّر العلاقة
- إذا كانت المعلومات ناقصة، وضّح ذلك"""

def format_context(articles: List[Dict], max_chars: int = 800) -> str:
    """Format search results as context for LLM."""
    context_parts = []
    for i, art in enumerate(articles, 1):
        code = art.get('code_name', 'غير معروف')
        num = art.get('article_number', '؟')
        content = art.get('content', '')[:max_chars]
        context_parts.append(f"[{i}] {code} - الفصل {num}:\n{content}\n")
    return "\n".join(context_parts)


def build_qa_messages(question: str, context: str, history: List[Dict]) -> List[Dict]:
    """Build messages array for LLM with conversation history."""
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"النصوص القانونية المتاحة:\n{context}"},
        {"role": "assistant", "content": "فهمت، سأجيب بناءً على هذه النصوص فقط."}
    ]
    
    # Add conversation history (last 6 messages to avoid token limits)
    for msg in history[-6:]:
        messages.append(msg)
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    return messages


# ===============
# Pydantic Models
# ===============

class QARequest(BaseModel):
    q: str
    session_id: Optional[str] = None
    limit: int = Query(5, description="Number of search results to use as context")


class QAResponse(BaseModel):
    answer: str
    sources: List[Dict]
    session_id: str


# ===============
# API Endpoints
# ===============

@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QARequest):
    """LLM-powered Q&A over legal articles with conversation support."""
    
    # 1. Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in _qa_sessions:
        _qa_sessions[session_id] = []
    history = _qa_sessions[session_id]
    
    # 2. Perform hybrid search
    tokenizer, model, device = get_model()
    bm25_vectorizer = get_bm25_vectorizer()
    client = get_qdrant_client()
    
    # Embed query - dense (AraBERT)
    query_dense = embed_text([request.q], tokenizer, model, device)[0]
    
    # Embed query - sparse (BM25/TF-IDF)
    query_sparse = bm25_vectorizer.transform([request.q])
    cx = query_sparse.tocoo()
    query_sparse_vec = models.SparseVector(
        indices=cx.col.tolist(),
        values=cx.data.tolist()
    )
    
    # Hybrid search with RRF fusion
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense.tolist(), using="dense", limit=50),
            models.Prefetch(query=query_sparse_vec, using="bm25", limit=50)
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=request.limit
    )
    
    # Format sources (include new fields from v7.1 data)
    sources = []
    for r in results.points:
        p = r.payload
        sources.append({
            "chunk_id": r.id,
            "article_id": p.get('article_id', ''),
            "code_name": p.get('code_name', ''),
            "article_number": p.get('article_number', ''),
            "content": p.get('content', '')[:300],
            "score": round(r.score, 4),
            "legal_features": p.get('legal_features', {}),
            "relations": p.get('relations', {}),
            "language": p.get('language', 'ar')
        })
    
    # 3. Format context for LLM
    context = format_context(sources)
    
    # 4. Call LLM
    answer = "عذراً، لا يمكنني الوصول إلى نموذج اللغة حالياً."
    llm = get_qa_llm()
    
    if llm:
        messages = build_qa_messages(request.q, context, history)
        
        try:
            # OpenRouter and HF both use requests (api_url + headers)
            if hasattr(llm, 'api_url'):
                import requests
                payload = {
                    "model": llm.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
                resp = requests.post(
                    llm.api_url,
                    headers=llm.headers,
                    json=payload,
                    timeout=30
                )
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    answer = f"خطأ في الاتصال بنموذج اللغة: {resp.status_code}"
            
            elif hasattr(llm, 'client'):  # Groq
                chat = llm.client.chat.completions.create(
                    model=llm.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                answer = chat.choices[0].message.content.strip()
            
        except Exception as e:
            answer = f"حدث خطأ أثناء معالجة السؤال: {str(e)[:100]}"
    
    # 5. Update conversation history
    history.append({"role": "user", "content": request.q})
    history.append({"role": "assistant", "content": answer})
    
    # Keep only last 10 messages
    if len(history) > 10:
        history = history[-10:]
    _qa_sessions[session_id] = history
    
    # 6. Return response
    return QAResponse(answer=answer, sources=sources, session_id=session_id)


@app.get("/qa/health")
async def qa_health():
    """Check LLM availability."""
    llm = get_qa_llm()
    return {
        "status": "ok" if llm else "no_llm",
        "llm_provider": llm.__class__.__name__ if llm else None,
        "collection": COLLECTION_NAME
    }


@app.delete("/qa/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _qa_sessions:
        del _qa_sessions[session_id]
        return {"status": "cleared"}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn
    print("Starting JORT Legal Q&A API (v7.1)...")
    print("API docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
