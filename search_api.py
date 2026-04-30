#!/usr/bin/env python3
"""
FastAPI wrapper for Qdrant V7collection search
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import models, Filter, FieldCondition, MatchValue
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Load .env
load_dotenv(Path(__file__).parent / '.env')

app = FastAPI(title="JORT Legal Search API")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients (lazy loaded)
_qdrant_client = None
_model = None
_tokenizer = None
_device = None
_bm25_vectorizer = None

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
COLLECTION_NAME = "V7collection_hybrid"  # Use hybrid collection
QDRANT_URL = os.getenv('QDRANT_URL', '')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            check_compatibility=False
        )
    return _qdrant_client

def get_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
    return _tokenizer, _model, _device

def get_bm25_vectorizer():
    global _bm25_vectorizer
    if _bm25_vectorizer is None:
        # Load articles to fit the vectorizer
        from pathlib import Path
        import json
        
        INPUT_DIR = Path("/mnt/d/projects/obsidian/second brain/10-Projects/11-Active/jort/parser/src/jort_parser/v7/output/complete_llm")
        INPUT_FILES = [
            "المجلة الجزائية_v7_complete.json",
            "المجلة التجارية_v7_complete_fixed.json",
            "مجلة الأحوال الشخصية_v7_complete_fixed.json",
        ]
        
        all_contents = []
        for filename in INPUT_FILES:
            filepath = INPUT_DIR / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    articles = data.get('articles', [])
                    all_contents.extend([art.get('content', '') for art in articles])
        
        _bm25_vectorizer = TfidfVectorizer(max_features=10000)
        _bm25_vectorizer.fit(all_contents)
    return _bm25_vectorizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(texts, tokenizer, model, device):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        model_output = model(**encoded)
    embeddings = mean_pooling(model_output, encoded['attention_mask'])
    return embeddings.cpu().numpy()

from typing import Optional

class SearchResult(BaseModel):
    code_name: str
    article_number: str
    content: str
    book: Optional[str] = None
    title: Optional[str] = None
    llm_legal_action: Optional[str] = None
    llm_crime_type: Optional[str] = None
    llm_banking_related: bool = False
    score: float

class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results"),
    code_name: str = Query(None, description="Filter by code name")
):
    """Hybrid search in V7collection_hybrid (dense + sparse)"""
    tokenizer, model, device = get_model()
    bm25_vectorizer = get_bm25_vectorizer()
    client = get_qdrant_client()

    # Embed query - dense (AraBERT)
    query_dense = embed_text([q], tokenizer, model, device)[0]

    # Embed query - sparse (BM25/TF-IDF)
    query_sparse = bm25_vectorizer.transform([q])
    cx = query_sparse.tocoo()
    query_sparse_vec = models.SparseVector(
        indices=cx.col.tolist(),
        values=cx.data.tolist()
    )

    # Build filter if needed
    query_filter = None
    if code_name:
        print(f"DEBUG: Filtering by code_name='{code_name}'")
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="code_name",
                    match=MatchValue(value=code_name)
                )
            ]
        )

    # Hybrid search with RRF fusion
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_dense.tolist(),
                using="dense",
                limit=50
            ),
            models.Prefetch(
                query=query_sparse_vec,
                using="bm25",
                limit=50
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        query_filter=query_filter
    )

    # Format results
    formatted_results = []
    for r in results.points:
        p = r.payload
        formatted_results.append(SearchResult(
            code_name=p.get('code_name', ''),
            article_number=p.get('article_number', ''),
            content=p.get('content', '')[:300] + '...' if len(p.get('content', '')) > 300 else p.get('content', ''),
            book=p.get('book'),
            title=p.get('title'),
            llm_legal_action=p.get('llm_legal_action'),
            llm_crime_type=p.get('llm_crime_type'),
            llm_banking_related=p.get('llm_banking_related', False),
            score=round(r.score, 4)
        ))

    # Sort by article_number (natural sorting)
    def try_int(x):
        try:
            # Handle formats like "14-3", "54", etc.
            parts = str(x.article_number).split('-')
            return tuple(int(p) for p in parts)
        except:
            return (9999,)  # Put unparseable at end

    formatted_results.sort(key=try_int)

    return SearchResponse(query=q, total=len(formatted_results), results=formatted_results)

@app.get("/codes")
async def get_codes():
    """Get list of available legal codes"""
    client = get_qdrant_client()
    try:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        codes = set()
        for point in result[0]:
            if 'code_name' in point.payload:
                codes.add(point.payload['code_name'])
        return {"codes": sorted(list(codes))}
    except:
        return {"codes": []}

@app.get("/health")
async def health():
    return {"status": "ok", "collection": COLLECTION_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
