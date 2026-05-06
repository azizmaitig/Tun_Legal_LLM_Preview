# JORT Parser - Hybrid Search Pipeline Architecture

## Overview

End-to-end pipeline for semantic search over Arabic legal articles using hybrid vector search (dense + sparse) with Qdrant.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  10-Projects/11-Active/jort/parser/src/jort_parser/v7/  │
│  ├── output/complete_llm/ (3 JSON files, 1154 articles)  │
│  │   ├── المجلة الجزائية_v7_complete.json (358 articles)    │
│  │   ├── المجلة التجارية_v7_complete_fixed.json (539 articles) │
│  │   └── مجلة الأحوال الشخصية_v7_complete_fixed.json (257) │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   EMBEDDING LAYER                            │
│                  embed_to_qdrant_hybrid.py                         │
├─────────────────────────────────────────────────────────────┤
│  1. LOAD MODELS                                                 │
│     • AraBERT: aubmindlab/bert-base-arabertv2 (dense)     │
│     • TF-IDF Vectorizer (sparse/BM25)                       │
│                                                                │
│  2. PROCESS ARTICLES (batches of 32)                         │
│     • Extract: article_number, content, metadata            │
│     • Dense embed: AraBERT → 768d vector                │
│     • Sparse embed: TF-IDF → indices + values           │
│                                                                │
│  3. UPSERT TO QDRANT                                        │
│     • Collection: V7collection_hybrid                         │
│     • Vectors: {"dense": 768d, "bm25": sparse}        │
│     • Payload: full content + all metadata fields          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    QDRANT CLOUD LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Collection: V7collection_hybrid                             │
│  URL: https://your-qdrant-instance.cloud.qdrant.io │
│                                                                │
│  Configuration:                                               │
│  ├── Dense vectors: 768d, Cosine distance                 │
│  ├── Sparse vectors: bm25, IDF modifier                    │
│  ├── Payload indexes:                                      │
│  │   ├── code_name (keyword)                             │
│  │   ├── article_number (keyword)                         │
│  │   ├── page_num (integer)                             │
│  │   ├── llm_banking_related (bool)                     │
│  │   ├── book (keyword)                                 │
│  │   └── llm_legal_action (keyword)                   │
│  └── Points: 1154 articles                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    QUERY LAYER (Hybrid Search)                    │
│                      search_api.py                                │
├─────────────────────────────────────────────────────────────┤
│  1. RECEIVE QUERY (FastAPI endpoint: /search)               │
│     • q: search text (Arabic)                                │
│     • limit: number of results (default: 10)                 │
│     • code_name: filter by legal code (optional)              │
│                                                                │
│  2. HYBRID QUERY (Qdrant FusionQuery + RRF)              │
│     • Dense branch: AraBERT embed query → 768d              │
│     • Sparse branch: TF-IDF embed query → BM25               │
│     • Fusion: Reciprocal Rank Fusion (RRF)                   │
│                                                                │
│  3. FILTERING (optional)                                       │
│     • Filter by code_name (exact match)                       │
│     • Uses Qdrant Filter + FieldCondition objects            │
│                                                                │
│  4. RESULTS PROCESSING                                        │
│     • Sort by article_number (natural sort)                  │
│     • Format: code_name, article_number, content, metadata   │
│     • Return: JSON with query, total, results             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                              │
│                       search.html                                 │
├─────────────────────────────────────────────────────────────┤
│  • Search box (Arabic text input)                           │
│  • Dropdown: filter by legal code (جميع القوانين)        │
│  • Dropdown: limit (5/10/20/50 results)                    │
│  • Results display:                                       │
│    ├── Arabic RTL layout                                 │
│    ├── Highlight search terms                               │
│    ├── Show metadata badges (book, action, type)         │
│    └── Score display (semantic similarity)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example

### Input Article (from JSON)
```json
{
  "article_number": "705",
  "content": "إن فتح الاعتماد يقتضي وضع وسائل للدفع...",
  "book": "الكتاب الخامس في العقود التجارية",
  "llm_legal_action": "إجراء",
  "llm_banking_related": true
}
```

### Stored in Qdrant
```
Point ID: 1001
├── Dense Vector: [0.1, 0.2, ..., 0.8] (768d - AraBERT)
├── Sparse Vector: {indices: [15, 42, 156], values: [0.5, 1.2, 0.8]} (BM25)
└── Payload:
    ├── code_name: "المجلة التجارية"
    ├── article_number: "705"
    ├── content: "إن فتح الاعتماد يقتضي..." (full text)
    ├── book: "الكتاب الخامس في العقود التجارية"
    ├── llm_legal_action: "إجراء"
    └── llm_banking_related: true
```

### Query: "السرقة"
```
1. AraBERT embed → dense_query (768d)
2. TF-IDF embed → sparse_query (indices + values)
3. Qdrant FusionQuery (RRF):
   ├── Dense prefetch: top 50 by semantic similarity
   ├── Sparse prefetch: top 50 by keyword match (BM25)
   └── RRF fusion → combined ranking
4. Filter (optional): code_name = "المجلة الجزائية"
5. Results: Top 10 articles sorted by article_number
```

---

## Component Summary

| Component | File | Purpose | Key Tech |
|-----------|------|---------|----------|
| **Embedding Script** | `embed_to_qdrant_hybrid.py` | Process articles → upsert to Qdrant | AraBERT, TF-IDF, Qdrant |
| **Search API** | `search_api.py` | FastAPI server for hybrid queries | FastAPI, Qdrant, RRF |
| **Frontend** | `search.html` | Web UI for search | Vanilla JS, Fetch API |
| **Index Creator** | `create_indexes.py` | Create payload indexes in Qdrant | Qdrant Python client |
| **Environment** | `.env` | API keys, URLs | dotenv |

---

## Search Modes

| Mode | Technique | Pros | Cons |
|------|-----------|------|------|
| **Dense Only** | AraBERT embeddings + Cosine similarity | Semantic understanding | Misses exact keyword matches |
| **Sparse Only** | BM25/TF-IDF keyword search | Exact keyword matching | No semantic understanding |
| **Hybrid (RRF)** | Dense + Sparse + Reciprocal Rank Fusion | Best of both worlds | Slightly slower, more complex |

---

## API Endpoints

### GET /health
```json
{"status": "ok", "collection": "V7collection_hybrid"}
```

### GET /codes
```json
{"codes": ["المجلة الجزائية", "المجلة التجارية", "مجلة الأحوال الشخصية"]}
```

### GET /search?q=السرقة&limit=10&code_name=المجلة الجزائية
```json
{
  "query": "السرقة",
  "total": 15,
  "results": [
    {
      "code_name": "المجلة الجزائية",
      "article_number": "705",
      "content": "إن فتح الاعتماد...",
      "book": "الكتاب الخامس",
      "llm_legal_action": "إجراء",
      "score": 0.8523
    }
  ]
}
```

---

## File Structure

```
10-Projects/11-Active/jort/parser/src/
├── .env                          # API keys, URLs
├── embed_to_qdrant_hybrid.py   # Hybrid embedding script
├── search_api.py                 # FastAPI server (hybrid search)
├── search.html                   # Frontend web UI
├── create_indexes.py             # Payload index creator
└── jort_parser/v7/
    ├── output/complete_llm/     # Input JSON files
    │   ├── المجلة الجزائية_v7_complete.json
    │   ├── المجلة التجارية_v7_complete_fixed.json
    │   └── مجلة الأحوال الشخصية_v7_complete_fixed.json
    └── ARCHITECTURE.md            # v7 extraction pipeline docs
```

---

## Key Configuration

### Qdrant Cloud
- **URL**: `https://your-qdrant-instance.cloud.qdrant.io:6333`
- **Collection**: `V7collection_hybrid`
- **API Key**: JWT format (from Qdrant Cloud console)

### Models
- **Dense**: `aubmindlab/bert-base-arabertv2` (768 dimensions)
- **Sparse**: TF-IDF (fitted on 1154 articles, vocab size: 10000)

### Search Parameters
- **Fusion method**: RRF (Reciprocal Rank Fusion)
- **Distance**: Cosine (for dense vectors)
- **Sparse modifier**: IDF (Inverse Document Frequency)
- **Default limit**: 10 results
- **Prefetch limit**: 50 per branch (dense + sparse)

---
