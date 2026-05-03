Status: 🚧 Experimental 


# JORT Legal Q&A v7.1

An Arabic-language AI assistant that answers questions about Tunisian law. Users ask questions in Arabic → the system performs **hybrid semantic search** over 8,458 legal article chunks across 10+ legal codes → an LLM synthesizes a grounded answer with cited sources, streamed in real-time.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend (Browser)                    │
│  Single-page app — RTL, dark theme, Arabic UI            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Sources  │  │   Chat   │  │ History  │               │
│  │ (left)   │  │ (center) │  │ (right)  │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP + SSE (text/event-stream)
┌────────────────────────▼─────────────────────────────────┐
│                   Backend (FastAPI :8002)                 │
│  JWT Auth → Rate Limit → Cache → Hybrid Search → LLM    │
└──┬──────────┬──────────┬──────────┬──────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐  ┌───────┐  ┌───────┐  ┌───────┐
│Qdrant│  │NVIDIA │  │MongoDB│  │Tavily │
│Cloud │  │NIM API│  │Atlas  │  │Search │
└──────┘  └───────┘  └───────┘  └───────┘
```

---

## Query-to-Answer Flow (13 Steps)
<img width="1793" height="881" alt="image" src="https://github.com/user-attachments/assets/e9080af8-459f-4b3a-96d8-a33433707e3c" />
1.Active LLM and  Web search provider    2. Chat Panel :User input field     3.legal sources: Legal code name ,Article number ,Relevance score
4. History Panel: Lists previous sessions ,“New Session”     5.Structured legal response: Definition,Explanation,Summary    6.user query 




```
User: "ما المقصود بالتشريع والترتيب المنظم؟"
```
### Step 1: Input Validation
```
Frontend → POST /qa/stream
{
  "q": "ما المقصود بالتشريع والترتيب المنظم؟",
  "limit": 15,
  "session_id": "usr_abc123:sess-xyz"
}

Backend validates:
├─ Length: 3–500 characters ✓
├─ Not empty ✓
└─ Auth: JWT token decoded → user_id extracted ✓
```

### Step 2: Rate Limiting
```
In-memory sliding window check:
├─ Session "usr_abc123:sess-xyz" → 2 requests in last 60s
├─ Limit: 10 req/min → ALLOWED ✓
└─ Record request timestamp
```

### Step 3: Cache Check
```
In-memory cache lookup (1-hour TTL):
├─ Key: "ما المقصود بالتشريع والترتيب المنظم؟:15"
├─ No cached result → proceed to search
└─ (If cached: skip to Step 8, serve from cache)
```

### Step 4: Session Retrieval
```
MongoDB query:
  db.qa_sessions.find({"user_id": "usr_abc123"})

Result:
  messages: [
    {"role": "user", "content": "ما هي شروط الطلاق في القانون التونسي؟"},
    {"role": "assistant", "content": "يقع الطلاق وفقاً للفصل 30..."}
  ]
```

### Step 5: 3-Branch Hybrid Search (Qdrant)

The query is embedded three ways in parallel:

**Branch A — Dense (Semantic)**
```
Query: "ما المقصود بالتشريع والترتيب المنظم؟"
    ↓ AraBERT (local, 768-dim, mean-pooled)
Dense vector: [0.12, -0.05, 0.89, ..., 0.34]
    ↓ Qdrant "dense" vector field (cosine distance)
Top 100 results by semantic similarity
```

**Branch B — Sparse (Keyword/BM25)**
```
Query: "ما المقصود بالتشريع والترتيب المنظم؟"
    ↓ TF-IDF vectorizer (vocab: ~10,000 terms)
Sparse vector: {indices: [42, 156, 891], values: [2.1, 1.8, 3.2]}
    ↓ Qdrant "bm25" sparse vector field (IDF modifier)
Top 100 results by keyword match
```

**Branch C — Keyword (Arabic keyword extraction)**
```
Query: "ما المقصود بالتشريع والترتيب المنظم؟"
    ↓ extract_arabic_keywords()
    ├─ Remove stop words: ما, المقصود, بـ, في
    └─ Extract: ["تشريع", "ترتيب", "منظم"]
    ↓ Qdrant KeywordQuery with MatchText on "content" field
Top 50 results by keyword presence
```

**Fusion — Reciprocal Rank Fusion (RRF)**
```
All 3 ranked lists combined:
  RRF_score = Σ (1 / (60 + rank_i))

Each result gets a fused score combining
its position across all 3 search methods
```

**Post-Reranking**
```
For each result:
  Jaccard(query_keywords, result_keywords)
  Final_score = (RRF_score × 0.7) + (Jaccard × 0.3)

Top 15 results selected as context
```

### Step 6: Sources Emitted (SSE Event #1)
```
Backend immediately sends to frontend:

event: sources
data: {
  "sources": [
    {
      "chunk_id": 7201,
      "code_name": "مجلة تقديم الخدمات المالية لغير المقيمين",
      "article_number": "7",
      "content": "يشير مصطلح التشريع والترتيب المنظم إلى...",
      "score": 0.91,
      "article_id": "code-financial-services__7"
    },
    { ...14 more sources... }
  ],
  "session_id": "sess-xyz"
}

Frontend renders source cards in left panel in real-time
```

### Step 7: LLM Generation (Streaming)
```
Messages sent to NVIDIA API:

System: "أنت مساعد قانوني متخصص في القانون التونسي..."
User:   "النصوص القانونية المتاحة:\n[1] مجلة تقديم الخدمات المالية لغير المقيمين - الفصل 7: يشير مصطلح التشريع...\n[2] ..."
Assistant: "فهمت."
User:   (conversation history from MongoDB)
User:   "ما المقصود بالتشريع والترتيب المنظم؟"

NVIDIA streams response token by token:
```

Each token is sent to the frontend via SSE:

```
event: token
data: {"token": "الم"}

event: token
data: {"token": "فهوم"}

event: token
data: {"token": " "}

event: token
data: {"token": "الق"}

event: token
data: {"token": "انون"}
...
```

Frontend appends tokens in real-time → user sees the answer being typed live.

### Step 8: Web Search Fallback (Conditional)
```
If LLM response contains failure indicators:
  "لا توجد معلومات كافية" / "غير متوفرة في النصوص"

Then:
  1. Send SSE "reset" event → frontend clears partial text
  2. Query Tavily: "التشريع والترتيب المنظم Tunisia القانون التونسي"
  3. Build QA_SYNTHESIS_PROMPT with local results + web results
  4. Second LLM call → new streaming response

If answer is good → skip web search entirely
```

### Step 9: Source Filtering
```
Full answer received:
  "يشير مصطلح التشريع والترتيب المنظم إلى مجموعة القوانين... [1]"

filter_used_sources():
  ├─ Parse citation markers: [1], [2], [W1]
  ├─ Match [1] → source #1 (مجلة تقديم الخدمات المالية, الفصل 7)
  ├─ Match keyword overlap for uncited sources
  └─ Return: only sources actually referenced in answer

This prevents showing irrelevant sources that were
searched but not used in the final answer
```

### Step 10: Sources Re-emitted (SSE Event #2)
```
event: sources
data: {
  "sources": [
    { "code_name": "مجلة تقديم الخدمات المالية لغير المقيمين", "article_number": "7", ... }
  ],
  "session_id": "sess-xyz"
}

Frontend updates source panel to show ONLY used sources
```

### Step 11: Cache Response
```
In-memory cache store:
  Key: "ما المقصود بالتشريع والترتيب المنظم؟:15"
  Value: {
    "answer": "المفهوم القانوني: التشريع والترتيب المنظم...",
    "sources": [...],
    "created_at": <timestamp>,
    "ttl": 3600  // 1 hour
  }
```

### Step 12: Save Session
```
MongoDB upsert:
  db.qa_sessions.update_one(
    {"user_id": "usr_abc123"},
    {"$push": {"messages": {
      "role": "assistant",
      "content": "المفهوم القانوني: التشريع والترتيب المنظم. يشير مصطلح...",
      "timestamp": "2026-05-03T12:00:00Z"
    }}}
  )

  Keep last 10 messages max
```

### Step 13: Done (SSE Event #3)
```
event: done
data: {"session_id": "sess-xyz"}

Frontend:
  ├─ Finalize answer display
  ├─ Show copy button + feedback stars
  ├─ Update sidebar with new session
  └─ Re-enable input
```

### Full SSE Event Timeline
```
Time  Event      What User Sees
────  ─────────  ─────────────────────────────────────────
0s    sources    Source cards appear in left panel
0.5s  token      "ي" appears in answer bubble
0.6s  token      "يقع" appears
0.7s  token      "يقع " appears
...   token      Answer streams word by word in real-time
5s    done       Answer finalized, copy + feedback shown
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **Vector DB** | Qdrant Cloud (AWS eu-central-1), collection `v7.1` |
| **Embeddings** | AraBERT `aubmindlab/bert-base-arabertv2` (local, 768-dim) |
| **Sparse** | scikit-learn TF-IDF (BM25, ~10K vocab) |
| **LLM** | NVIDIA NIM — `meta/llama-4-maverick-17b-128e-instruct` |
| **Auth** | PyJWT (HS256, 72h) + bcrypt |
| **Sessions** | MongoDB Atlas (Motor async driver) |
| **Web Search** | Tavily API |
| **Arabic NLP** | PyArabic (normalization, numeral conversion) |
| **Frontend** | Vanilla JS, marked.js, Cairo/Amiri fonts, RTL, PWA |

---

## File Structure

```
v7_1/final/
├── qa_api.py (1,657 lines)           # Main FastAPI app
├── search_api.py (220 lines)          # Standalone search API
├── config.py                          # Prompts, model names, patterns
├── web_search.py                      # Tavily web search module
├── utils.py                           # Arabic normalization, JSON repair
├── create_user.py                     # Admin CLI for user management
├── .env                               # API keys
├── requirements.txt                   # Dependencies
├── Procfile                           # Deployment config
├── llm/
│   ├── base.py                        # Retry/backoff logic
│   └── nvidia_provider.py             # NVIDIA Chat API
├── static/
│   ├── qa.html (3,115 lines)          # Single-file SPA frontend
│   ├── sw.js                          # Service worker
│   ├── manifest.json                  # PWA manifest
│   ├── marked.min.js                  # Markdown parser
│   └── icon-192.png / icon-512.png    # PWA icons
└── input/new_data/
    └── chunks.json (~168K lines)      # Pre-chunked legal data
```

---

## Auth System

```
Admin: python create_user.py "aziz123" "Aziz"
  ↓
MongoDB users collection:
  {
    "user_id": "usr_550e8400...",
    "passcode_hash": "$2b$12$...",    # bcrypt
    "name": "Aziz",
    "is_active": true
  }
  ↓
User enters passcode → POST /auth/login
  ↓
Backend: bcrypt.checkpw(passcode, stored_hash)
  ↓
JWT token (HS256, 72h expiry, payload: {user_id, iat, exp})
  ↓
Frontend: localStorage.setItem("jort_token", token)
  ↓
All requests: Authorization: Bearer <token>
  ↓
Backend: get_current_user() decodes JWT → extracts user_id
  ↓
All session queries: {"user_id": user_id} filter
```

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/login` | ✅ | Login → JWT token |
| `POST` | `/qa/stream` | ✅ | Streaming Q&A (SSE) |
| `POST` | `/qa` | ✅ | Non-streaming Q&A |
| `GET` | `/qa/article` | ✅ | Full article by code + number |
| `GET` | `/qa/sessions` | ✅ | List user sessions |
| `GET` | `/qa/session/{id}` | ✅ | Get session messages |
| `DELETE` | `/qa/sessions/{id}` | ✅ | Delete session |
| `POST` | `/qa/web_search` | ✅ | Standalone web search |
| `GET` | `/qa/health` | ✅ | Health check |
| `POST` | `/qa/feedback` | ❌ | Feedback (1–5 stars) |
| `GET` | `/qa/feedback/stats` | ❌ | Feedback analytics |
| `GET` | `/` | ✅ | Serve frontend |

---

## Data: Legal Codes Indexed

| Code | English |
|---|---|
| المجلة الجزائية | Penal Code |
| مجلة الأحوال الشخصية | Personal Status Code |
| المجلة التجارية | Commercial Code |
| مجلة الالتزامات والعقود | Code of Obligations and Contracts |
| مجلة الشركات التجارية | Commercial Companies Code |
| مجلة الإجراءات المدنية والتجارية | Civil and Commercial Procedure |
| مجلة التأمين | Insurance Code |
| مجلة التحكيم | Arbitration Code |
| مجلة الشغل البحري | Maritime Labor Code |
| مجلة حماية الطفل | Child Protection Code |
| القانون الدولي الخاص | Private International Law |
...
**Total: 8,458 chunks** across 10+ legal codes

---

## Deployment

| Aspect | Details |
|---|---|
| **Hosting** | Self-hosted on local PC (WSL) |
| **Server** | `uvicorn qa_api:app --host 0.0.0.0 --port 8002` |
| **Public access** | Cloudflare Tunnel (temporary URL) |
| **Budget** | $0 (all external services on free tiers) |
| **PaaS ready** | `Procfile` + `runtime.txt` for Render/Railway |
