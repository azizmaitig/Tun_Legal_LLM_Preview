# JORT Parser v7.1 - LLM Legal Assistant (Experimental)

**Status:** 🚧 Experimental - DO NOT use in production yet

## Overview

v7.1 adds a GPT-style Q&A interface for Tunisian law, powered by:
- **Hybrid Search**: AraBERT (dense) + BM25 (sparse) + RRF fusion
- **LLM Grounding**: Answers based on retrieved legal articles
- **Conversation Memory**: Supports follow-up questions

This is a separate directory from v7 - both can coexist without conflicts.

## Features

### LLM Legal Assistant
- Natural language Q&A in Arabic (e.g., "ما هي عقوبة العنف السياسي؟")
- Answers grounded in your Qdrant database (1154 articles)
- Citations to specific articles and legal codes
- Conversation history for follow-up questions

### Technical Stack
- **Backend**: FastAPI (`qa_api.py`)
- **Frontend**: Vanilla JS chat interface (`qa.html`)
- **LLM Providers**: HuggingFace (primary) + Groq (fallback)
- **Vector DB**: Qdrant Cloud (`V7collection_hybrid`)

## Quick Start

### 1. Prerequisites
```bash
# Ensure you have the API keys in v7_1/.env
# (Copied from v7, already has HF_TOKEN, GROQ_API_KEY, OPENROUTER_API_KEY)
```

### 2. Start the Q&A API Server
```bash
cd "/mnt/d/projects/obsidian second brain/10-Projects/11-Active/jort/parser/src/jort_parser/v7_1"
python3 qa_api.py
```

Server runs on: `http://localhost:8001`
API docs: `http://localhost:8001/docs`

### 3. Open the Chat Interface
```bash
# From src/ directory
cd "/mnt/d/projects/obsidian second brain/10-Projects/11-Active/jort/parser/src"
python3 -m http.server 8080
```

Visit: `http://localhost:8080/jort_parser/v7_1/qa.html`

## API Endpoints

### POST /qa
Q&A with conversation support

**Request:**
```json
{
  "q": "ما هي عقوبة العنف السياسي؟",
  "session_id": "optional-uuid",
  "limit": 5
}
```

**Response:**
```json
{
  "answer": "العنف السياسي ليس له تعريف واحد صريح...",
  "sources": [
    {
      "code_name": "المجلة الجزائية",
      "article_number": "218",
      "content": "إن الفصل 218 ينص على...",
      "score": 0.8523
    }
  ],
  "session_id": "uuid-123"
}
```

### GET /qa/health
Check LLM availability

### DELETE /qa/sessions/{session_id}
Clear conversation history

## File Structure

```
v7_1/
├── qa_api.py              # FastAPI server (NEW)
├── qa.html                # Chat interface (NEW)
├── README.md              # This file (NEW)
├── .env                   # API keys (copied from v7)
├── config.py              # Config (copied from v7)
├── utils.py               # Utilities (copied from v7)
├── llm/                   # LLM providers (copied from v7)
│   ├── __init__.py
│   ├── base.py
│   ├── hf_provider.py
│   ├── groq_provider.py
│   └── openrouter_provider.py
└── (other v7 files...)   # Not used by v7.1 Q&A
```

## How It Works

```
User Question (qa.html)
    ↓
POST /qa (qa_api.py)
    ↓
1. Hybrid Search in Qdrant
   (AraBERT + BM25 + RRF)
    ↓
2. Top 5 articles as context
    ↓
3. LLM Call (HF/Groq/nividia)
   + conversation history
    ↓
4. Grounded Answer + Sources
    ↓
Display in Chat Interface
```

## Example Conversation

**User:** ما هي عقوبة العنف السياسي؟

**Assistant:** العنف السياسي ليس له تعريف واحد صريح في نص قانوني مستقل، لكنه يُعاقب عليه حسب الأفعال المرتكبة وفق المجلة الجزائية.

📌 أمثلة من النصوص القانونية:
- الفصل 218 من المجلة الجزائية: يعاقب بالسجن كل من يتعمد الاعتداء بالعنف
- الفصل 222: يعاقب على التهديد بالعنف

**User:** هل يمكن أن تصل العقوبة إلى السجن؟

**Assistant:** نعم، يمكن أن تصل العقوبة إلى السجن وفق الفصل 218...

<img width="1569" height="1071" alt="legal_llm_mobile" src="https://github.com/user-attachments/assets/75228c83-a6b4-4e1c-aa82-e60996446658" />


## Configuration

Edit `config.py` to adjust:
- `QA_MAX_CONTEXT_ARTICLES`: Number of search results to use (default: 5)
- `QA_MAX_TOKENS`: Max tokens in LLM response (default: 1000)
- `QA_TEMPERATURE`: LLM temperature (default: 0.1 for accuracy)
- `QA_MAX_HISTORY`: Conversation history length (default: 10 messages)

## Testing

### Test API Health
```bash
curl http://localhost:8001/qa/health
```

### Test Q&A
```bash
curl -X POST http://localhost:8001/qa \
  -H "Content-Type: application/json" \
  -d '{"q": "ما هي عقوبة السرقة؟", "limit": 3}'
```

## Known Issues / TODO

- [ ] Add score_threshold parameter to /qa endpoint
- [ ] Add "Did you mean?" suggestions for misspelled Arabic queries
- [ ] Add streaming responses for better UX
- [ ] Add feedback mechanism (thumbs up/down)
- [ ] Deploy to production (cloud hosting)

## Differences from v7

| Feature | v7 | v7.1 |
|---------|----|------|
| Article Extraction | ✅ | ❌ (not needed) |
| LLM Enhancement | ✅ | ❌ (not needed) |
| Hybrid Search API | ✅ (search_api.py) | ✅ (reused) |
| **LLM Q&A** | ❌ | ✅ (qa_api.py) |
| **Chat Interface** | ❌ | ✅ (qa.html) |

## License

Same as v7 (MIT)

---

**Created:** 2026-04-29  
**Based on:** JORT Parser v7  
**Status:** Experimental - Testing Phase
