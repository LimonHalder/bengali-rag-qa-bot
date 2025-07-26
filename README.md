cat > README.md <<EOF
# Bengali RAG QA API

This project provides a Bengali question-answering API built with FastAPI using Retrieval-Augmented Generation (RAG). It integrates a vector database (ChromaDB), SBERT-based embeddings for Bengali, and Google Gemini via LangChain to generate answers from both MCQ and passage-based contexts.

---

## Features

- Retrieves relevant context from both MCQ and passage documents  
- Combines long-term memory (ChromaDB) and short-term history (chat memory)  
- Generates responses using Gemini 2.5 Flash via LangChain  
- API served with FastAPI and browsable via Swagger UI  
- Optional static frontend served via /conversation endpoint  

---

## API Endpoints

### POST /ask

Submit a user question and get a Bengali answer based on retrieved knowledge.

**Request JSON:**

```json
{
  "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
}
```

**Response JSON:**

```json
{
  "answer": "মামাকে"
}
```

---

## Project Structure

```
.
├── main.py                # FastAPI app with RAG logic
├── index.html             # Static frontend (optional)
├── chroma_db/             # ChromaDB persistent storage
└── README.md
```

---

## Getting Started

### 1. Clone and enter the project

```bash
git clone https://github.com/LimonHalder/bengali-rag-qa-bot.git
cd bengali-rag-qa-bot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Set up environment variable

> **Note:** For production, use proper environment variable management instead of hardcoding.

You can either set your API key inside main.py (not recommended for production):

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

Or export the environment variable in your shell:

```bash
export GOOGLE_API_KEY=your-api-key  # Linux/macOS
set GOOGLE_API_KEY=your-api-key     # Windows CMD
\$env:GOOGLE_API_KEY="your-api-key"  # PowerShell
```

---

## Run the Server

```bash
uvicorn main:app --reload
```

Open in your browser:

- Swagger Docs: http://127.0.0.1:8000/docs  
- Static frontend: http://127.0.0.1:8000/conversation

---

## Notes

- Embedding model: l3cube-pune/bengali-sentence-similarity-sbert  
- LLM: Google Gemini via LangChain  
- Vector DB: ChromaDB (local persistent)

---

## License

This project is for educational and research purposes only. All external models and APIs are subject to their respective licenses.
EOF
