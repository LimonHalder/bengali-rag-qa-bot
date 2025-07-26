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

## 🔌 API Documentation

### POST `/ask`

Submit a Bengali or English question and get a context-based answer from the RAG system.

- **Method:** `POST`  
- **Content-Type:** `application/json`  
- **Response:** `application/json`

#### ✅ Request Body

```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

| Field | Type   | Required | Description                        |
|-------|--------|----------|------------------------------------|
| query | string | ✅       | The question in Bengali or English |

#### 📦 Sample Responses

```json
{ "answer": "শম্ভুনাথ বাবু" }
```

```json
{ "answer": "শম্ভুনাথ সেন" }
```

```json
{ "answer": "মামাকে" }
```

#### 🔁 Example cURL Command

```bash
curl -X POST http://127.0.0.1:8000/ask \\
     -H "Content-Type: application/json" \\
     -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

#### 📘 Swagger UI (Interactive Docs)

Once the server is running:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Static frontend: [http://127.0.0.1:8000/conversation](http://127.0.0.1:8000/conversation)

---

## Project Structure

```
BENGALI-RAG-QA-BOT/
├── __pycache__/
├── _chroma_db/
├── embedding_pipeline/
│   ├── ingest.py
│   ├── llm_enhancer.py
│   └── ocr_pipeline.py
├── resource/
│   ├── data/
│   ├── output_extracted/
│   ├── process_data/
│   │   ├── section1.txt
│   │   └── section2.txt
│   └── HSC26-Bangla1st-Paper.pdf
├── venv/
├── .gitignore
├── index.html
├── main.py
├── rag_logic.py
├── README.md
└── requirements.txt
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

You can either set your API key inside main.py:

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

Or export the environment variable:

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

## Evaluation Metrics

We evaluate the RAG system using cosine similarity between embeddings from the **\`sentence-transformers/all-MiniLM-L6-v2\`** model:

- **Groundedness Score:** Similarity between the generated answer and retrieved context embeddings.
- **Relevance Score:** Similarity between the query and retrieved context embeddings.

### Code snippet

```python
from sentence_transformers import SentenceTransformer, util

eval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

answer_emb = eval_model.encode(answer, convert_to_tensor=True)
context_emb = eval_model.encode(" ".join(top_contexts), convert_to_tensor=True)
groundedness_score = util.cos_sim(answer_emb, context_emb).item()

query_emb = eval_model.encode(query, convert_to_tensor=True)
relevance_score = util.cos_sim(query_emb, context_emb).item()
```

---

## Q&A

### 1. What method or library did you use to extract the text?

- **PyMuPDF (fitz)** and **PyPDF2** were used to divide and extract text.
- **pytesseract (OCR)** was applied to handle complex or image-based regions (e.g., MCQ answer boxes).
- Text was stored in `passage_raw.txt` and `mcq_raw.txt`.

### 2. What chunking strategy did you use?

- For **passage-type text**: `chunk_size = 400`, `overlap = 200`.
- For **MCQ-type text**: `chunk_size = 100`, `overlap = 10`.

Why this works:
- Long overlapping chunks maintain context.
- Smaller chunks isolate MCQs for better semantic retrieval.

### 3. What embedding model did you use?

- Model: `l3cube-pune/bengali-sentence-similarity-sbert`  
- It captures Bengali semantics and syntax better than multilingual models.

### 4. How is similarity calculated?

- Cosine similarity is used between query and document embeddings.
- Embeddings are stored in **ChromaDB** for efficient vector retrieval.

### 5. What happens with vague queries?

- Vague queries may return less relevant results.
- Semantic chunking and consistent embeddings help mitigate this.

### 6. How can it be improved?

- Use semantic chunking.
- Fine-tune the embedding model on domain data.
- Re-rank retrieved chunks with a cross-encoder.
- Expand corpus and improve OCR preprocessing.

---

EOF