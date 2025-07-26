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
## Evaluation Metrics

We evaluate the RAG system using cosine similarity between embeddings from the **`sentence-transformers/all-MiniLM-L6-v2`** model:

- **Groundedness Score:** Similarity between the generated answer and retrieved context embeddings, indicating how well the answer is supported.
- **Relevance Score:** Similarity between the query and retrieved context embeddings, indicating the relevance of the context.

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

### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

In the project, **PyMuPDF (fitz)** and **PyPDF2** libraries were used to extract divide the pdf into two section for passage and mcq. The OCR techniques (using **pytesseract**) integrated are used from extract the text from the sectioned pdf and store in two different txt files. 
Yes, OCr can not proper extract mcq question's answer from the answer box(SL:ans pair of 100 mcq).

**Formatting challenges:**
PDF files often had inconsistent line breaks, embedded images, and mixed formatting which led to fragmented or broken sentences. Bengali-specific script complexities (like conjuncts and vowel signs) sometimes caused extraction artifacts, requiring extra cleanup and normalization during preprocessing.

Later i use a llm for structured output suitable for vector embedding.

---

### 2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

The chunking strategy employed is primarily **paragraph-based chunking combined with a character limit** (around 500–700 characters). Text was split at natural paragraph breaks (double newlines or explicit markers), then further segmented if too long.

Two way. One for passage type text, where i used chnk_size=400 with overlappinig=200.
another on is for formated mcq type text ,  where i used chunk_size=100, overlap=10.

**Why it works well:**
---

Different chunk sizes suit different text types: larger, overlapping chunks capture full context in long passages, improving relevance, while smaller chunks better isolate individual MCQs for precise retrieval. This balance ensures both accurate and efficient document retrieval.


---

### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

The project uses the **HuggingFace sentence-transformers model `l3cube-pune/bengali-sentence-bert`** for generating embeddings.

**Reasons for choice:**

* It’s specifically fine-tuned for Bengali language semantics, improving representation quality over generic multilingual models.
* Sentence-BERT models encode sentences or paragraphs into fixed-length vectors optimized for semantic similarity tasks, ensuring semantically similar texts have embeddings close in vector space.
* This helps capture meaning beyond keyword matching, encoding nuances of Bengali text including syntax and semantics.

---

### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

**Similarity comparison:** The system uses **cosine similarity** between the query embedding and each document chunk embedding to find the most relevant chunks.

**Storage setup:** Embeddings are stored in **ChromaDB**, a vector database that supports fast similarity search and persistent storage.

**Why this method and setup:**

* Cosine similarity is widely adopted for embedding comparison due to its effectiveness in measuring semantic closeness independent of vector length.
* ChromaDB offers efficient indexing, scalable vector storage, and quick retrieval necessary for a responsive RAG system.
* This combination ensures retrieval of semantically relevant document chunks to feed into the LLM for answer generation.

---

### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

**Ensuring meaningful comparison:**

* Both the query and document chunks are encoded using the same Bengali sentence-BERT model, ensuring consistent embedding space and semantic alignment.
* Paragraph-based chunking maintains coherent semantic units, avoiding fragmented or ambiguous embeddings.
* Retrieval uses top-k nearest neighbors to focus on the most semantically relevant chunks.

**If the query is vague or missing context:**

* The system may retrieve less relevant or overly broad chunks, leading to generic or off-topic answers.
* Lack of context reduces embedding precision, so similarity scores drop or the results become noisy.
* To mitigate, one could incorporate query expansion, conversational history, or clarify ambiguous queries to improve retrieval focus.

---

### 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

The results from the current setup are reasonably relevant due to:

* Use of a Bengali-specific embedding model.
* Logical paragraph-based chunking that preserves semantic meaning.
* Efficient similarity search in a vector database.

**Potential improvements:**

* **Better chunking:** Using semantic segmentation or adaptive chunking that respects sentence boundaries or topical shifts.
* **Embedding models:** Fine-tuning the sentence-BERT model further on domain-specific data or experimenting with larger multilingual models like XLM-R or LaBSE for improved contextual understanding.
* **Corpus expansion:** Incorporating more comprehensive and diverse documents to enrich knowledge coverage.
* **Query enhancement:** Including query expansion or context enrichment to reduce ambiguity.
* **Re-ranking:** Applying a cross-encoder re-ranking step after retrieval to improve final answer relevance.

---

