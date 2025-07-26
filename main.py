# rag_api.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from langchain_google_genai import ChatGoogleGenerativeAI

# ==== Configuration ====
os.environ["GOOGLE_API_KEY"] = "AIzaSyAfJtcte_TRfn-W8EuqevAKefz3e1FayNw"  # Replace with env var in production
persist_dir = "./chroma_db"
collection_name = "qa_collection"
embed_model_id = "l3cube-pune/bengali-sentence-similarity-sbert"

# ==== Initialize Components ====
model = HuggingFaceEmbeddings(model_name=embed_model_id)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
client = PersistentClient(path=persist_dir)
collection = client.get_collection(name=collection_name)

app = FastAPI(title="Bengali RAG QA API")

# ==== Short-term Memory (in-memory) ====
chat_history: List[Tuple[str, str]] = []

# ==== Request Body ====
class QuestionRequest(BaseModel):
    query: str


# ==== Core RAG Logic ====
def answer_question_dual(query, chat_history=None, top_k=10):
    query_embedding = model.embed_query(query)

    mcq_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source": "mcq"}
    )

    passage_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source": "passage"}
    )

    combined_results = []

    def extract_results(results, source):
        docs = results.get("documents", [[]])[0]
        scores = results.get("distances", [[]])[0]
        for doc, score in zip(docs, scores):
            combined_results.append({
                "doc": doc,
                "score": score,
                "source": source
            })

    extract_results(mcq_results, "mcq")
    extract_results(passage_results, "passage")

    combined_results = sorted(combined_results, key=lambda x: x["score"])
    top_contexts = [res["doc"] for res in combined_results]

    short_term_text = ""
    if chat_history:
        for role, message in chat_history[-5:]:
            short_term_text += f"{role.title()}: {message}\n"


    prompt = (
        "You are a helpful assistant. Use the following long-term context and chat history to answer the question.\n\n"
        f"📚 Long-Term Context (MCQ + Passage):\n{''.join(top_contexts)}\n\n"
        f"💬 Short-Term Chat History:\n{short_term_text}"
        f"\n❓ Question: {query}\nAnswer in Bengali:"
    )

    response = llm.invoke(prompt)
    return response.content.strip()


# ==== API Endpoint ====
@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    user_query = payload.query.strip()
    if not user_query:
        return {"error": "Empty query"}

    chat_history.append(("user", user_query))
    answer = answer_question_dual(user_query, chat_history=chat_history)
    chat_history.append(("assistant", answer))

    return {"answer": answer}


# ==== Serve Static Frontend ====
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/conversation")
def serve_frontend():
    return FileResponse("index.html")
