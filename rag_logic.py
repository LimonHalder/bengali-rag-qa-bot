import os
from typing import List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer, util

# ==== Configuration ====
os.environ["GOOGLE_API_KEY"] = "AIzaSyAfJtcte_TRfn-W8EuqevAKefz3e1FayNw"  # Replace with env var in production
persist_dir = "./chroma_db"
collection_name = "qa_collection"
embed_model_id = "l3cube-pune/bengali-sentence-similarity-sbert"

# ==== Initialize Components ====
model = HuggingFaceEmbeddings(model_name=embed_model_id)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
client = PersistentClient(path=persist_dir)

# Ensure the collection exists
try:
    collection = client.get_collection(name=collection_name)
except Exception:
    client.create_collection(name=collection_name)
    collection = client.get_collection(name=collection_name)

# ==== Evaluation Model ====
eval_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ==== Short-term Memory (in-memory) ====
chat_history: List[Tuple[str, str]] = []


def answer_question_dual(query: str, chat_history: List[Tuple[str, str]] = None, top_k: int = 10) -> dict:
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
        f"\n❓ Question: {query}\nAnswer in one or two Bengali words:"
    )

    response = llm.invoke(prompt)
    answer = response.content.strip()

    # === Evaluation ===
    answer_emb = eval_model.encode(answer, convert_to_tensor=True)
    context_emb = eval_model.encode(" ".join(top_contexts), convert_to_tensor=True)
    groundedness_score = util.cos_sim(answer_emb, context_emb).item()

    query_emb_eval = eval_model.encode(query, convert_to_tensor=True)
    relevance_score = util.cos_sim(query_emb_eval, context_emb).item()

    # Optional: Log
    print(f"[🔍 Evaluation] Groundedness Score: {groundedness_score:.2f}")
    print(f"[🔍 Evaluation] Relevance Score: {relevance_score:.2f}")

    return {
        "answer": answer,
        "groundedness_score": round(groundedness_score, 4),
        "relevance_score": round(relevance_score, 4)
    }
