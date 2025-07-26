from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Dict, Any

from rag_logic import answer_question_dual, chat_history

app = FastAPI(title="Bengali RAG QA API")

class QuestionRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(payload: QuestionRequest) -> Dict[str, Any]:
    user_query = payload.query.strip()
    if not user_query:
        return {"error": "Empty query"}

    chat_history.append(("user", user_query))
    result = answer_question_dual(user_query, chat_history=chat_history)
    answer = result["answer"]
    chat_history.append(("assistant", answer))

    return {
        "answer": answer,
        "groundedness_score": result.get("groundedness_score"),
        "relevance_score": result.get("relevance_score"),
    }

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/conversation")

@app.get("/conversation")
def serve_frontend():
    return FileResponse("index.html")

