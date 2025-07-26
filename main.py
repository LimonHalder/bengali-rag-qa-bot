from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple

from rag_logic import answer_question_dual, chat_history

app = FastAPI(title="Bengali RAG QA API")


class QuestionRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    user_query = payload.query.strip()
    if not user_query:
        return {"error": "Empty query"}

    chat_history.append(("user", user_query))
    answer = answer_question_dual(user_query, chat_history=chat_history)
    chat_history.append(("assistant", answer))

    return {"answer": answer}


app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/conversation")
def serve_frontend():
    return FileResponse("index.html")
