from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_graph

app = FastAPI(
    title="Model Compression Agent",
    description="RAG + LangGraph agent for model compression papers",
    version="0.1.0"
)

agent = build_graph()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    is_relevant: bool
    is_grounded: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = agent.invoke({
        "question": request.question,
        "documents": [],
        "generation": "",
        "is_relevant": False,
        "is_grounded": False
    })
    return QueryResponse(
        question=request.question,
        answer=result["generation"],
        is_relevant=result["is_relevant"],
        is_grounded=result["is_grounded"]
    )