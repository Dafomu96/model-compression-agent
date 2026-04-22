from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

CHROMA_DIR = "data/chroma_db"

# --- Estado del grafo ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_relevant: bool
    is_grounded: bool

# --- Componentes ---
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")

# --- Nodos ---
def retrieve(state: AgentState) -> AgentState:
    print("--- NODO: RETRIEVE ---")
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"documents": [doc.page_content for doc in docs]}

def grade_documents(state: AgentState) -> AgentState:
    print("--- NODO: GRADE ---")
    llm = get_llm()
    question = state["question"]
    documents = "\n\n".join(state["documents"])

    messages = [
        SystemMessage(content="""You are a grader assessing relevance of retrieved documents to a user question.
        Answer only 'yes' or 'no'. 'yes' if the documents contain information relevant to answer the question."""),
        HumanMessage(content=f"Question: {question}\n\nDocuments: {documents}")
    ]
    result = llm.invoke(messages)
    is_relevant = "yes" in result.content.lower()
    return {"is_relevant": is_relevant}

def generate(state: AgentState) -> AgentState:
    print("--- NODO: GENERATE ---")
    llm = get_llm()
    question = state["question"]
    documents = "\n\n".join(state["documents"])

    messages = [
        SystemMessage(content="""You are an expert assistant on model compression techniques (pruning, quantization, knowledge distillation).
        Answer the question based on the provided documents. Be concise and cite the source papers when possible."""),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{documents}")
    ]
    result = llm.invoke(messages)
    return {"generation": result.content}

def no_answer(state: AgentState) -> AgentState:
    print("--- NODO: NO ANSWER ---")
    return {"generation": "No encontré información relevante en los papers para responder esta pregunta."}

def check_hallucination(state: AgentState) -> AgentState:
    print("--- NODO: HALLUCINATION CHECK ---")
    llm = get_llm()
    documents = "\n\n".join(state["documents"])
    generation = state["generation"]

    messages = [
        SystemMessage(content="""You are a grader checking if an answer is grounded in the provided documents.
        Answer only 'yes' or 'no'. 'yes' if the answer is supported by the documents, 'no' if it contains information not found in the documents."""),
        HumanMessage(content=f"Documents: {documents}\n\nAnswer: {generation}")
    ]
    result = llm.invoke(messages)
    is_grounded = "yes" in result.content.lower()
    return {"is_grounded": is_grounded}

# --- Routing ---
def route_after_grade(state: AgentState) -> str:
    if state["is_relevant"]:
        return "generate"
    return "no_answer"

def route_after_hallucination(state: AgentState) -> str:
    if state["is_grounded"]:
        return END
    return "generate"

# --- Construcción del grafo ---
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)
    graph.add_node("hallucination_check", check_hallucination)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("generate", "hallucination_check")
    graph.add_conditional_edges("hallucination_check", route_after_hallucination)
    graph.add_edge("no_answer", END)

    return graph.compile()

# --- Test ---
if __name__ == "__main__":
    agent = build_graph()
    result = agent.invoke({
        "question": "What is knowledge distillation and how does it compress neural networks?",
        "documents": [],
        "generation": "",
        "is_relevant": False,
        "is_grounded": False
    })
    print("\n=== RESPUESTA ===")
    print(result["generation"])