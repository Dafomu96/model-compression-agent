import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PAPERS_DIR = Path("data/papers")
CHROMA_DIR = Path("data/chroma_db")

def load_pdfs():
    docs = []
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Encontrados {len(pdf_files)} PDFs")
    
    for pdf_path in pdf_files:
        print(f"Cargando: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Añadir metadata con el arxiv_id
        arxiv_id = pdf_path.stem
        for page in pages:
            page.metadata["arxiv_id"] = arxiv_id
            page.metadata["source"] = pdf_path.name
        
        docs.extend(pages)
    
    print(f"Total páginas cargadas: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks generados: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    print("Creando embeddings y vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"Vectorstore creado en {CHROMA_DIR}")
    return vectorstore

if __name__ == "__main__":
    docs = load_pdfs()
    chunks = split_documents(docs)
    create_vectorstore(chunks)
    print("Ingesta completada!")