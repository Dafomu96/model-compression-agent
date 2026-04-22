# \# Model Compression Agent

# 

# RAG + LangGraph agent that answers questions about neural network compression techniques (pruning, quantization, knowledge distillation) based on 57 scientific papers from arXiv.

# 

# \## Architecture

# User Question

# │

# ▼

# \[RETRIEVE] → ChromaDB vector search (top-4 chunks)

# │

# ▼

# \[GRADE] → LLM evaluates document relevance

# │

# ┌──┴──┐

# │     │

# ▼     ▼

# \[GEN] \[NO ANS]

# │

# ▼

# Answer + Sources

# 

# \## Stack

# 

# \- \*\*LLM\*\*: Llama 3.3 70B via Groq API

# \- \*\*Embeddings\*\*: all-MiniLM-L6-v2 (sentence-transformers)

# \- \*\*Vector store\*\*: ChromaDB

# \- \*\*Orchestration\*\*: LangGraph

# \- \*\*API\*\*: FastAPI

# 

# \## Results

# 

# Evaluated on 5 domain-specific questions:

# \- \*\*Relevance\*\*: 5/5 (100%)

# \- Topics covered: knowledge distillation, pruning, quantization, lottery ticket hypothesis

# 

# \## Setup

# 

# ```bash

# git clone https://gitlab.com/Dafomu96/model-compression-agent

# cd model-compression-agent

# pip install -r requirements.txt

# cp .env.example .env  # add your GROQ\_API\_KEY

# ```

# 

# Index the papers:

# ```bash

# python src/ingest.py

# ```

# 

# Run the API:

# ```bash

# uvicorn src.api:app --reload

# ```

# 

# Open `http://localhost:8000/docs` to interact with the agent.

# 

# Run evaluation:

# ```bash

# python src/evaluate.py

# ```

# 

# \## Self-hosted LLM (optional)

# 

# Replace Groq with Ollama for fully offline usage:

# ```python

# from langchain\_ollama import ChatOllama

# llm = ChatOllama(model="llama3.2")

# ```

# 

# \## Paper corpus

# 

# 57 papers from arXiv covering:

# \- Neural network pruning (magnitude-based, structured, unstructured)

# \- Knowledge distillation (response-based, feature-based)

# \- Model quantization (post-training, quantization-aware training)

