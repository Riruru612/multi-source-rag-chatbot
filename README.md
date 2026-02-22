# Confidence-Driven Multi-Agent RAG System with Tool Orchestration and Evaluation Layer

**A production-style Retrieval-Augmented Generation (RAG) system built using LangChain, FAISS, SentenceTransformers, and Streamlit.**

This application allows users to upload documents (PDF, DOCX, TXT, or URLs) and ask contextual questions. The system uses confidence-based routing to decide whether to answer using uploaded documents, the web, or both — improving control and reliability over traditional LLM-only systems.



## Key Highlights
	•	Conversational RAG with session memory
	•	FAISS vector store for efficient similarity search
	•	Embedding-based reranking using SentenceTransformer
	•	Confidence-driven hybrid tool routing
	•	Web fallback via Tavily Search API
	•	LLM-based critic for answer validation
	•	Transparent agent decision display in the UI

## This project demonstrates applied knowledge of:
	•	Retrieval-Augmented Generation (RAG)
	•	Vector similarity search
	•	Embedding-based reranking
	•	Hybrid deterministic + LLM systems
	•	Multi-agent coordination
	•	Production-style modular architecture

## Architecture Overview

User Query
↓
Embedding-Based Reranker (Confidence Score)
↓
Deterministic Routing Layer
↓
Research Agent (Tool Execution)
• Document Tool (RAG + FAISS + Reranking)
• Web Tool (Tavily API)
↓
Critic Agent (LLM Evaluation)
↓
Final Answer

## Routing Strategy

## Tool selection is determined using cosine similarity confidence scores:
	•	High confidence → Document only
	•	Medium confidence → Document + Web
	•	Low confidence → Web
	•	Very low confidence → Reject as irrelevant

**This avoids over-reliance on LLM-based tool selection and improves system interpretability.**

## Tech Stack

**Component               Technology**
Frontend                Streamlit
Framework               LangChain
Vector Store            FAISS
Embeddings              SentenceTransformers (all-MiniLM-L6-v2)
Reranking               Cosine Similarity
LLM Provider            Groq (Llama 3.3 70B)
Web Search              Tavily API

## Installation

 # 1. Clone the repository
git clone https://github.com/Riruru612/multi-source-rag-chatbot.git

# 2. Navigate into the project
cd multi-source-rag-chatbot

# 3. Create a virtual environment
conda create -n rag_env python=3.10
conda activate rag_env

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create a .env file and add:
GROQ_API=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token

# 6. Run the application
streamlit run app.py

# What This Project Demonstrates
	•	Implementation of a modular RAG pipeline
	•	Embedding-based document reranking
	•	Deterministic tool routing using similarity confidence
	•	Multi-agent orchestration (Research + Critic)
	•	Clean separation of concerns in system design
	•	Production-ready project structuring
