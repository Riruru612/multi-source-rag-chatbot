import streamlit as st
from langchain_groq import ChatGroq
from config import GROQ_API
from embeddings.hf_embeddings import get_embeddings
from Documents.document_loader import load_documents
from vectorstore.faiss_store import create_vectorstore, RerankRetriever
from chain.rag_chain import create_rag_chain
from utils.session_history import get_session_history
from tools.document_tool import create_document_tool
from tools.web_tool import web_search
from agents.coordinator import run_multi_agent_system


st.title("Multi-Agent Conversational Chatbot")

if "store" not in st.session_state:
    st.session_state.store = {}


embeddings = get_embeddings()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API,
    temperature=0
)

session_id = st.text_input("Session ID", value="default_session")

input_type = st.selectbox(
    "Select input type",
    ["PDF", "DOCX", "Text", "URL"]
)

uploaded_files = None
text_input = None
url_input = None

if input_type == "PDF":
    uploaded_files = st.file_uploader(
        "Upload PDF",
        type="pdf",
        accept_multiple_files=True
    )

elif input_type == "DOCX":
    uploaded_files = st.file_uploader(
        "Upload DOCX",
        type="docx",
        accept_multiple_files=True
    )

elif input_type == "Text":
    uploaded_files = st.file_uploader(
        "Upload Text File",
        type="txt",
        accept_multiple_files=True
    )
    text_input = st.text_area("Or paste text here")

elif input_type == "URL":
    url_input = st.text_input("Enter URL")


documents = load_documents(
    input_type,
    uploaded_files,
    text_input,
    url_input
)

if documents:
    vectorstore = create_vectorstore(documents, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    retriever = RerankRetriever(
        base_retriever=base_retriever,
        rerank_k=3
    )
    conversation_rag_chain = create_rag_chain(
        llm,
        retriever,
        lambda s: get_session_history(st.session_state.store, s)
    )
    document_tool = create_document_tool(
        conversation_rag_chain,
        session_id
    )

    user_input = st.text_input("Your Question:")

    if user_input:

        result = run_multi_agent_system(
        llm=llm,
        question=user_input,
        document_tool=document_tool,
        web_tool=web_search,
        retriever=retriever 
    )

        st.write("### Agent Decision Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Routing Decision", result["decision"])

        with col2:
            st.metric("Retrieval Confidence", round(result["confidence"], 3))

        st.write("**Critic Verdict:**", result["critic_verdict"])

        st.write("---")

        st.write("### Assistant Response")
        st.write(result["answer"])