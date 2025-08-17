import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, Docx2txtLoader, TextLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

st.title("Conversational Chatbot")
st.write("Upload your content (PDF, DOCX, Text, or URL) and chat with it.")

groq_api_key = os.getenv("GROQ_API")
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)

session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

input_type = st.selectbox("Select input type", ["PDF", "DOCX", "Text", "URL"])

documents = []
if input_type == "PDF":
    upload_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if upload_file:
        for up in upload_file:
            temppdf = "./temp.pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.getvalue())
                temp_path = tmp.name
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path)

elif input_type == "DOCX":
    upload_file = st.file_uploader("Upload DOCX", type="docx", accept_multiple_files=True)
    if upload_file:
        for up in upload_file:
            tempdoc = "./temp.docx"
            with open(tempdoc, "wb") as file:
                file.write(up.getvalue())
            loader = Docx2txtLoader(tempdoc)
            documents.extend(loader.load())

elif input_type == "Text":
    upload_file = st.file_uploader("Upload Text File", type="txt", accept_multiple_files=True)
    if upload_file:
        for up in upload_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(up.getvalue())
                temp_path = tmp.name
            loader = TextLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path)
    text_input = st.text_area("Or paste text here")
    if text_input:
        documents.append(Document(page_content=text_input))

elif input_type == "URL":
    url_input = st.text_input("Enter URL")
    if url_input:
        loader = WebBaseLoader(url_input)
        documents.extend(loader.load())

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualized_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question, reformulate it into a standalone question. Do not answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt)

    system_prompt = "You are a helpful chatbot that answers user questions based strictly on the provided context.\n\n{context}"
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your Question:")
    if user_input:
        session_history = get_session_history(session_id)
        res = conversation_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.write("Assistant:", res["answer"])