import tempfile
import os
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, Docx2txtLoader, TextLoader
)
from langchain_core.documents import Document

def load_documents(input_type, uploaded_files=None, text_input=None, url_input=None):
    documents = []

    if input_type == "PDF" and uploaded_files:
        for up in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.getvalue())
                temp_path = tmp.name
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path)

    elif input_type == "DOCX" and uploaded_files:
        for up in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(up.getvalue())
                temp_path = tmp.name
            loader = Docx2txtLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path)

    elif input_type == "Text":
        if uploaded_files:
            for up in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(up.getvalue())
                    temp_path = tmp.name
                loader = TextLoader(temp_path)
                documents.extend(loader.load())
                os.remove(temp_path)

        if text_input:
            documents.append(Document(page_content=text_input))

    elif input_type == "URL" and url_input:
        loader = WebBaseLoader(url_input)
        documents.extend(loader.load())

    return documents