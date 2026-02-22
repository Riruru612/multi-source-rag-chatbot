from typing import List
from pydantic import Field
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrievers.reranker import rerank_documents


def create_vectorstore(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(splits, embeddings)


class RerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)
    rerank_k: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        reranked_docs = rerank_documents(
            query,
            docs,
            top_k=self.rerank_k
        )

        for i, doc in enumerate(reranked_docs):
            print(f"\nRERANKED DOC  {i+1}:\n{doc.page_content[:300]}")

        return reranked_docs