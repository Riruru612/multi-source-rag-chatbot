from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
reranker_model = SentenceTransformer("all-MiniLM-L6-v2")


def rerank_documents(query, docs, top_k=3):

    if not docs:
        return []

    query_embedding = reranker_model.encode(query)

    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = reranker_model.encode(doc_texts)

    scores = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = []

    for doc, score in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        top_docs.append(doc)

    return top_docs