from agents.research_agent import research_execute
from agents.critic_agent import critic_evaluate


def run_multi_agent_system(
    llm,
    question,
    document_tool,
    web_tool,
    retriever
):

    try:
        docs = retriever.get_relevant_documents(question)

        if docs:
            top_doc = docs[0]
            confidence = top_doc.metadata.get("rerank_score", 0)
            print("Confidence raw:", confidence, type(confidence))
        else:
            confidence = 0

    except Exception:
        confidence = 0
    if confidence < 0.25:
       return {
        "decision": "REJECT",
        "confidence": confidence,
        "critic_verdict": "N/A",
        "answer": "This question is not related to the uploaded document."
    }

    elif confidence > 0.6:
        decision = "DOCUMENT"

    elif confidence > 0.45:
        decision = "BOTH"

    else:
        decision = "WEB"
    answer = research_execute(
        decision,
        question,
        document_tool,
        web_tool
    )

    verdict = critic_evaluate(llm, question, answer)

    if verdict == "RESEARCH_AGAIN":
        answer = research_execute(
            "BOTH",
            question,
            document_tool,
            web_tool
        )

    return {
        "answer": answer,
        "confidence": confidence,
        "decision": decision,
        "critic_verdict": verdict
    }