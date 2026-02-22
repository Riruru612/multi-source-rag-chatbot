from langchain.tools import tool


def create_document_tool(rag_chain, session_id):

    @tool
    def document_search(query: str) -> str:
        """Search the uploaded document for relevant information.
        Use this tool when the question is related to the uploaded content.
        """

        result = rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )

        return result["answer"]

    return document_search