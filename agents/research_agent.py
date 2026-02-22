def research_execute(decision, question, document_tool, web_tool):
    """
    Execute tools based on planner decision.
    """

    if decision == "DOCUMENT":
        return document_tool.invoke(question)

    elif decision == "WEB":
        return web_tool.invoke(question)

    elif decision == "BOTH":
        doc_part = document_tool.invoke(question)
        web_part = web_tool.invoke(question)
        return doc_part + "\n\n" + web_part

    else:
        return document_tool.invoke(question)