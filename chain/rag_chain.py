from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

def create_rag_chain(llm, retriever, get_session_history):

    contextualized_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and latest question, reformulate it into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualized_prompt
    )

    system_prompt = """
    You are a precise AI assistant.

    Rules:
    - Answer in 3-5 short sentences.
    - Be clear and concise.
    - Do not repeat information.
    - Do not add extra explanation.
    - Use bullet points if appropriate.
    - Only answer what is asked.
    - Base your answer strictly on the provided context.

    Context:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )