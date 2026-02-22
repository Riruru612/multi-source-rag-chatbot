def critic_evaluate(llm, question, answer):
    """
    Evaluate if answer is sufficient and grounded.
    """

    prompt = f"""
You are a critic agent.

Question:
{question}

Answer:
{answer}

Evaluate:
1. Is the answer relevant?
2. Is it complete?
3. Is additional research needed?

Respond with:
APPROVED
or
RESEARCH_AGAIN
"""

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()

    if "RESEARCH_AGAIN" in verdict:
        return "RESEARCH_AGAIN"

    return "APPROVED"