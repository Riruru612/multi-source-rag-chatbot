from langchain.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    response = tavily.search(query=query, max_results=3)
    results = [r["content"] for r in response["results"]]
    return "\n".join(results)