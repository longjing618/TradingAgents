import json
import os
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from ollama import Client

from .config import get_config


@tool
def ollama_web_search(query: str) -> List[Dict[str, Any]]:
    """
    Args:
        query: The search query string and get the top 10 results ONLY and no more
    Returns:
        List of search results with title, url, and content
    """
    api_key = os.environ.get("OLLAMA_API_KEY")
    client = Client(
        host="https://ollama.com/api",  # use the cloud endpoint
        headers={"Authorization": f"Bearer {api_key}"},
    )

    try:
        response = client.web_search(query)
        return response.get("results", [])
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


@tool
def ollama_web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetch content from a specific URL using Ollama's web fetch API.

    Args:
        url: The URL to fetch content from
    Returns:
        Dictionary with title, content, and links from the page
    """
    try:
        api_key = os.environ.get("OLLAMA_API_KEY")
        client = Client(
            host="https://ollama.com/api",  # use the cloud endpoint
            headers={"Authorization": f"Bearer {api_key}"},
        )

        response = client.web_fetch(url)
        return {
            "title": response.title,
            "content": response.content,
            "links": response.Links,
        }
    except Exception as e:
        return {"error": f"Fetch failed: {str(e)}"}


def get_global_news_ollama(curr_date, look_back_days=7, limit=5):
    config = get_config()

    model = ChatOpenAI(
        model=config["quick_think_llm"],
        base_url=config["backend_url"],
    )

    agent = create_agent(model, tools=[ollama_web_search, ollama_web_fetch])

    # per https://share.google/aimode/GbZ3L99UZMeiALegm
    # no need to update prompt for "tool is available"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles.",
            ),
        ]
    )
    result = agent.invoke(
        prompt.invoke(
            {
                "look_back_days": look_back_days,
                "curr_date": curr_date,
                "limit": limit,
            }
        )
    )

    # https://forum.langchain.com/t/how-to-retireve-the-final-ai-message-after-invoke/2078/2
    messages = result["messages"]
    # Find the last assistant message (the graph may end on a ToolMessage otherwise)
    last_ai = next(m for m in reversed(messages) if getattr(m, "type", None) == "ai")
    # Safely extract human-readable text

    print(f"DEBUG: get_global_news_ollama: {last_ai.content}")
    return last_ai.content
