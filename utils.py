import re
from typing import List
from langchain_tavily.tavily_search import TavilySearch
from typing import List, Dict
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Utility function to create a safe filename from a blog title
def safe_filename(title: str) -> str:
    # lower case
    name = title.lower()

    # replace spaces with underscore
    name = name.replace(" ", "_")

    # remove anything that's NOT a-z, 0-9, _ or -
    name = re.sub(r"[^a-z0-9_-]", "", name)

    # avoid empty names
    if not name:
        name = "blog"

    return f"{name}.md"


async def perform_research(query: str):
    # This is a placeholder implementation. You would replace this with actual calls to your search tool or API.
    search_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=2, search_depth="basic")  # type: ignore

    response = await search_tool.ainvoke({"query": query})
    
    return response

# normalizing the research results into a consistent format for the reducer to consume
def normalize_tavily_results(results: List[Dict]) -> List[Dict]:
    normalized = []

    for r in results:
        normalized.append({
            "title": r.get("title"),
            "content": r.get("content"),
            "url": r.get("url"),
            # "source": None,           # Tavily doesn't provide this
            # "published_date":'date:unknown',   # Tavily doesn't provide this
        })

    return normalized


# if __name__ == "__main__":
#     research_results = asyncio.run(perform_research("Oracle trending news"))
#     print(research_results)
