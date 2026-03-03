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

#  perform tavily search for a given query and return the results
async def perform_research(query: str):
    # This is a placeholder implementation. You would replace this with actual calls to your search tool or API.
    search_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=2, search_depth="basic")  # type: ignore

    response = await search_tool.ainvoke({"query": query})
    
    return response

# normalizing the research results into a consistent format for the reducer to consume
def normalize_tavily_results(results: List[Dict]) -> List[Dict]:
    normalized = []

    for r in results:
        content = r.get("content") or ""
        normalized.append({
            "title": r.get("title"),
            "content": content[:300],  # Truncate to prevent Groq function-calling failures
            "url": r.get("url"),
        })  

    return normalized


# if __name__ == "__main__":
#     research_results = asyncio.run(perform_research("Oracle trending news"))
#     print(research_results)


# parser mode 

def parse_mode(user_input: str):
    if ":" not in user_input:
        return None, user_input

    prefix, content = user_input.split(":", 1)
    prefix = prefix.strip().lower()
    content = content.strip()

    allowed_modes = {"chat", "generate", "refine", "publish"}

    if prefix in allowed_modes:
        return prefix, content

    return None, user_input