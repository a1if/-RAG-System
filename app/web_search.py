import os
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import traceback

load_dotenv()

# Initialize the Tavily client
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str, num_results: int = 3) -> Tuple[str, List[Dict[str, str]]]:
    """
    Performs a web search using Tavily and formats the results for RAG.

    Args:
        query: The user's search query, generated from the conversation context.
        num_results: The number of web search results to fetch.

    Returns:
        A tuple containing:
        - A combined context string formatted for the LLM.
        - A list of dictionaries representing the sources.
    """
    print(f"DEBUG: [web_search] Performing web search for query: '{query}' with max_results={num_results}")

    context_string = ""
    sources = []

    if not client.api_key:
        print("❌ TAVILY_API_KEY is not set.")
        return "Error: Tavily API key is missing.", []

    try:
        # Perform the search using Tavily
        results = client.search(
            query=query,
            search_depth="advanced",
            max_results=num_results, # The dynamic parameter is used here
        )
        
        if not results or "results" not in results or not results["results"]:
            print(f"⚠️ No results found for query: '{query}'")
            return "No relevant information found for the query.", []

        chunks = []
        
        # Process each result
        for result in results["results"]:
            title = result.get("title", "No Title")
            raw_content = result.get("content", "No content available.")
            content = raw_content
            url = result.get("url", "#")

            # Create a formatted chunk for the context string
            chunks.append(f"### {title}\n\n{content}\n\n[Source]({url})")

            sources.append({
                "name": title,
                "url": url,
                "type": "web"
            })

        context_string = "\n\n---\n\n".join(chunks)

    except Exception as e:
        print(f"❌ Error during Tavily web search for query '{query}': {e}")
        traceback.print_exc()
        raise e

    if not context_string:
         context_string = "Web search completed, but no suitable context was extracted."

    return context_string, sources

