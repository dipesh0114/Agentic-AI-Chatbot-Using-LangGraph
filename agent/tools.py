# agent/tools.py
# ============================================================
# TOOL DEFINITIONS
# ============================================================
# Tools are external capabilities the agent can call.
# Each tool is a Python function decorated with @tool.
# The agent (LLM) decides WHEN and WHICH tool to invoke.
# ============================================================

import wikipedia
import requests
from langchain_core.tools import tool
from duckduckgo_search import DDGS


# ------------------------------------------------------------------
# Tool 1: DuckDuckGo Web Search
# ------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for real-time information.
    Use this tool when the user asks about:
    - Current events or news
    - Recent facts not in training data
    - Prices, weather, sports results, etc.

    Args:
        query: The search query string.

    Returns:
        A string with top search results.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No search results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"Title: {r.get('title', 'N/A')}\n"
                f"Body: {r.get('body', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
            )
        return "\n".join(formatted)

    except Exception as e:
        return f"Search failed: {str(e)}"


# ------------------------------------------------------------------
# Tool 2: Wikipedia Search
# ------------------------------------------------------------------
@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for detailed factual information.
    Use this tool when the user asks about:
    - Historical events, people, places
    - Scientific concepts, definitions
    - Encyclopedic knowledge

    Args:
        query: The topic to search for on Wikipedia.

    Returns:
        A summary of the Wikipedia article.
    """
    try:
        # Search for top results
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return "No Wikipedia articles found for this query."

        # Get summary of the best match
        page = wikipedia.page(search_results[0], auto_suggest=False)
        summary = wikipedia.summary(search_results[0], sentences=5)

        return (
            f"Wikipedia Article: {page.title}\n"
            f"URL: {page.url}\n\n"
            f"Summary:\n{summary}"
        )

    except wikipedia.exceptions.DisambiguationError as e:
        # If there are multiple matches, pick the first
        try:
            summary = wikipedia.summary(e.options[0], sentences=5)
            return f"Wikipedia Summary (disambiguation resolved):\n{summary}"
        except Exception:
            return f"Multiple results found: {', '.join(e.options[:5])}"

    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."

    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


# ------------------------------------------------------------------
# Tool 3: Calculator / Math Evaluator
# ------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Use this tool when the user asks to:
    - Calculate numbers, percentages, equations
    - Solve arithmetic or algebra

    Args:
        expression: A valid Python math expression string.
                    Examples: "2 + 2", "100 * 0.15", "2 ** 10"

    Returns:
        The result of the calculation as a string.
    """
    try:
        # Only allow safe math operations
        allowed_names = {
            k: v for k, v in vars(__builtins__).items()
            if k in ["abs", "round", "min", "max", "sum", "pow", "int", "float"]
        }
        import math
        allowed_names.update({k: v for k, v in vars(math).items() if not k.startswith("_")})

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Calculation error: {str(e)}. Please provide a valid math expression."


# ------------------------------------------------------------------
# Tool 4: Get Current Date & Time
# ------------------------------------------------------------------
@tool
def get_current_datetime() -> str:
    """
    Get the current date and time.
    Use this tool when the user asks:
    - 'What time is it?'
    - 'What is today's date?'
    - 'What day is it?'

    Returns:
        Current date, time, and day of the week.
    """
    from datetime import datetime
    now = datetime.now()
    return (
        f"Current Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Current Time: {now.strftime('%I:%M:%S %p')}\n"
        f"Timezone: Local System Time"
    )


# ------------------------------------------------------------------
# All tools bundled for easy import
# ------------------------------------------------------------------
ALL_TOOLS = [
    web_search,
    wikipedia_search,
    calculator,
    get_current_datetime,
]