# agent/nodes.py
# ============================================================
# NODE DEFINITIONS
# ============================================================
# Nodes are the individual steps in the LangGraph workflow.
# Each node is a Python function that:
#   1. Receives the current AgentState
#   2. Does some work (calls LLM, runs a tool, etc.)
#   3. Returns a dict to UPDATE the state
#
# LangGraph automatically merges the returned dict into state.
# ============================================================

import os
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import ALL_TOOLS


# ------------------------------------------------------------------
# Load the LLM based on environment config
# ------------------------------------------------------------------
def load_llm():
    """
    Loads the LLM based on LLM_PROVIDER env variable.
    Supports: groq (default/free), openai, anthropic
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"[LLM] Using Groq → {model}")
        return ChatGroq(
            model=model,
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"[LLM] Using OpenAI → {model}")
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        print(f"[LLM] Using Anthropic → {model}")
        return ChatAnthropic(
            model=model,
            temperature=0.7,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Set it to: groq | openai | anthropic"
        )


# ------------------------------------------------------------------
# System Prompt — Defines the Agent's Personality & Rules
# ------------------------------------------------------------------
SYSTEM_PROMPT = SystemMessage(content="""
You are an intelligent Agentic AI Assistant powered by LangGraph.

You have access to the following tools:
- **web_search**: Search the internet for real-time information
- **wikipedia_search**: Look up encyclopedic or factual knowledge
- **calculator**: Perform mathematical calculations
- **get_current_datetime**: Get the current date and time

IMPORTANT RULES:
1. Always think step-by-step before answering.
2. Use tools when you need real-time or factual information.
3. Be concise but thorough in your responses.
4. If you are unsure, say so and use a tool to verify.
5. After using a tool, analyze the result and respond helpfully.
6. You can use multiple tools in sequence if needed.
""")


# ------------------------------------------------------------------
# Global LLM instance with tools bound
# ------------------------------------------------------------------
_llm_with_tools = None


def get_llm_with_tools():
    """Lazy-loads and returns the LLM bound with tools."""
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = load_llm()
        _llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


# ------------------------------------------------------------------
# Node 1: Agent Node (The Brain — calls the LLM)
# ------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """
    The main agent node. This is where the LLM thinks and decides:
    - Should I answer directly?
    - Should I call a tool?

    The LLM sees the full conversation history + system prompt.
    If it decides to use a tool, it returns a ToolCall in the message.
    LangGraph's router then sends it to the tools node.
    """
    llm_with_tools = get_llm_with_tools()

    # Prepend system message to the conversation
    messages_with_system = [SYSTEM_PROMPT] + state["messages"]

    # Invoke the LLM
    response = llm_with_tools.invoke(messages_with_system)

    # Return the AI's response message to be added to state
    return {"messages": [response]}


# ------------------------------------------------------------------
# Node 2: Tools Node (The Hands — executes tool calls)
# ------------------------------------------------------------------
# ToolNode is a built-in LangGraph node that:
# 1. Looks at the last AI message for tool_calls
# 2. Executes each tool with the provided arguments
# 3. Returns ToolMessage(s) with results back into the state
tools_node = ToolNode(tools=ALL_TOOLS)


# ------------------------------------------------------------------
# Router: Decides whether to call tools or stop
# ------------------------------------------------------------------
def should_use_tools(state: AgentState) -> str:
    """
    Conditional edge function — the router.
    
    Checks the last message in state:
    - If it has tool_calls → route to "tools" node
    - Otherwise → route to END (reply is ready)
    
    Returns:
        "tools" or "end"
    """
    last_message = state["messages"][-1]

    # Check if the LLM wants to call any tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # No tool calls — the agent is done
    return "end"