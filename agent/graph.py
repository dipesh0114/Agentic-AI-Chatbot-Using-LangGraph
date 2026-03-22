# agent/graph.py
# ============================================================
# LANGGRAPH GRAPH CONSTRUCTION
# ============================================================
# This is where we wire everything together:
#   - Define the graph with our AgentState
#   - Add nodes (agent, tools)
#   - Add edges (linear and conditional)
#   - Compile with memory checkpointer
#
# THE FLOW:
#
#   START
#     ↓
#  [agent_node]  ← LLM thinks, decides if tools needed
#     ↓
#  (router) ─── tool_calls? ──YES──► [tools_node] ─► back to [agent_node]
#                                                              ↕ (loop)
#              NO (done)
#                ↓
#              END
# ============================================================

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import agent_node, tools_node, should_use_tools

# Load .env file automatically
load_dotenv()


def build_graph():
    """
    Builds and compiles the LangGraph agentic chatbot graph.

    Returns:
        A compiled LangGraph graph with:
        - Agent node (LLM)
        - Tools node (tool executor)
        - Conditional routing (ReAct loop)
        - In-memory persistence (conversation memory)
    """

    # ----------------------------------------------------------
    # 1. Initialize the StateGraph with our custom state schema
    # ----------------------------------------------------------
    graph_builder = StateGraph(AgentState)

    # ----------------------------------------------------------
    # 2. Add Nodes
    # ----------------------------------------------------------
    # "agent" → calls the LLM to decide next action
    graph_builder.add_node("agent", agent_node)

    # "tools" → executes whichever tool the agent selected
    graph_builder.add_node("tools", tools_node)

    # ----------------------------------------------------------
    # 3. Add Edges (define the flow)
    # ----------------------------------------------------------
    # START → agent: Every conversation starts at the agent node
    graph_builder.add_edge(START, "agent")

    # agent → conditional: Does the agent want to call a tool?
    graph_builder.add_conditional_edges(
        source="agent",          # From: agent node
        path=should_use_tools,   # Router function
        path_map={
            "tools": "tools",    # Yes → go to tools node
            "end": END,          # No  → conversation is done
        }
    )

    # tools → agent: After tool runs, go back to agent to process result
    # This creates the ReAct loop: Think → Act → Observe → Think → ...
    graph_builder.add_edge("tools", "agent")

    # ----------------------------------------------------------
    # 4. Compile with Memory (MemorySaver = in-memory SQLite)
    # ----------------------------------------------------------
    # MemorySaver stores the full conversation state per thread_id.
    # This gives the bot persistent memory within a session.
    # For multi-session persistence, swap with SqliteSaver.
    memory = MemorySaver()

    compiled_graph = graph_builder.compile(checkpointer=memory)

    print("[Graph] Agentic chatbot graph compiled successfully.")
    return compiled_graph


# ----------------------------------------------------------
# Singleton graph instance (reused across requests)
# ----------------------------------------------------------
_graph = None


def get_graph():
    """Returns the singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ----------------------------------------------------------
# Helper: Chat with the agent
# ----------------------------------------------------------
def chat(user_message: str, thread_id: str = "default") -> str:
    """
    Send a message to the agent and get a response.

    Args:
        user_message: The user's text input.
        thread_id: Unique session/thread ID for memory isolation.
                   Same thread_id = same conversation memory.

    Returns:
        The agent's final response as a string.
    """
    from langchain_core.messages import HumanMessage

    graph = get_graph()

    # Config identifies the conversation thread for memory
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )

    # The last message in state is the agent's final response
    final_message = result["messages"][-1]
    return final_message.content