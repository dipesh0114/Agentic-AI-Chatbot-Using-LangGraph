from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
 
 
class AgentState(TypedDict):
    """
    The central state object for our Agentic Chatbot.
 
    Fields:
        messages: Full conversation history.
                  'add_messages' is a reducer that APPENDS new
                  messages instead of replacing the whole list.
                  This is key for multi-turn conversations.
    """
    messages: Annotated[list[BaseMessage], add_messages]