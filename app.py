# app.py
# ============================================================
# STREAMLIT WEB UI
# ============================================================
# Run with: streamlit run app.py
# ============================================================

import uuid
import streamlit as st
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Import graph after env is loaded
from agent.graph import chat, get_graph

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="🤖 Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .tool-badge {
        display: inline-block;
        background: #e8f4fd;
        border: 1px solid #b8d9f7;
        color: #1a73e8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic AI Chatbot</h1>
    <p>Powered by LangGraph · ReAct Agent · Multi-Tool Support</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # New Conversation Button
    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    # Available Tools Display
    st.subheader("🛠️ Available Tools")
    tools_info = {
        "🔍 Web Search": "Real-time internet search via DuckDuckGo",
        "📖 Wikipedia": "Encyclopedic knowledge lookup",
        "🧮 Calculator": "Mathematical calculations",
        "🕐 Date & Time": "Current date and time",
    }
    for tool_name, description in tools_info.items():
        with st.expander(tool_name):
            st.caption(description)

    st.divider()

    # Example Prompts
    st.subheader("💡 Try These")
    example_prompts = [
        "What is today's date?",
        "What is 15% of 4,250?",
        "Search the web for latest AI news",
        "Who was Albert Einstein? Use Wikipedia.",
        "What is LangGraph? Search the web.",
        "Calculate the compound interest on ₹50,000 at 8% for 3 years",
    ]
    for prompt in example_prompts:
        if st.button(f"▶ {prompt[:45]}...", key=prompt, use_container_width=True):
            st.session_state.pending_prompt = prompt

    st.divider()
    st.caption("Built with LangGraph · LangChain · Streamlit")

# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # Each session has a unique thread_id for isolated memory
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ------------------------------------------------------------------
# Display Chat History
# ------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------------
# Handle pending prompt from sidebar buttons
# ------------------------------------------------------------------
pending = st.session_state.pop("pending_prompt", None)

# ------------------------------------------------------------------
# Chat Input
# ------------------------------------------------------------------
user_input = st.chat_input("Ask me anything... I can search the web, do math, and more!")

# Use pending prompt if sidebar button was clicked
if pending:
    user_input = pending

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = chat(
                    user_message=user_input,
                    thread_id=st.session_state.thread_id
                )
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease check your API keys in `.env`."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ------------------------------------------------------------------
# Show Thread ID (for debugging)
# ------------------------------------------------------------------
with st.sidebar:
    st.caption(f"🔑 Session ID: `{st.session_state.thread_id[:8]}...`")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")