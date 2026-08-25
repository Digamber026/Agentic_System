import asyncio
import uuid
import importlib.util
import pathlib

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_backend_path = pathlib.Path(__file__).parent / '04_langgraph_mcp_backend.py'
_spec = importlib.util.spec_from_file_location('langgraph_mcp_backend', _backend_path)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)
chatbot = _backend.chatbot
retrieve_all_threads = _backend.retrieve_all_threads


def sync_from_async_gen(agen):
    """Drive an async generator to completion on a throwaway event loop so
    it can be consumed by st.write_stream, which needs a plain generator."""
    loop = asyncio.new_event_loop()
    try:
        aiterator = agen.__aiter__()
        while True:
            try:
                yield loop.run_until_complete(aiterator.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    # AsyncSqliteSaver's sync get_state() only works from a thread other than
    # the one that created it; use the async interface instead (aget_state),
    # driven on a fresh event loop, same as retrieve_all_threads() does.
    async def _aget_state():
        state = await chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])

    return asyncio.run(_aget_state())


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            agen = chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
            for message_chunk, metadata in sync_from_async_gen(agen):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )