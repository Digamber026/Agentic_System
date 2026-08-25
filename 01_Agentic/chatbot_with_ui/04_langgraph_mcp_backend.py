import asyncio
import os
from typing import Annotated, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

mcp_client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": [os.path.join(THIS_DIR, "mcp-math-server", "main.py")],
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        },
    }
)


def load_mcp_tools() -> list[BaseTool]:
    """Fetch tools from each MCP server one at a time, skipping any server
    that's down instead of losing every tool because one server failed."""
    mcp_tools = []
    for server_name in mcp_client.connections:
        try:
            mcp_tools += asyncio.run(mcp_client.get_tools(server_name=server_name))
        except Exception as exc:
            print(f"[mcp] skipping '{server_name}': {exc}")
    return mcp_tools


tools = [search_tool, get_stock_price, *load_mcp_tools()]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call.

    Async because the MCP tools (arith/expense) only support async
    invocation, so ToolNode — and therefore this whole graph — must run
    through `.astream()`/`.ainvoke()` rather than the sync `.stream()`.
    """
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
# AsyncSqliteSaver (not the sync SqliteSaver) because the graph runs async.
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = asyncio.run(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return asyncio.run(_alist_threads())
