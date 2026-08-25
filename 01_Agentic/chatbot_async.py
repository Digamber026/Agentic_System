import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
 
load_dotenv()
 
# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")
 
# Define Tool
@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Performs basic mathematical operations like add, subtract, multiply, divide, power, and modulus."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    elif operation == "power":
        return a ** b
    elif operation == "modulus":
        return a % b
    else:
        raise ValueError("Invalid operation")
 
tools = [calculator]
llm_with_tools = llm.bind_tools(tools)
 
# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]
 
# Async Chat Node
async def chat_node(state: State):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}
 
tool_node = ToolNode(tools)
 
# Graph Builder Function
def build_graph():
    builder = StateGraph(State)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", tool_node)
 
    builder.add_edge(START, "chat_node")
    builder.add_conditional_edges("chat_node", tools_condition)
    builder.add_edge("tools", "chat_node")
 
    return builder.compile()
 
# Async Execution
async def main():
    chatbot = build_graph()
    prompt = "Find the modulus of 123456 and 789 and present the result like a cricket commentator."
    response = await chatbot.ainvoke({"messages": [("user", prompt)]})
    print(response["messages"][-1].content)
 
if __name__ == "__main__":
    asyncio.run(main())