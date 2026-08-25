import os
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
 
# Define sample synchronous calculator tool
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
 
# Bind tool to LLM
tools = [calculator]
llm_with_tools = llm.bind_tools(tools)
 
# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]
 
# Define Nodes
def chat_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
 
tool_node = ToolNode(tools)
 
# Build Graph
builder = StateGraph(State)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)
 
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")
 
chatbot = builder.compile()
 
# Execution
if __name__ == "__main__":
    prompt = "Find the modulus of 123456 and 789 and present the result like a cricket commentator."
    response = chatbot.invoke({"messages": [("user", prompt)]})
    print(response["messages"][-1].content)