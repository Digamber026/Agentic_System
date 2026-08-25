import sqlite3 
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver 
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random

# 1. LLM 
load_dotenv()
llm = ChatOpenAI()

# 2. Tools 
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num:float, second_num:float, operation:str)->dict:
    '''
    perform a basic arithmetic operation on tow numbers.
    Supported operations:add, sub, mul, div
    '''
    try:
        if operation=='add':
            result = first_num + second_num
        elif operation=='sub':
            result = first_num - second_num
        elif operation =='mul':
            result = first_num * second_num
        elif operation =='div':
            if second_num == 0:
                return  {'error':'Division by zero is not allowed'}
            result = first_num / second_num
        else:
            return {'error':f"Unsupported operation '{operation}'"}

        return {'first_num':first_num, 'second_num':second_num, 'operation':operation, 'result':result}
    except Exception as e :
        return {'error':str(e)}


@tool
def get_stock_price(symbol:str) -> dict:
    '''
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    '''
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=86DI5WD8O1DEVQXN"
    r = requests.get(url)
    return r.json()

# Make tool list 
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# 3. state 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Nodes 

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)

    return {'messages':response}

tool_node = ToolNode(tools)

# 5. Checkpointer 

conn = sqlite3.connect(database = 'chatbot.db' , check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)


# 6. Graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# # test 
# CONFIG = {'configurable':{'thread_id': 'thread-1'}}

# response = chatbot.invoke(
#                 {'messages':[HumanMessage(content='what is my name  ')]},      # initialized state
#                 config = CONFIG,   # config                                
#             )
# print(response)


# 7. Helper
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
