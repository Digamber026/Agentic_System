import sqlite3
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver 
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage


load_dotenv()
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages':response}


conn = sqlite3.connect(database = 'chatbot.db' , check_same_thread = False)

#checkpointer
# checkpointer = InMemorySaver()
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# # test 
# CONFIG = {'configurable':{'thread_id': 'thread-1'}}

# response = chatbot.invoke(
#                 {'messages':[HumanMessage(content='what is my name  ')]},      # initialized state
#                 config = CONFIG,   # config                                
#             )
# print(response)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
