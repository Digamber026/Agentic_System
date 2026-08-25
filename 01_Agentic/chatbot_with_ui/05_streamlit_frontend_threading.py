import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import importlib.util
import pathlib

_backend_path = pathlib.Path(__file__).parent / '03_langgraph_tool_backend.py'
_spec = importlib.util.spec_from_file_location('langgraph_db_backend', _backend_path)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)
chatbot = _backend.chatbot
retrieve_all_threads = _backend.retrieve_all_threads

# utility function
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config= {'configurable':{'thread_id': thread_id}}).values.get('messages', [])

def get_thread_title(thread_id):
    messages = load_conversation(thread_id)
    first_user_msg = next((msg.content for msg in messages if isinstance(msg, HumanMessage)), None)
    if not first_user_msg:
        return 'New Chat'
    title = first_user_msg.strip().splitlines()[0]
    return title[:30] + '...' if len(title) > 30 else title

# session setup
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# sidebar ui
st.sidebar.title('LangGraph chatBot')

if st.sidebar.button('new chat'):
    reset_chat()

st.sidebar.header('My Conversation')

# list threads newest-first; clicking one switches the active session
for thread_id in st.session_state['chat_threads'][::-1]:
    # show a readable title (like ChatGPT) instead of the raw thread_id
    if st.sidebar.button(get_thread_title(thread_id), key=str(thread_id)):
        # make this thread the active session
        st.session_state['thread_id'] = thread_id
        # pull that session's saved messages back from the chatbot's checkpoint
        messages = load_conversation(thread_id)

        temp_messages = []

        # convert LangChain message objects into the {role, content} dicts the UI renders
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else :
                role = 'assistant'
            temp_messages.append({'role' : role, 'content' : msg.content})

        st.session_state['message_history'] = temp_messages

# always reflect the currently selected thread/session
# CONFIG = {'configurable':{'thread_id': st.session_state['thread_id']}}

CONFIG = {
'configurable':{'thread_id': st.session_state['thread_id']},
'metadata':{
    'thread_id':st.session_state['thread_id']
},
'run_name':'chat_turn'
}

# main ui (loading the coversation history)
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type_here')

if user_input:

    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

      # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})