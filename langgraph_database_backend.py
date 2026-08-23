from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, Annotated
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3


# Initialize Groq Compound Mini (0.6s Ultra Fast Speed + High 70,000 Token Limit)
llm = ChatGroq(
    model="groq/compound-mini",
    temperature=0.7
)

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

SYSTEM_PROMPT = SystemMessage(
    content="You are GraphMind AI assistant. Respond in clean Markdown. Do NOT include literal HTML tags like <br> or <br/> inside tables or text lists."
)

# Define state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_core.messages import AIMessage

# Define conversation node with smart context trimming (guarantees < 1500 tokens per request)
def Chat_node(state: ChatState):
    raw_messages = state['messages']
    # Select last 6 messages
    recent = raw_messages[-6:] if len(raw_messages) > 6 else raw_messages
    
    # Trim individual old message text if too long to prevent TPM rate limits
    trimmed_msgs = []
    for m in recent:
        content_str = str(m.content)
        if len(content_str) > 800:
            content_str = content_str[:800] + "... [context truncated]"
        if isinstance(m, HumanMessage):
            trimmed_msgs.append(HumanMessage(content=content_str))
        else:
            trimmed_msgs.append(AIMessage(content=content_str))

    input_msgs = [SYSTEM_PROMPT] + trimmed_msgs
    response = llm.invoke(input_msgs)
    if isinstance(response.content, str):
        response.content = response.content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return {"messages": [response]}

# Setup SQLite persistence
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False) 
check_pointer = SqliteSaver(conn=conn)              

# Build LangGraph graph
graph = StateGraph(ChatState)
graph.add_node("Chat_node", Chat_node)
graph.add_edge(START, "Chat_node")
graph.add_edge("Chat_node", END)

# Compile chatbot graph with SQLite checkpointer
chatbot = graph.compile(checkpointer=check_pointer)

def retrive_all_threads():
    """Retrieve list of unique thread IDs stored in SQLite database."""
    all_threads = set()
    temp = check_pointer.list(None)
    for thread in temp:
        tid = str(thread.config['configurable']['thread_id'])
        all_threads.add(tid)
    return list(all_threads)

def get_thread_preview(thread_id: str) -> str:
    """Retrieve first user message as preview title for sidebar display."""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
        messages = state.values.get('messages', [])
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                text = str(msg.content).strip().replace("\n", " ")
                return text[:28] + ("..." if len(text) > 28 else "")
    except Exception:
        pass
    return "New Conversation"

def delete_thread(thread_id: str):
    """Remove a thread and its memory from SQLite database."""
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return False

def search_threads(query: str) -> list[str]:
    """Search threads by checking if query text exists in any message of the thread."""
    if not query or not query.strip():
        return retrive_all_threads()
    
    q = query.strip().lower()
    matching_threads = []
    all_tids = retrive_all_threads()
    
    for tid in all_tids:
        try:
            state = chatbot.get_state(config={'configurable': {'thread_id': str(tid)}})
            messages = state.values.get('messages', [])
            for msg in messages:
                if msg.content and q in str(msg.content).lower():
                    matching_threads.append(tid)
                    break
        except Exception:
            continue
            
    return matching_threads