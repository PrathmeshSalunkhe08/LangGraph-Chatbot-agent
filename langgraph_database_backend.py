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


# Initialize Groq LLM with native Tool Calling support (0.8s Speed)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage

# Define Useful Agent Tools
@tool
def calculator_tool(expression: str) -> str:
    """Use this tool to calculate mathematical expressions precisely (e.g., 25 * 40 + 150)."""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"Calculated Result: {result}"
    except Exception as e:
        return f"Math Error: {e}"

@tool
def workspace_file_reader(filename: str) -> str:
    """Use this tool to read text or code files from the local project workspace (e.g., requirements.txt, .gitignore, README.md)."""
    try:
        if not os.path.exists(filename):
            return f"File '{filename}' not found."
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()[:2000]
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def database_inspector(query_type: str) -> str:
    """Use this tool to check live statistics and thread counts stored in the chatbot.db SQLite database."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT count(distinct thread_id) FROM checkpoints")
        count = cursor.fetchone()[0]
        return f"Database Statistics: Total active conversation threads in chatbot.db = {count}"
    except Exception as e:
        return f"Database error: {e}"

@tool
def get_current_time_tool(query: str) -> str:
    """Use this tool to get the current date and time."""
    from datetime import datetime
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# List of tools and binding to LLM
tools = [calculator_tool, workspace_file_reader, database_inspector, get_current_time_tool]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content="You are GraphMind AI assistant equipped with tools. Use your available tools when necessary to solve user queries accurately. Respond in clean Markdown."
)

# Define state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define conversation node
def Chat_node(state: ChatState):
    raw_messages = state['messages']
    recent = raw_messages[-6:] if len(raw_messages) > 6 else raw_messages
    
    trimmed_msgs = []
    for m in recent:
        if isinstance(m, HumanMessage):
            content_str = str(m.content)
            if len(content_str) > 800:
                content_str = content_str[:800] + "... [context truncated]"
            trimmed_msgs.append(HumanMessage(content=content_str))
        elif isinstance(m, AIMessage):
            content_str = str(m.content)
            if len(content_str) > 800:
                content_str = content_str[:800] + "... [context truncated]"
            trimmed_msgs.append(AIMessage(content=content_str, tool_calls=getattr(m, 'tool_calls', [])))
        else:
            trimmed_msgs.append(m)

    input_msgs = [SYSTEM_PROMPT] + trimmed_msgs
    response = llm_with_tools.invoke(input_msgs)
    if isinstance(response.content, str):
        response.content = response.content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return {"messages": [response]}

# Setup SQLite persistence
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False) 
check_pointer = SqliteSaver(conn=conn)              

# Build LangGraph graph with ToolNode and conditional edges
tool_node = ToolNode(tools=tools)

graph = StateGraph(ChatState)
graph.add_node("Chat_node", Chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "Chat_node")
graph.add_conditional_edges("Chat_node", tools_condition)
graph.add_edge("tools", "Chat_node")

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