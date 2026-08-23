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

import math

# Define Useful Agent Tools
@tool
def scientific_calculator_tool(expression: str) -> str:
    """Use this tool for ANY mathematical or scientific calculations (arithmetic, trigonometry, logarithms, exponentials, factorials, square roots, geometry).
    Available functions: sin, cos, tan, asin, acos, atan, radians, degrees, log, log10, log2, exp, sqrt, factorial, comb, perm, gcd, pi, e, pow, abs, round.
    Examples: 'sin(radians(30)) + log10(100)', 'factorial(10) / sqrt(144)', 'pi * pow(5, 2)'."""
    try:
        safe_scope = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
        safe_scope.update({
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        })
        result = eval(expression, {"__builtins__": None}, safe_scope)
        if isinstance(result, float):
            result = round(result, 8)
        return f"Scientific Calculation Output: {result}"
    except Exception as e:
        return f"Calculation Error: {e}. Use standard math syntax e.g., sin(radians(30)), sqrt(16), log10(100), factorial(5), pi, pow(x, y)."

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
def get_current_time_tool(query: str) -> str:
    """Use this tool to get the current date and time."""
    from datetime import datetime
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# List of tools and binding to LLM
tools = [scientific_calculator_tool, workspace_file_reader, get_current_time_tool]
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