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

import requests
import yfinance as yf

@tool
def stock_crypto_price_tool(symbol_or_name: str) -> str:
    """Use this tool to get the live price, daily high/low, and market summary for stocks (e.g. AAPL, TSLA, MSFT, RELIANCE.NS) or Crypto (e.g. bitcoin, ethereum, solana, dogecoin)."""
    clean = symbol_or_name.strip().lower()
    
    # 1. Try Crypto price check via CoinGecko API
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={clean}&vs_currencies=usd&include_24hr_change=true"
        res = requests.get(url, timeout=5).json()
        if clean in res:
            price = res[clean]['usd']
            change = res[clean].get('usd_24h_change', 0.0)
            return f"Live Crypto ({clean.upper()}): ${price:,.2f} USD (24h Change: {change:+.2f}%)"
    except Exception:
        pass
        
    # 2. Try Stock ticker price check via Yahoo Finance
    try:
        ticker = yf.Ticker(symbol_or_name.upper())
        hist = ticker.history(period="1d")
        if not hist.empty:
            last_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[-1]
            high_price = hist['High'].iloc[-1]
            low_price = hist['Low'].iloc[-1]
            return f"Live Stock ({symbol_or_name.upper()}): Current Price: ${last_price:.2f} USD | Open: ${open_price:.2f} | High: ${high_price:.2f} | Low: ${low_price:.2f}"
    except Exception as e:
        return f"Market Data Error for '{symbol_or_name}': {e}"
        
    return f"Could not find live market data for symbol '{symbol_or_name}'. Try using stock tickers like AAPL, TSLA, NVDA or crypto names like bitcoin, ethereum."

@tool
def google_serper_search_tool(query: str) -> str:
    """Use this tool to search Google for live news, real-time events, current sports scores, documentation, and facts."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key or not api_key.strip():
        return "Google Serper API key not configured in .env."
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query}
        res = requests.post(url, headers=headers, json=payload, timeout=8).json()
        
        results = []
        if "answerBox" in res and "snippet" in res["answerBox"]:
            results.append(f"Direct Answer: {res['answerBox']['snippet']}")
            
        organic = res.get("organic", [])[:4]
        for item in organic:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append(f"• [{title}]({link}): {snippet}")
            
        return "\n".join(results) if results else "No Google search results found."
    except Exception as e:
        return f"Google Serper Search Error: {e}"

@tool
def weather_forecast_tool(city: str) -> str:
    """Use this tool to get live weather forecast, temperature (°C & °F), weather conditions, humidity, and wind speed for any city in the world (e.g. Mumbai, Delhi, London, Tokyo, New York)."""
    clean_city = city.strip()
    try:
        url = f"https://wttr.in/{clean_city}?format=j1"
        res = requests.get(url, timeout=6).json()
        curr = res['current_condition'][0]
        area = res['nearest_area'][0]['areaName'][0]['value']
        country = res['nearest_area'][0]['country'][0]['value']
        temp_c = curr['temp_C']
        temp_f = curr['temp_F']
        feels_like = curr['FeelsLikeC']
        desc = curr['weatherDesc'][0]['value']
        humidity = curr['humidity']
        wind = curr['windspeedKmph']
        
        return (
            f"Live Weather for {area}, {country}:\n"
            f"• Temperature: {temp_c}°C ({temp_f}°F) [Feels like: {feels_like}°C]\n"
            f"• Condition: {desc}\n"
            f"• Humidity: {humidity}%\n"
            f"• Wind Speed: {wind} km/h"
        )
    except Exception as e:
        return f"Weather Fetch Error for city '{city}': {e}. Please check the city name."

@tool
def get_current_time_tool(query: str) -> str:
    """Use this tool to get the current date and time."""
    from datetime import datetime
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# List of tools and binding to LLM
tools = [scientific_calculator_tool, stock_crypto_price_tool, weather_forecast_tool, google_serper_search_tool, workspace_file_reader, get_current_time_tool]
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