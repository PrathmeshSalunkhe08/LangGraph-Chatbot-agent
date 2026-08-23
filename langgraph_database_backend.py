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

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
}

@tool
def weather_forecast_tool(city: str) -> str:
    """Use this tool to get 100% accurate, official meteorological live weather forecast, exact temperature (°C & °F), weather conditions, humidity, and wind speed for any city in the world (e.g. Mumbai, Delhi, London, Tokyo, New York)."""
    clean_city = city.strip()
    try:
        # 1. Geocode city name to exact coordinates using Open-Meteo Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_city}&count=1"
        geo_res = requests.get(geo_url, timeout=5).json()
        if not geo_res.get("results"):
            return f"Location '{city}' not found. Please check city spelling."
            
        loc = geo_res["results"][0]
        name = loc.get("name", "")
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")
        lat, lon = loc["latitude"], loc["longitude"]
        
        # 2. Fetch official meteorological weather data from Open-Meteo Radar
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature"
        w_res = requests.get(w_url, timeout=5).json()
        curr = w_res["current"]
        
        temp_c = curr["temperature_2m"]
        temp_f = round((temp_c * 9/5) + 32, 1)
        feels_c = curr.get("apparent_temperature", temp_c)
        humidity = curr["relative_humidity_2m"]
        wind = curr["wind_speed_10m"]
        code = curr.get("weather_code", 0)
        condition = WEATHER_CODES.get(code, "Clear/Partly Cloudy")
        
        loc_str = f"{name}, {admin1}, {country}" if admin1 else f"{name}, {country}"
        
        return (
            f"Official Weather Data for {loc_str}:\n"
            f"• Temperature: {temp_c}°C ({temp_f}°F) [Feels like: {feels_c}°C]\n"
            f"• Condition: {condition}\n"
            f"• Relative Humidity: {humidity}%\n"
            f"• Wind Speed: {wind} km/h"
        )
    except Exception as e:
        return f"Weather Fetch Error for city '{city}': {e}."

LANG_MAP = {
    "hindi": "hi", "marathi": "mr", "spanish": "es", "french": "fr",
    "german": "de", "japanese": "ja", "chinese": "zh", "russian": "ru",
    "arabic": "ar", "portuguese": "pt", "italian": "it", "gujarati": "gu", "bengali": "bn"
}

@tool
def language_translator_tool(text_and_target_language: str) -> str:
    """Use this tool to translate text into target languages (e.g. Hindi, Marathi, German, Spanish, French, Japanese, etc.).
    Input format string: 'text_to_translate | target_language' (e.g., 'Welcome to GraphMind AI | Hindi' or 'Welcome to GraphMind AI | German')."""
    try:
        if "|" in text_and_target_language:
            parts = text_and_target_language.split("|")
            text = parts[0].strip()
            target_lang = parts[1].strip().lower()
        else:
            text = text_and_target_language.strip()
            target_lang = "hindi"
            
        lang_code = LANG_MAP.get(target_lang, target_lang[:2])
        url = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(text)}&langpair=en|{lang_code}"
        res = requests.get(url, timeout=6).json()
        translated = res.get("responseData", {}).get("translatedText", text)
        
        return f"Translation ({target_lang.capitalize()}): {translated}"
    except Exception as e:
        return f"Translation Error: {e}"

@tool
def get_current_time_tool(query: str) -> str:
    """Use this tool to get the current date and time."""
    from datetime import datetime
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# List of tools and binding to LLM
tools = [scientific_calculator_tool, stock_crypto_price_tool, weather_forecast_tool, language_translator_tool, google_serper_search_tool, workspace_file_reader, get_current_time_tool]
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