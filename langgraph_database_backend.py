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


# Initialize Groq LLM with native Tool Calling support (GPT-OSS-120B - 500k daily tokens, 0.8s speed)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_retries=3
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
    """Use this tool to search Google for live news, real-time events, current sports scores, documentation, biographies, and facts."""
    try:
        # Primary: 100% Free Unlimited Live Search via DDGS
        from ddgs import DDGS
        ddg_results = list(DDGS().text(query, max_results=5))
        if ddg_results:
            results = []
            for idx, item in enumerate(ddg_results, 1):
                title = item.get("title", "")
                body = item.get("body", "")
                results.append(f"Search Result {idx}: {title}\nDetails: {body}")
            return "\n\n".join(results)
    except Exception:
        pass

    # Fallback to Serper API if configured
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key or not api_key.strip():
        return "Live Web Search: No search results returned."
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "gl": "in", "hl": "en", "num": 5}
        res = requests.post(url, headers=headers, json=payload, timeout=3).json()
        
        results = []
        if "answerBox" in res and "snippet" in res["answerBox"]:
            results.append(f"Direct Answer: {res['answerBox']['snippet']}")
            
        organic = res.get("organic", [])[:5]
        for idx, item in enumerate(organic, 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            results.append(f"Result {idx}: {title}\nDetails: {snippet}")
            
        return "\n\n".join(results) if results else "No web search results found."
    except Exception as e:
        return f"Web Search Error: {e}"

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
        geo_res = requests.get(geo_url, timeout=3).json()
        if not geo_res.get("results"):
            return f"Location '{city}' not found. Please check city spelling."
            
        loc = geo_res["results"][0]
        name = loc.get("name", "")
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")
        lat, lon = loc["latitude"], loc["longitude"]
        
        # 2. Fetch official meteorological weather data from Open-Meteo Radar
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature"
        w_res = requests.get(w_url, timeout=3).json()
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
    Input format string: 'text_to_translate | target_language'."""
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
        res = requests.get(url, timeout=3).json()
        translated = res.get("responseData", {}).get("translatedText", text)
        
        return f"Translation ({target_lang.capitalize()}): {translated}"
    except Exception as e:
        return f"Translation Error: {e}"

@tool
def wikipedia_research_tool(query_topic: str) -> str:
    """Use this tool to search Wikipedia for deep encyclopedic summaries, historical facts, scientific concepts, biographies, and geographical details."""
    clean_topic = query_topic.strip().replace(" ", "_")
    headers = {"User-Agent": "GraphMindAI/1.0 (Educational Assistant)"}
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean_topic)}"
        res = requests.get(url, headers=headers, timeout=3).json()
        
        if res.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(query_topic)}&format=json"
            search_res = requests.get(search_url, headers=headers, timeout=3).json()
            search_items = search_res.get("query", {}).get("search", [])
            if search_items:
                first_title = search_items[0]["title"].replace(" ", "_")
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(first_title)}"
                res = requests.get(url, headers=headers, timeout=3).json()
            else:
                return f"No Wikipedia article found for '{query_topic}'."
                
        title = res.get("title", query_topic)
        extract = res.get("extract", "No summary available.")
        page_url = res.get("content_urls", {}).get("desktop", {}).get("page", "")
        
        return f"Wikipedia Summary: **{title}**\n\n{extract}\n\nRead more: {page_url}"
    except Exception as e:
        return f"Wikipedia Fetch Error for '{query_topic}': {e}"

@tool
def get_current_time_tool(query: str) -> str:
    """Use this tool to get the current date and time."""
    from datetime import datetime
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# List of tools and binding to LLM
tools = [scientific_calculator_tool, stock_crypto_price_tool, weather_forecast_tool, language_translator_tool, wikipedia_research_tool, google_serper_search_tool, workspace_file_reader, get_current_time_tool]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are GraphMind AI, an elite, highly intelligent, and articulate AI assistant matching ChatGPT-4o and Claude 3.5 Sonnet standards.\n\n"
        "### 🛠️ TOOL-SPECIFIC EXCELLENCE GUIDELINES:\n"
        "1. 🌐 **Live Web Search & News (`google_serper_search_tool`)**: Fetch live events, recent news, and biographies. Synthesize web findings cleanly.\n"
        "2. 📚 **Encyclopedia & History (`wikipedia_research_tool`)**: Retrieve accurate historical, scientific, and biographical data.\n"
        "3. 📈 **Crypto & Stocks (`stock_crypto_price_tool`)**: Present current prices, 24h market trends, market cap, and volume cleanly in bold metrics.\n"
        "4. 🌦️ **Live Weather (`weather_forecast_tool`)**: Display temperature, weather conditions, humidity, and wind speed in clear formatted summaries.\n"
        "5. 🧮 **Scientific Calculator (`scientific_calculator_tool`)**: Show step-by-step mathematical solutions and final results formatted cleanly.\n"
        "6. 🌐 **Language Translator (`language_translator_tool`)**: Deliver fluent, natural translations with cultural context.\n"
        "7. 📁 **Workspace File Reader (`workspace_file_reader`)**: Inspect local project files accurately when requested by the user.\n"
        "8. ⏰ **Current Time & Date (`get_current_time_tool`)**: Provide accurate real-time date and timestamp information.\n\n"
        "### 🌟 CORE ADAPTIVE RESPONSE PRINCIPLES:\n"
        "1. **NATURAL & EASY TO UNDERSTAND**: Communicate in clear, simple, human language. Break down complex topics using intuitive analogies, clean formatting, bold key phrases, and structured sections.\n"
        "2. **DYNAMIC ADAPTATION (Match User Intent Exactly)**:\n"
        "   - ⚡ **Conversational & Simple Queries** (e.g., 'hi', 'who is X', 'what is Y'): Give a direct, warm, conversational, and well-structured answer.\n"
        "   - 📌 **Summary & Brief Requests** (e.g., 'summary', 'in short', 'top 10', 'briefly'): Provide a concise 2-sentence overview followed by key bullet points.\n"
        "   - 🔍 **In-Depth & Detailed Requests** (e.g., 'in detail', 'explain thoroughly', 'full story', 'news report'): Deliver a comprehensive deep-dive analysis complete with background context, timelines, tables, and thorough section breakdowns.\n"
        "3. **DEFAULT LANGUAGE (ENGLISH DEFAULT)**: Always respond in clear, professional ENGLISH by default (even for Roman Hinglish/Hindi query phrases like 'Chatronki gunj pune'). Only respond in Hindi or Marathi if the user explicitly asks (e.g., 'IN HINDI', 'IN MARATHI', 'हिंदी में बताओ') or types directly in Devanagari script.\n"
        "4. **SEAMLESS TOOL INTEGRATION**: Synthesize search data and tool results naturally into elegant Markdown prose. Never expose raw tool syntax, internal function names, or JSON snippets to the user."
    )
)

# Define state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define conversation node
def Chat_node(state: ChatState):
    raw_messages = state['messages']
    
    # Preserve recent history including ToolMessages (up to last 6 messages)
    recent = raw_messages[-6:] if len(raw_messages) > 6 else raw_messages
    
    trimmed_msgs = []
    for m in recent:
        if isinstance(m, HumanMessage):
            content_str = str(m.content)
            if len(content_str) > 600:
                content_str = content_str[:600] + "... [truncated]"
            trimmed_msgs.append(HumanMessage(content=content_str))
        elif isinstance(m, AIMessage):
            content_str = str(m.content) if m.content else ""
            if len(content_str) > 600:
                content_str = content_str[:600] + "... [truncated]"
            # Strip past tool_calls payloads from history sent to LLM to prevent Groq parser exceptions
            if content_str:
                trimmed_msgs.append(AIMessage(content=content_str))
        elif isinstance(m, ToolMessage):
            # Include tool output as clean text context for the model
            content_str = str(m.content)
            if len(content_str) > 600:
                content_str = content_str[:600] + "... [truncated]"
            trimmed_msgs.append(HumanMessage(content=f"[Context from Search Tool]: {content_str}"))
        else:
            trimmed_msgs.append(m)

    input_msgs = [SYSTEM_PROMPT] + trimmed_msgs
    try:
        response = llm_with_tools.invoke(input_msgs)
    except Exception as e:
        response = AIMessage(content=f"⚠️ **Note:** {str(e)[:150]}. Please try again.")
        
    if isinstance(response.content, str):
        response.content = response.content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        
    # Guard against empty response content + empty tool calls
    if not response.content and not getattr(response, 'tool_calls', None):
        response.content = "I'm ready! How can I assist you with news, weather, stock prices, math, or research today?"
        
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