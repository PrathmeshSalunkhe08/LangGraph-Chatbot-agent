# 🤖 Stateful LangGraph Chatbot with SQLite Database Memory

A complete, persistent multi-thread AI chatbot system built with **LangGraph**, **Groq (Llama 3.1 8B Instant)**, **SQLite Persistence**, and **Streamlit UI**.

---

## ✨ Features

- **🧠 Persistent Memory Across Sessions**: Every conversation thread is recorded into an SQLite database (`chatbot.db`) using `SqliteSaver`.
- **🔀 Multi-Thread Support**: Seamlessly create new chat sessions, switch between existing threads, or delete unwanted conversations.
- **⚡ Real-Time Streaming**: Stream response tokens directly to the Streamlit UI with `chatbot.stream(...)`.
- **🎨 Modern Dark UI**: Enhanced with glassmorphism style headers, custom CSS layout, badges, formatted markdown rendering, and per-thread deletion controls.
- **🛡️ Fallback & Error Handling**: Graceful exception catching for API limits and network state.

---

## 🏗️ Architecture

- **Backend (`langgraph_database_backend.py`)**:
  - Defines `ChatState` with `Annotated[list[BaseMessage], add_messages]`.
  - Configures `StateGraph` compiled with `SqliteSaver` checkpointer.
  - Exposes thread management functions: `retrive_all_threads()`, `get_thread_preview()`, and `delete_thread()`.
- **Frontend (`langgraph_database_frontend.py`)**:
  - Interactive Streamlit app with session state management.
  - Multi-conversation sidebar with live topic preview titles.
  - Streaming output via `st.write_stream`.

---

## 🚀 How to Run

### 1. Requirements & Setup
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Set API Key
Ensure `.env` contains your Groq API Key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Streamlit Application
Launch the web interface:
```bash
streamlit run langgraph_database_frontend.py
```
Open your browser at `http://localhost:8501`.
