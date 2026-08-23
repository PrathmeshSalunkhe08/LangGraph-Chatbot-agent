import streamlit as st
from langgraph_database_backend import chatbot, retrive_all_threads, get_thread_preview, delete_thread, search_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os

# Path to custom avatars
BOT_AVATAR_PATH = "bot_avatar.png" if os.path.exists("bot_avatar.png") else "🤖"
USER_AVATAR_PATH = "user_avatar.png" if os.path.exists("user_avatar.png") else ("user_avatar.jpg" if os.path.exists("user_avatar.jpg") else "👤")


# -----------------------------------------------------------------------------
# 1. Page Configuration & Aesthetic Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphMind AI",
    page_icon=BOT_AVATAR_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cyberpunk Emerald & Neon Violet Aesthetic CSS (Phase 2 Polish)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Main container background */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #050811 75%) !important;
        color: #f8fafc !important;
    }

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }
    
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-top: 1.2rem;
    }

    /* Sidebar Headers */
    .sidebar-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34d399 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    
    .sidebar-caption {
        font-size: 0.78rem;
        color: #94a3b8;
        letter-spacing: 0.02em;
        font-weight: 500;
    }

    /* Main Workspace Header */
    .main-header-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 15, 25, 0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 1.4rem 2rem;
        border-radius: 18px;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34d399 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .main-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 0.3rem;
        font-weight: 500;
        font-style: italic;
    }

    /* Tech Badges */
    .tech-badge-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid rgba(52, 211, 153, 0.35);
        display: inline-block;
        margin-right: 8px;
        letter-spacing: 0.03em;
    }

    .tech-badge-violet {
        background: rgba(139, 92, 246, 0.12);
        color: #c084fc;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid rgba(192, 132, 252, 0.35);
        display: inline-block;
        margin-right: 8px;
        letter-spacing: 0.03em;
    }

    /* Hero state card */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(139, 92, 246, 0.08) 100%);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 18px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        max-width: 580px;
        margin: 0 auto;
    }

    /* Streamlit Chat Messages Override with Hover/Glow Effects */
    div[data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 1.1rem 1.4rem !important;
        margin-bottom: 1.1rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    }

    /* User Message Bubble (Student Avatar + Violet Accent) */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        background: rgba(17, 24, 39, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.35) !important;
        border-left: 5px solid #8b5cf6 !important;
    }

    /* Assistant Message Bubble (Robot Avatar + Teal Accent) */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
        background: rgba(6, 78, 59, 0.25) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        border-left: 5px solid #10b981 !important;
    }

    div[data-testid="stChatMessage"] p, 
    div[data-testid="stChatMessage"] li {
        color: #f8fafc !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
    }

    /* Sidebar Buttons */
    div[data-testid="stSidebar"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Secondary Sidebar Conversation Item */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #cbd5e1 !important;
        text-align: left !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(16, 185, 129, 0.18) !important;
        color: #34d399 !important;
        border-color: #10b981 !important;
        transform: translateY(-1px);
    }

    /* Active Conversation Button */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: 1px solid #a78bfa !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.35) !important;
    }

    /* Delete Button Styling */
    div[data-testid="stSidebar"] button[help="Delete thread"] {
        padding: 4px 8px !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #fca5a5 !important;
    }

    div[data-testid="stSidebar"] button[help="Delete thread"]:hover {
        background-color: #ef4444 !important;
        color: #ffffff !important;
        border-color: #ef4444 !important;
    }

    /* Modern Larger Chat Input Bar */
    div[data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 2px solid rgba(139, 92, 246, 0.4) !important;
        background-color: #0b0f19 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        padding: 4px 6px !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 18px rgba(139, 92, 246, 0.3) !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        font-size: 1.02rem !important;
    }
    
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }

    /* Neon Send Button */
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #10b981 0%, #8b5cf6 100%) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stChatInput"] button:hover {
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.6) !important;
        transform: scale(1.05) !important;
    }

    pre, code {
        background-color: #020617 !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Utility Functions
# -----------------------------------------------------------------------------

def generate_thread_id():
    """Generate unique UUID v4 string for conversation threads."""
    return str(uuid.uuid4())

def reset_chat():
    """Start a fresh chat thread session."""
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    st.session_state['message_history'] = []
    st.rerun()

def add_thread(thread_id):
    """Ensure thread_id string is recorded in session_state list."""
    tid = str(thread_id)
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if tid not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(tid)

def load_conversation(thread_id):
    """Load conversation messages snapshot from LangGraph SQLite checkpointer."""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
        return state.values.get('messages', [])
    except Exception as e:
        st.error(f"Error loading conversation state: {e}")
        return []


# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    db_threads = retrive_all_threads()
    st.session_state['chat_threads'] = db_threads if db_threads else []

if 'thread_id' not in st.session_state:
    if st.session_state['chat_threads']:
        st.session_state['thread_id'] = st.session_state['chat_threads'][-1]
    else:
        new_id = generate_thread_id()
        st.session_state['thread_id'] = new_id
        add_thread(new_id)

add_thread(st.session_state['thread_id'])

# Sync message history for initial active thread if empty
if not st.session_state['message_history'] and st.session_state['thread_id']:
    initial_msgs = load_conversation(st.session_state['thread_id'])
    formatted = []
    for msg in initial_msgs:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        formatted.append({'role': role, 'content': msg.content})
    st.session_state['message_history'] = formatted


# -----------------------------------------------------------------------------
# 4. Sidebar UI
# -----------------------------------------------------------------------------

with st.sidebar:
    sb_col1, sb_col2 = st.columns([0.28, 0.72])
    with sb_col1:
        if os.path.exists("bot_avatar.png"):
            st.image("bot_avatar.png", width=55)
        else:
            st.markdown("🤖")
    with sb_col2:
        st.markdown('<div class="sidebar-brand">GraphMind AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-caption">Conversations that remember.</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # New Chat Primary Button
    if st.button("✨ New Chat", use_container_width=True, type="primary"):
        reset_chat()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #f8fafc; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem;'>🔍 Conversations</h4>", unsafe_allow_html=True)

    # Search bar input for real-time conversation filtering
    search_query = st.text_input("Search", placeholder="Search by name or content...", key="sidebar_search", label_visibility="collapsed")
    
    # Filter threads in real-time
    all_threads_reversed = st.session_state['chat_threads'][::-1]
    
    if search_query and search_query.strip():
        matching_tids = set(search_threads(search_query))
        threads_list = [tid for tid in all_threads_reversed if tid in matching_tids]
        st.caption(f"Found **{len(threads_list)}** matching conversation(s)")
    else:
        threads_list = all_threads_reversed

    # Display threads list with clean spacing
    if not threads_list:
        if search_query and search_query.strip():
            st.warning(f"No conversations match '{search_query}'.")
        else:
            st.info("No saved conversations yet.")
    else:
        for tid in threads_list:
            is_active = (tid == st.session_state['thread_id'])
            preview_title = get_thread_preview(tid)
            
            # Format display label with search match indicator
            icon = "⚡ " if is_active else ("🔍 " if search_query else "💬 ")
            label = f"{icon}{preview_title}"
            
            col1, col2 = st.columns([0.83, 0.17])
            with col1:
                btn_kind = "primary" if is_active else "secondary"
                if st.button(label, key=f"btn_{tid}", use_container_width=True, type=btn_kind):
                    st.session_state['thread_id'] = tid
                    msgs = load_conversation(tid)
                    temp_messages = []
                    for msg in msgs:
                        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                        temp_messages.append({'role': role, 'content': msg.content})
                    st.session_state['message_history'] = temp_messages
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"del_{tid}", help="Delete thread"):
                    delete_thread(tid)
                    st.session_state['chat_threads'].remove(tid)
                    if st.session_state['thread_id'] == tid:
                        if st.session_state['chat_threads']:
                            st.session_state['thread_id'] = st.session_state['chat_threads'][-1]
                            msgs = load_conversation(st.session_state['thread_id'])
                            st.session_state['message_history'] = [
                                {'role': 'user' if isinstance(m, HumanMessage) else 'assistant', 'content': m.content}
                                for m in msgs
                            ]
                        else:
                            reset_chat()
                    st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Active User Profile Card
    u_col1, u_col2 = st.columns([0.28, 0.72])
    with u_col1:
        if USER_AVATAR_PATH != "👤":
            st.image(USER_AVATAR_PATH, width=46)
        else:
            st.markdown("👤")
    with u_col2:
        st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#f8fafc;'>Active User</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem; color:#34d399;'>● Online Session</div>", unsafe_allow_html=True)

    with st.expander("⚙️ System Architecture", expanded=False):
        st.markdown("**LLM Provider:** Groq (Compound-Mini)")
        st.markdown("**Checkpoint Engine:** SqliteSaver (`chatbot.db`)")
        st.markdown(f"**Current Thread:**\n`{st.session_state['thread_id'][:18]}...`")


# -----------------------------------------------------------------------------
# 5. Main Workspace Header
# -----------------------------------------------------------------------------

hdr_col1, hdr_col2 = st.columns([0.09, 0.91])
with hdr_col1:
    if os.path.exists("bot_avatar.png"):
        st.image("bot_avatar.png", width=68)
    else:
        st.markdown("🤖")
with hdr_col2:
    st.markdown(f"""
    <div class="main-header-card">
        <div>
            <div class="main-title">GraphMind AI Assistant</div>
            <div class="main-subtitle">Stateful conversations powered by LangGraph</div>
            <div style="margin-top: 10px;">
                <span class="tech-badge-emerald">LangGraph</span>
                <span class="tech-badge-violet">SQLite Memory</span>
                <span class="tech-badge-emerald">Groq Compound Mini</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Main Chat Area & Hero State
# -----------------------------------------------------------------------------

# Render Hero Banner if session history is empty
if not st.session_state['message_history']:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    hero_col1, hero_col2, hero_col3 = st.columns([0.42, 0.16, 0.42])
    with hero_col2:
        if os.path.exists("bot_avatar.png"):
            st.image("bot_avatar.png", use_container_width=True)
    st.markdown("""
        <div class="hero-title">👋 Welcome to GraphMind AI</div>
        <div class="hero-subtitle">Conversations that remember. Ask questions, test context memory across multiple turns, or start a new thread anytime!</div>
    </div>
    """, unsafe_allow_html=True)

# Render conversation message history
for message in st.session_state['message_history']:
    msg_avatar = BOT_AVATAR_PATH if message['role'] == 'assistant' else USER_AVATAR_PATH
    with st.chat_message(message['role'], avatar=msg_avatar):
        st.markdown(message['content'])

# User message input bar
user_input = st.chat_input("Message GraphMind AI...")

if user_input:
    # 1. Store and display user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user', avatar=USER_AVATAR_PATH):
        st.markdown(user_input)

    # 2. Config linking LangGraph checkpoint memory to active thread_id
    config = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # 3. Stream AI assistant response
    with st.chat_message("assistant", avatar=BOT_AVATAR_PATH):
        def ai_stream_generator():
            try:
                for chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
                ):
                    if isinstance(chunk, AIMessage) and chunk.content:
                        yield chunk.content
            except Exception as e:
                yield f"⚠️ **Error generating response:** {str(e)}"

        ai_response = st.write_stream(ai_stream_generator())

    # 4. Save response to session state
    if ai_response:
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_response})
    
    # 5. Rerun to refresh sidebar title preview if this was the first message
    if len(st.session_state['message_history']) <= 2:
        st.rerun()
