# filename: tibb_ui_v2.py
import os
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from datetime import datetime

# ---------------------------
# Backend / Core logic: UNCHANGED
# (kept exactly as provided; UI uses it)
# ---------------------------

# Apply nest_asyncio patch
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for missing API keys
if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing. Please set it in your .env file.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please set it in your .env file.")
    st.stop()

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("pharm")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define system prompt template (unchanged)
system_prompt_template = """
Your name is CoMUI MB-2 Pharmacology Chatbot. You are a Professor specializing in Pharmacology in CoMUI. Answer questions very very elaborately and accurately. Use the following information to answer the user's question:

{doc_content}

Provide very brief accurate and helpful health response based on the provided information and your expertise.
"""

def generate_response(question):
    """Generate a response using Pinecone retrieval and Gemini 2.0 Flash."""
    # Create event loop for current thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Embed the user's question
    query_embed = embed_model.embed_query(question)
    query_embed = [float(val) for val in query_embed]  # Ensure standard floats
    
    # Query Pinecone for relevant documents - MODIFIED: top_k=3
    results = pinecone_index.query(
        vector=query_embed,
        top_k=5,  # CHANGED from 2 to 3
        include_values=False,
        include_metadata=True
    )
    
    # Extract document contents - MODIFIED: Added terminal printing
    doc_contents = []
    print("\n" + "="*50)
    print(f"RETRIEVED DOCUMENTS FOR: '{question}'")
    for i, match in enumerate(results.get('matches', [])):
        text = match['metadata'].get('text', '')
        doc_contents.append(text)
        print(f"\nDOCUMENT {i+1}:\n{text}\n")
    print("="*50 + "\n")
    
    doc_content = "\n".join(doc_contents).replace('{', '{{').replace('}', '}}') if doc_contents else "No additional information found."
    
    # Format the system prompt with retrieved content
    formatted_prompt = system_prompt_template.format(doc_content=doc_content)
    
    # Rebuild chat history from session state
    chat_history = ChatMessageHistory()
    for msg in st.session_state.chat_history:
        if msg.get("role") == "user":
            chat_history.add_user_message(msg.get("content"))
        elif msg.get("role") == "assistant":
            chat_history.add_ai_message(msg.get("content"))
    
    # Initialize memory with chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True
    )
    
    # Create the conversation prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(formatted_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    
    # Initialize Gemini 2.0 Flash model with explicit client
    chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.17,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create the conversation chain
    conversation = LLMChain(
        llm=chat,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Generate the response
    res = conversation({"question": question})
    
    return res.get('text', '')

# ---------------------------
# UI / UX - FUTURISTIC STYLING
# ---------------------------

# Page config
st.set_page_config(
    page_title="Tibb 1.0 — Pharmacology MB2 Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="🩺"
)

# Initialize UI session state items
if "chat_history" not in st.session_state:
    # Keep the same assistant greeting but add timestamp metadata for UI
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello Impeccabillem Warrior, I'm your CoMUI Pharmacology MB2 Assistant. How can I assist you today?", "ts": datetime.now().isoformat()}
    ]
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "sending" not in st.session_state:
    st.session_state.sending = False

# ---------- CSS + JS ----------
# CSS for futuristic UI. Light/Dark styles depend on `st.session_state.dark_mode`.
dark_css = """
:root{
  --bg:#0f1724;
  --card:#0b1220;
  --muted:#9aa6b2;
  --accent:#00e5a8;
  --accent2:#5eead4;
  --bubble-user:#112233;
  --bubble-assist:#072f3f;
  --glass: rgba(255,255,255,0.03);
}
body { background: linear-gradient(180deg, #071428 0%, #08192a 100%); color: #e6eef6; }
.header { font-family: 'Inter', sans-serif; }
.card { background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:16px; padding:14px; box-shadow: 0 6px 30px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.03); }
.chat-window { max-height:60vh; overflow:auto; padding:12px; border-radius:12px; }
.msg { padding:12px 16px; margin:8px 0; border-radius:14px; display:inline-block; max-width:78%; font-size:15px; line-height:1.4; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
.msg.user { background: linear-gradient(90deg,#042a2c, #033a66); color:#e6f7ff; border-bottom-right-radius:4px; float:right; text-align:right; }
.msg.assistant { background: linear-gradient(90deg,#072a3a,#0a5261); color:#e9fff8; border-bottom-left-radius:4px; float:left; text-align:left; }
.meta { font-size:11px; color:var(--muted); margin-top:6px; }
.avatar { width:36px; height:36px; border-radius:10px; display:inline-block; vertical-align:middle; margin-right:8px; }
.row { display:flex; align-items:flex-end; gap:10px; }
.clearfix::after { content: ''; clear: both; display: table; }
.footer-input { width:100%; border-radius:12px; padding:12px; font-size:15px; background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); color:inherit; }
.send-btn { padding:10px 16px; border-radius:12px; background: linear-gradient(90deg,var(--accent),var(--accent2)); color:#07202a; border:none; font-weight:600; cursor:pointer; box-shadow: 0 8px 20px rgba(0,230,168,0.12); }
.small { font-size:12px; color:var(--muted); }
.loader {
  width:46px;height:46px;border-radius:50%;background:conic-gradient(var(--accent), var(--accent2), transparent); animation:spin 1.2s linear infinite; margin:8px auto;
}
@keyframes spin { to { transform: rotate(360deg);} }
@media (max-width:720px){
  .chat-window { max-height:50vh; }
  .msg { font-size:14px; }
}
"""

light_css = """
:root{
  --bg:#f7fbff;
  --card:#ffffff;
  --muted:#556070;
  --accent:#0ea5b7;
  --accent2:#06b6d4;
  --bubble-user:#e6fffb;
  --bubble-assist:#eef8ff;
  --glass: rgba(0,0,0,0.03);
}
body { background: linear-gradient(180deg, #f8fbff 0%, #eef8ff 100%); color:#06283d; }
.header { font-family: 'Inter', sans-serif; }
.card { background: linear-gradient(135deg, rgba(0,0,0,0.02), rgba(255,255,255,0.03)); border-radius:14px; padding:14px; box-shadow: 0 6px 20px rgba(2,10,25,0.04); border: 1px solid rgba(2,10,25,0.03); }
.chat-window { max-height:60vh; overflow:auto; padding:12px; border-radius:12px; }
.msg { padding:12px 16px; margin:8px 0; border-radius:14px; display:inline-block; max-width:78%; font-size:15px; line-height:1.4; box-shadow: 0 2px 12px rgba(2,6,23,0.04); }
.msg.user { background: linear-gradient(90deg,#dffaf6,#c9fbff); color:#06222a; border-bottom-right-radius:4px; float:right; text-align:right; }
.msg.assistant { background: linear-gradient(90deg,#f0fbff,#e6f7ff); color:#06222a; border-bottom-left-radius:4px; float:left; text-align:left; }
.meta { font-size:11px; color:var(--muted); margin-top:6px; }
.avatar { width:36px; height:36px; border-radius:10px; display:inline-block; vertical-align:middle; margin-right:8px; }
.row { display:flex; align-items:flex-end; gap:10px; }
.clearfix::after { content: ''; clear: both; display: table; }
.footer-input { width:100%; border-radius:12px; padding:12px; font-size:15px; background: rgba(0,0,0,0.02); border:1px solid rgba(2,6,23,0.04); color:inherit; }
.send-btn { padding:10px 16px; border-radius:12px; background: linear-gradient(90deg,var(--accent),var(--accent2)); color:#ffffff; border:none; font-weight:600; cursor:pointer; box-shadow: 0 8px 20px rgba(6,182,212,0.08); }
.small { font-size:12px; color:var(--muted); }
.loader {
  width:46px;height:46px;border-radius:50%;background:conic-gradient(var(--accent), var(--accent2), transparent); animation:spin 1.2s linear infinite; margin:8px auto;
}
@keyframes spin { to { transform: rotate(360deg);} }
@media (max-width:720px){
  .chat-window { max-height:50vh; }
  .msg { font-size:14px; }
}
"""

# choose css depending on dark_mode
css = dark_css if st.session_state.dark_mode else light_css

# small JS to auto scroll to bottom of chat container and to safely keep focus
scroll_js = """
<script>
function scrollToBottom(){
  const cw = document.getElementById('chat-window');
  if(cw) cw.scrollTop = cw.scrollHeight;
}
setTimeout(scrollToBottom, 150);
</script>
"""

# ---------- Header & Sidebar ----------
# Create a two-column header: title + small controls
header_col1, header_col2 = st.columns([4,1])
with header_col1:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px">
          <div style="width:52px;height:52px;border-radius:12px;background:linear-gradient(90deg,#00e5a8,#5eead4);display:flex;align-items:center;justify-content:center;font-weight:700;color:#07202a;font-size:22px">
            🩺
          </div>
          <div>
            <div style="font-size:20px;font-weight:700">Tibb 1.0</div>
            <div style="font-size:12px;color: #9aa6b2">CoMUI MB-2 Pharmacology Assistant</div>
          </div>
        </div>
        """, unsafe_allow_html=True
    )
with header_col2:
    # theme toggle
    if st.button("Toggle Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
    st.write("")  # small spacer

# Sidebar (collapsible)
with st.sidebar:
    st.markdown("## About Tibb 1.0")
    st.markdown("Futuristic UI, backed by your ComUI MB-2 slides. Ask Pharmacology MB-2 questions and get evidence-based replies.")
    st.markdown("---")
    st.markdown("### How to use\n1. Type a clear question in the box below.\n2. Press **Send** or hit Enter.\n3. Use **Reset chat** to clear conversation.")
    st.markdown("---")
    if st.button("Reset chat"):
        st.session_state.chat_history = [
            {"role":"assistant", "content":"Hello Impeccabillem Warrior, I'm your CoMUI Pharmacology MB2 Assistant. How can I assist you today?", "ts": datetime.now().isoformat()}
        ]
        st.experimental_rerun()
    st.markdown("---")
    st.markdown("Made for medical students • Keep questions clinical and professional.")

# Inject CSS (unsafe HTML)
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------- Main chat card ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

# Chat window container (id used by JS to scroll)
st.markdown('<div id="chat-window" class="chat-window">', unsafe_allow_html=True)

# Render chat history (custom bubbles)
def render_chat():
    # ensure older entries without timestamp display properly
    for i, msg in enumerate(st.session_state.chat_history):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        ts = msg.get("ts")
        if not ts:
            ts = datetime.now().isoformat()
            st.session_state.chat_history[i]["ts"] = ts
        timestr = datetime.fromisoformat(ts).strftime("%b %d, %H:%M")
        # avatar / name
        if role == "user":
            avatar_html = '<div class="avatar" style="background:linear-gradient(90deg,#042a2c,#033a66);display:flex;align-items:center;justify-content:center;color:white">U</div>'
            msg_html = f"""
            <div class="row clearfix" style="justify-content:flex-end;">
              <div style="max-width:86%;text-align:right;">
                <div class="msg user">{content}</div>
                <div class="meta" style="text-align:right">{timestr}</div>
              </div>
              {avatar_html}
            </div>
            """
        else:
            avatar_html = '<div class="avatar" style="background:linear-gradient(90deg,#072a3a,#0a5261);display:flex;align-items:center;justify-content:center;color:white">A</div>'
            msg_html = f"""
            <div class="row clearfix" style="justify-content:flex-start;">
              {avatar_html}
              <div style="max-width:86%;text-align:left;">
                <div class="msg assistant">{content}</div>
                <div class="meta">{timestr}</div>
              </div>
            </div>
            """
        st.markdown(msg_html, unsafe_allow_html=True)

render_chat()

st.markdown('</div>', unsafe_allow_html=True)  # close chat-window container

# Footer input area: use a form so Enter can submit
with st.form(key="input_form", clear_on_submit=False):
    user_text = st.text_area("Ask your Pharmacology question(s):", height=80, key="user_input", placeholder="E.g., 'Explain mechanism of action and side effects of aminoglycosides.'")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        send = st.form_submit_button("Send", help="Send question to assistant")
    with col2:
        # keep a soft 'reset input' button (doesn't reset history)
        if st.form_submit_button("Clear input"):
            st.session_state.user_input = ""
    with col3:
        st.write("")  # spacer

# If user pressed Send in the form
if send and user_text and not st.session_state.sending:
    # append user message with timestamp
    st.session_state.chat_history.append({"role":"user", "content": user_text, "ts": datetime.now().isoformat()})
    st.session_state.sending = True

    # Show futuristic loader while computing — use custom CSS loader
    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<div class="card" style="text-align:center;padding:16px"><div class="loader"></div><div class="small">Deep Reasoning Activated…</div></div>', unsafe_allow_html=True)

    # call your unchanged backend
    try:
        response = generate_response(user_text)
    except Exception as e:
        response = "⚠️ Error generating response. See logs. " + str(e)

    # remove loader
    placeholder.empty()

    # append assistant response
    st.session_state.chat_history.append({"role":"assistant", "content": response, "ts": datetime.now().isoformat()})
    st.session_state.sending = False

    # re-render chat to include latest messages
    st.experimental_rerun()

# If not sending, still show hint under input
if not st.session_state.sending:
    st.markdown('<div class="small" style="margin-top:8px">Tip: Be specific; include drug names, doses or scenarios for better answers.</div>', unsafe_allow_html=True)

# Insert scroll JS to push view to bottom of chat
st.markdown(scroll_js, unsafe_allow_html=True)

# close main card
st.markdown('</div>', unsafe_allow_html=True)
