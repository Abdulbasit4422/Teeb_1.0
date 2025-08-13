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

# Define system prompt template
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
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            chat_history.add_ai_message(msg["content"])
    
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
        temperature=0.2,
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

# --- App Title Bar with Icon ---
st.markdown("""
<div class='main-header'>
    <h1>Tibb 1.0 💊 </h1>
    <p>AI-Powered Pharmacology Assistant for CoMUI MB-2 Students </p>
</div>
""", unsafe_allow_html=True)

st.write("Ask your Pharmacology MB-2 questions and receive response based on our knowledge base of your ComUI MB-2 slides.")

# Page configuration
st.set_page_config(
    page_title="Tibb 1.0 - AI Pharmacology Assistant",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# Streamlit app layout remains unchanged
#st.title("Tibb 1.0")
#st.write("Ask your Pharmacology MB-2 questions and receive response based on our knowledge base of your ComUI MB-2 slides.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello Impeccabillem Warrior, I'm your CoMUI Pharmacology MB2 Assistant. How can I assist you today?"}
    ]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("💬 Ask your Pharmacology questions and let's see how I can help...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Deep Reasoning Activated..."):
        response = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})


# --- UI/UX Enhancements ---


# Initialize theme state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS for futuristic and medical theme
def get_css_theme():
    theme_class = "dark" if st.session_state.dark_mode else "light"
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    :root {{
        --primary-color: #00c6ff; /* Bright Blue */
        --secondary-color: #0072ff; /* Darker Blue */
        --accent-color: #00ff99; /* Teal/Green Accent */
        --background-color-light: #f0f2f6; /* Light Gray */
        --background-color-dark: #1a1a2e; /* Dark Blue/Purple */
        --card-background-light: #ffffff; /* White */
        --card-background-dark: #2e2e4a; /* Darker Purple */
        --text-color-light: #333333; /* Dark Gray */
        --text-color-dark: #e0e0e0; /* Light Gray */
        --border-color: rgba(0, 255, 153, 0.3); /* Semi-transparent accent */
        --shadow-color: rgba(0, 0, 0, 0.1);
        --font-family-heading: 'Orbitron', sans-serif;
        --font-family-body: 'Roboto', sans-serif;
    }}

    .{theme_class} {{
        --background-color: var(--background-color-{theme_class});
        --card-background: var(--card-background-{theme_class});
        --text-color: var(--text-color-{theme_class});
    }}

    .stApp {{
        background: linear-gradient(135deg, var(--background-color), var(--background-color-light));
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        font-family: var(--font-family-body);
        color: var(--text-color);
        transition: all 0.3s ease;
    }}

    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Header/Title Bar */
    .main-header {{
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 198, 255, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }}

    .main-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 3s infinite;
    }}

    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}

    .main-header h1 {{
        font-family: var(--font-family-heading);
        color: white;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 2px;
    }}

    .main-header p {{
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }}

    /* Sidebar */
    .stSidebar > div:first-child {{
        background: linear-gradient(180deg, var(--card-background), var(--background-color));
        border-right: 3px solid var(--primary-color);
        box-shadow: 4px 0 20px var(--shadow-color);
        padding-top: 2rem;
    }}

    .stSidebar .stButton > button {{
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
    }}

    .stSidebar .stButton > button:hover {{
        background: linear-gradient(45deg, var(--secondary-color), var(--accent-color));
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 198, 255, 0.4);
    }}

    /* Chat Messages */
    .stChatMessage {{
        background-color: var(--card-background);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px var(--shadow-color);
        transition: all 0.3s ease-in-out;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }}

    .stChatMessage::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, var(--border-color), transparent, var(--border-color));
        z-index: -1;
        border-radius: 20px;
    }}

    .stChatMessage[data-testid="user-message"] {{
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-bottom-right-radius: 5px;
        margin-left: 20%;
        transform: translateX(10px);
    }}

    .stChatMessage[data-testid="assistant-message"] {{
        background-color: var(--card-background);
        color: var(--text-color);
        border-bottom-left-radius: 5px;
        margin-right: 20%;
        transform: translateX(-10px);
    }}

    .stChatMessage:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 30px var(--shadow-color);
    }}

    /* Input area */
    .stChatInput > div {{
        background: var(--card-background);
        border-radius: 25px;
        border: 3px solid var(--primary-color);
        box-shadow: 0 4px 20px rgba(0, 198, 255, 0.2);
        transition: all 0.3s ease;
    }}

    .stChatInput > div:focus-within {{
        border-color: var(--accent-color);
        box-shadow: 0 0 0 0.3rem rgba(0, 255, 153, 0.25);
        transform: scale(1.02);
    }}

    .stChatInput input {{
        background: transparent !important;
        border: none !important;
        color: var(--text-color) !important;
        font-size: 1.1rem !important;
        padding: 1rem 1.5rem !important;
    }}

    /* Loading animation */
    .stSpinner > div {{
        border: 4px solid rgba(0, 198, 255, 0.3);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite, pulse 2s ease-in-out infinite;
        margin: 2rem auto;
    }}

    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
    }}

    /* Timestamp styling */
    .timestamp {{
        font-size: 0.8em;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.5rem;
        display: block;
        font-style: italic;
    }}

    .stChatMessage[data-testid="assistant-message"] .timestamp {{
        color: rgba(51, 51, 51, 0.6);
    }}

    /* Theme toggle button */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
    }}

    .theme-toggle:hover {{
        transform: scale(1.1) rotate(180deg);
        box-shadow: 0 6px 20px rgba(0, 198, 255, 0.5);
    }}

    /* Responsive design */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 2rem;
        }}
        
        .stChatMessage[data-testid="user-message"] {{
            margin-left: 10%;
        }}
        
        .stChatMessage[data-testid="assistant-message"] {{
            margin-right: 10%;
        }}
    }}

    /* Info cards */
    .info-card {{
        background: linear-gradient(135deg, var(--card-background), rgba(0, 198, 255, 0.1));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid var(--accent-color);
        box-shadow: 0 4px 15px var(--shadow-color);
        transition: all 0.3s ease;
    }}

    .info-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 6px 25px var(--shadow-color);
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--background-color);
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, var(--secondary-color), var(--accent-color));
    }}

</style>
"""

st.markdown(get_css_theme(), unsafe_allow_html=True)

# Theme toggle button
#col1, col2, col3 = st.columns([1, 1, 1])
#with col2:
    #if st.button("🌓 Toggle Theme", key="theme_toggle"):
        #st.session_state.dark_mode = not st.session_state.dark_mode
        #st.rerun()

# --- Sidebar Content ---
with st.sidebar:
    st.markdown("<h2 style='font-family: var(--font-family-heading); color: var(--primary-color); text-align: center;'>📚 App Information</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h4>🤖 About Tibb 1.0</h4>
        <p>Tibb 1.0 is an AI-powered pharmacology assistant designed for CoMUI MB-2 students. 
        It leverages advanced AI models and acomprehensive ComUI Materials knowledge base to provide accurate 
        and elaborate answers to your pharmacology questions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: var(--font-family-heading); color: var(--primary-color); text-align: center;'>📋 Instructions</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h4>🚀 How to Use</h4>
        <ol>
            <li>Type your pharmacology question in the input box below</li>
            <li>Press Enter or click the send button</li>
            <li>Tibb 1.0 will provide a detailed response based on its knowledge base</li>
            <li>Use the 'Clear Chat' button to reset the conversation</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    
    # App stats
    st.markdown("""
    <div class='info-card'>
        <h4>📊 Session Stats</h4>
        <p><strong>Messages:</strong> {}</p>
        <p><strong>Theme:</strong> {}</p>
        <p><strong>Status:</strong> 🟢 Online</p>
    </div>
    """.format(
        len(st.session_state.get('chat_history', [])),
        "Dark" if st.session_state.get('dark_mode', False) else "Light"
    ), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 0.8em; color: var(--text-color); font-style: italic;'>⚡ Powered by Gemini 2.0 Flash & Pinecone</p>", unsafe_allow_html=True)


# Initialize chat history in session state
from datetime import datetime

#if "chat_history" not in st.session_state:
    #st.session_state.chat_history = [
       # {"role": "assistant", "content": "Hello Impeccabillem Warrior, I'm your CoMUI Pharmacology MB2 Assistant. How can I assist you today?", "timestamp": datetime.now().strftime("%H:%M:%S")}
   # ]

# Main chat container
#st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat history with enhanced styling
#for i, message in enumerate(st.session_state.chat_history):
    #with st.chat_message(message["role"]):
        # Add message content with timestamp
        #content_with_timestamp = f"{message['content']}\n\n<span class='timestamp'>⏰ {message['timestamp']}</span>"
        #st.markdown(content_with_timestamp, unsafe_allow_html=True)
timestamp = message.get("timestamp", "")  # fallback to empty string if missing
if timestamp:
    content_with_timestamp = f"{message['content']}\n\n<span class='timestamp'>⏰ {timestamp}</span>"
else:
    content_with_timestamp = message['content']

st.markdown(content_with_timestamp, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Enhanced input area
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

# Handle user input with enhanced UX
##user_input = st.chat_input("💬 Ask your Pharmacology questions and let's see how I can help...")

'''if user_input:
    # Add user message with timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    
    with st.chat_message("user"):
        user_content = f"{user_input}\n\n<span class='timestamp'>⏰ {current_time}</span>"
        st.markdown(user_content, unsafe_allow_html=True)
    
    st.session_state.chat_history.append({
        "role": "user", 
        "content": user_input, 
        "timestamp": current_time
    })
    
    # Show enhanced loading animation with custom message
    #with st.spinner("🧠 Activating Neural Pathways... 🔬 Analyzing Pharmacology Data... ⚡ Generating Response..."):
        #response = generate_response(user_input)'''
    
    # Add assistant message with timestamp
response = None  # Prevent NameError

# If you're ready to re-enable:
# response = generate_response(user_input)

   
if response:
    response_time = datetime.now().strftime("%H:%M:%S")
    
    with st.chat_message("assistant"):
        assistant_content = f"{response}\n\n<span class='timestamp'>⏰ {response_time}</span>"
        st.markdown(assistant_content, unsafe_allow_html=True)
    
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response, 
        "timestamp": response_time
    })

st.markdown("</div>", unsafe_allow_html=True)

# Add some spacing at the bottom for better UX
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Footer with additional info
st.markdown("""
<div style='text-align: center; padding: 2rem; margin-top: 3rem; border-top: 2px solid var(--primary-color); background: linear-gradient(135deg, var(--card-background), rgba(0, 198, 255, 0.05));'>
    <h4 style='color: var(--primary-color); font-family: var(--font-family-heading);'>🎓 Tibb 1.0 - Your AI Pharmacology Companion</h4>
    <p style='color: var(--text-color); margin: 0.5rem 0;'>Empowering ComUI MB-2 students with AI-driven pharmacology insights</p>
    <p style='font-size: 0.9em; color: var(--text-color); opacity: 0.8;'> Built ❤️ using RAG, Gemini 2.0 Flash. </p>
</div>
""", unsafe_allow_html=True)