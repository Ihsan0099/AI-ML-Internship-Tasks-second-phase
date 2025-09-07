import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os, asyncio

# 🔑 Set Gemini API key
# Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = "paste your owen api key"  

# 🔧 Fix: ensure event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Core Chatbot Logic (Kept as is) ---

# Load the document
loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings with Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
db = FAISS.from_documents(texts, embeddings)

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory)

# --- Updated Streamlit UI ---

# Page configuration
st.set_page_config(
    page_title="Context Aware Chatbot",
    page_icon="✨",
    layout="centered"
)

# Custom CSS for a professional look
st.markdown("""
<style>
/* General body and background */
.stApp {
    background-color: #0b1f3c; /* Dark blue background */
    color: #e0e6f0; /* Light gray text */
    font-family: 'Inter', sans-serif;
}

/* Header styling */
.main h1 {
    color: #4CAF50; /* A nice green color for the title */
    text-align: center;
    font-weight: bold;
    font-size: 3em;
    margin-bottom: 0.5em;
    text-shadow: 2px 2px 4px #000000;
}
.main .stMarkdown p {
    color: #aeb5c2; /* Lighter text for the subtitle */
    text-align: center;
    font-style: italic;
    font-size: 1.1em;
    margin-bottom: 2em;
}

/* Chat message bubbles */
.stChatMessage {
    background-color: #17365d; /* Darker blue for chat bubbles */
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stChatMessage.st-assistant {
    background-color: #2b496b; /* Slightly different shade for assistant */
    margin-right: 20%;
}
.stChatMessage.st-user {
    background-color: #3b5a7b; /* User messages a bit lighter */
    margin-left: 20%;
}

/* Chat input field */
.stTextInput > div > div > input {
    background-color: #0b1f3c !important;
    color: #e0e6f0 !important;
    border: 1px solid #4CAF50 !important;
    border-radius: 20px !important;
    padding: 10px 20px !important;
}
.stTextInput > label {
    display: none; /* Hide the label */
}
</style>
""", unsafe_allow_html=True)

# Main content
st.title("Gemini RAG Chatbot")
st.markdown("Ask me anything about the provided document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa.invoke({"question": prompt})
            st.markdown(response["answer"])
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})