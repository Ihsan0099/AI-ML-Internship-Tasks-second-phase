# ✨ Gemini RAG Chatbot

A **context-aware chatbot** built with [Streamlit](https://streamlit.io/) and [LangChain](https://www.langchain.com/). It uses **Google Gemini** for chat + embeddings and retrieves answers from `data.txt` using **FAISS**. The bot remembers past conversations with **LangChain memory** and provides context-aware responses.

---

## 📂 Files
- `app.py` → Main Streamlit chatbot app  
  - Loads `data.txt` into a FAISS vector store  
  - Creates embeddings using **GoogleGenerativeAIEmbeddings**  
  - Uses **Gemini 1.5 Flash** as the chat model  
  - Adds **ConversationalBufferMemory** to maintain chat history  
  - Runs a **ConversationalRetrievalChain** for context-based answers  
  - Custom CSS styles for a professional chatbot UI  
- `data.txt` → Knowledge base (Machine Learning tutorial)  
  - Covers **types of ML**: Supervised, Unsupervised, Reinforcement, Semi & Self-Supervised  
  - Explains the **ML pipeline**: problem definition → data collection → preprocessing → EDA → feature engineering → model training → evaluation → deployment  
  - Includes tutorials on **data cleaning, EDA, feature engineering, model evaluation, and gradient descent** with Python code examples  

---

## 🚀 Setup & Run
```bash
# Clone repo
git clone https://github.com/your-username/gemini-rag-chatbot.git
cd gemini-rag-chatbot

# Create venv
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install streamlit langchain langchain-community langchain-google-genai faiss-cpu

# Add Google API key (edit app.py or export env var)
export GOOGLE_API_KEY="your_api_key_here"   # Mac/Linux
set GOOGLE_API_KEY="your_api_key_here"      # Windows

# Run app
streamlit run app.py
