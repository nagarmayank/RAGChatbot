# Generate a streamlit page that expalins the whole project
import streamlit as st

st.set_page_config(page_title="RAGChatbot Home", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")

st.title("🤖 RAGChatbot")
st.markdown("""
A Retrieval-Augmented Generation (RAG) chatbot with PDF document upload, vector database creation, and a modern chat UI using Streamlit.

---

## 🚀 Features

- **Chatbot UI:** Conversational interface using [streamlit-chat](https://github.com/AI-Yash/st-chat)
- **PDF Upload:** Add new knowledge by uploading PDF files
- **Vector Database:** Uses Qdrant and HuggingFace embeddings for retrieval
- **Sidebar Navigation:** Easily switch between chat and document upload pages
- **Math & Web Search:** Supports math operations and web search fallback
- **Agent Architecture:** Modular agents for math, RAG, and a supervisor agent to orchestrate tool and agent selection
- **Kubernetes Manifests:** Easily deploy the app on Kubernetes using provided YAML files

---

## 🧠 Agent Architecture

- **Supervisor Agent:** Orchestrates and routes queries to specialized agents (math, RAG, etc.).
- **Math Agent:** Handles mathematical queries (add, subtract, multiply, divide).
- **RAG Agent:** Answers questions using your uploaded documents and the vector database.
- **Extensible:** Add more agents/tools (e.g., web search) by extending the `agents/` folder.

---

## ☸️ Kubernetes Deployment

Kubernetes YAML files are provided in the `manifests/` folder for easy deployment.

**How to use:**
1. Build and push your Docker image.
2. Apply the manifests:
   ```bash
   kubectl apply -f manifests/
   ```
3. Expose the pod using a Service or port-forward:
   ```bash
   kubectl port-forward pod/ragchatbot 8501:8501
   ```
4. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏁 Getting Started (Locally)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAGChatbot.git
   cd RAGChatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Qdrant (Vector DB)**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run Chat.py
   ```

---

## 📁 Project Structure

```
RAGChatbot/
├── Chat.py                # Main chatbot UI
├── pages/
│   └── 1_Add Documents.py # PDF upload and vector DB update page
├── agents/                # Modular agent definitions (supervisor, math, rag, etc.)
├── utils/                 # Utility modules (config, helpers, etc.)
├── manifests/             # Kubernetes YAML files for deployment
├── data/                  # Uploaded PDF files
├── vectordb/              # Vector database files (if using local FAISS)
├── requirements.txt
├── Dockerfile
├── README.md
```

---

## ℹ️ Notes

- Uploaded PDFs are stored in the `data/` directory.
- The vector database is stored in Qdrant (or `vectordb/` if using FAISS).
- The app uses CPU for embedding generation by default.
- For best results, use high-quality, text-based PDFs.
- You can extend the agent system by adding new tools or agents in the `agents/` folder and registering them with the supervisor agent.

---

**Enjoy your RAG-powered chatbot with modular agent orchestration and Kubernetes support!**
""")