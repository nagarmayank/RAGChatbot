# RAGChatbot

A Retrieval-Augmented Generation (RAG) chatbot with PDF document upload, vector database creation, and a modern chat UI using Streamlit.

## Features

- **Chatbot UI:** Conversational interface using [streamlit-chat](https://github.com/AI-Yash/st-chat)
- **PDF Upload:** Add new knowledge by uploading PDF files
- **Vector Database:** Uses Qdrant and HuggingFace embeddings for retrieval
- **Sidebar Navigation:** Easily switch between chat and document upload pages
- **Math & Web Search:** Supports math operations and web search fallback
- **Agent Architecture:** Modular agents for math, RAG, and a supervisor agent to orchestrate tool and agent selection
- **Kubernetes Manifests:** Easily deploy the app on Kubernetes using provided YAML files

## Agent Architecture

The `agents/` folder contains modular agents:

- **supervisor_agent:**  
  The main orchestrator agent. It routes user queries to the appropriate specialized agent (e.g., math agent, RAG agent) based on the query type. It uses LangGraph's `create_supervisor_agent` to bind and manage multiple agents.
- **math_agent:**  
  Handles mathematical queries using tool calling (add, subtract, multiply, divide).
- **rag_agent:**  
  Handles retrieval-augmented generation by searching the vector database for relevant context and generating answers from your uploaded documents.
- **Other agents/tools:**  
  You can extend the system by adding more agents or tools (e.g., web search, code execution) in the `agents/` folder and registering them with the supervisor agent.

## Kubernetes Deployment

Kubernetes YAML files are provided in the `manifests/` folder for easy deployment.

### How to Use

1. **Build and push your Docker image**  
   Make sure your image is available in a registry accessible by your Kubernetes cluster.

2. **Apply the manifests**  
   ```bash
   kubectl apply -f manifests/
   ```

3. **What’s included in `manifests/`:**
   - **ragchatbot-pod.yaml**: Pod definition for the RAGChatbot app, including health checks and persistent volumes for data and vector DB.
   - (You can add more files for services, deployments, ingress, etc., as needed.)

4. **Access the app**  
   Expose the pod using a Kubernetes Service or port-forward:
   ```bash
   kubectl port-forward pod/ragchatbot 8501:8501
   ```
   Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Getting Started (Local)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/RAGChatbot.git
cd RAGChatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
streamlit
streamlit-chat
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
PyMuPDF
qdrant-client
langchain-qdrant
tqdm
python-dotenv
```

### 3. Run Qdrant (Vector DB)

You can run Qdrant locally using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Run the Streamlit app

```bash
streamlit run Chat.py
```

- The chatbot UI will open in your browser.
- Use the sidebar to navigate to **Add Documents** for uploading PDFs.

### 5. Add Documents

- Go to the **Add Documents** page via the sidebar.
- Upload your PDF files.
- The app will process and update the vector database automatically.

### 6. Chat

- Return to the **Chatbot** page.
- Ask questions based on your uploaded documents or perform math/web queries.

## Project Structure

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

## Notes

- Uploaded PDFs are stored in the `data/` directory.
- The vector database is stored in Qdrant (or `vectordb/` if using FAISS).
- The app uses CPU for embedding generation by default.
- For best results, use high-quality, text-based PDFs.
- You can extend the agent system by adding new tools or agents in the `agents/` folder and registering them with the supervisor agent.

---

**Enjoy your RAG-powered chatbot with modular agent orchestration and Kubernetes support!**