# RAGChatbot

A chatbot with advanced capabilities including vector database document upload, modular agent architecture, and a modern chat UI using Streamlit.

---

## 🚀 Features

- **Chatbot UI:** Conversational interface using [streamlit-chat](https://github.com/AI-Yash/st-chat)
- **Chat Interface:** Modern chat interface with message history and user input
- **PDF Upload:** Add new knowledge by uploading PDF files
- **Observability:** Track user interactions and agent responses in Langsmith
- **User Feedback:** Track user feedback for each chat response in Langsmith
- **Vector Database:** Uses Qdrant database hosted as a managed service with persistent storage
- **Application Monitoring:** Integrated with Prometheus and Grafana
- **Docker Support:** Containerized application for easy deployment
- **Scalability:** Built-in support for scaling and load balancing

---

## 🧠 Agent Architecture

- **Supervisor Agent:** Orchestrates and routes queries to specialized agents (math, RAG, etc.)
- **Math Agent:** Supports math operations (add, subtract, multiply, divide)
- **Web Search Agent:** Integrates web search capabilities for general queries
- **RAG (Retrieval-Augmented Generation) Agent:** Answers questions using uploaded documents
- **LLM Agent:** Uses LLM for directly responding to queries
- **Ambiguous Queries:** Handles ambiguous queries by routing to the appropriate agent

---

## ☸️ Deployment

Kubernetes YAML files are provided in the `manifests/` folder for easy deployment.

**How to use:**
1. Add environment variables as provided in the `.env.example` file.
2. Build and push your Docker image (automated using GitHub Actions).
3. Pull the latest Docker image from Docker Hub:
   ```bash
   docker pull nagarmayank/ragchatbot:latest
   ```
4. Apply the manifests:
   ```bash
   kubectl apply -f manifests/
   ```
5. Open [http://localhost](http://localhost) in your browser to access the Streamlit application.

---

## 📁 Project Structure

```
RAGChatbot/
├── Home.py                # UI Home page with project overview
├── pages/
│   ├── 1_Add Documents.py # PDF upload and vector DB update page
│   ├── 2_Chat.py          # Chat interface for interacting with the chatbot
├── agents/                # Modular agent definitions (supervisor, math, rag, etc.)
├── utils/                 # Utility modules (db_config, helper_methods)
├── manifests/             # Kubernetes YAML files for deployment
├── data/                  # Uploaded PDF files
├── requirements.txt
├── Dockerfile
├── README.md
```

---

## Notes

- Uploaded PDFs are stored in the `data/` directory.
- The vector database is managed by Qdrant.
- The app uses CPU for embedding generation by default.
- For best results, use high-quality, text-based PDFs.
- You can extend the agent system by adding new tools or agents in the `agents/` folder and registering them with the supervisor agent.

---

**Enjoy your chatbot with modular agent orchestration