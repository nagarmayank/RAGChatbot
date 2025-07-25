# Generate a streamlit page that expalins the whole project
import streamlit as st

st.set_page_config(page_title="Chatbot Home", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")

st.title("🤖 Chatbot")
st.markdown("""
A chatbot with various capabilities like vector database document upload, and a modern chat UI using Streamlit.

---

## 🚀 Features

- **Chatbot UI:** Conversational interface using [streamlit-chat](https://github.com/AI-Yash/st-chat)
- **Chat Interface:** Modern chat interface with message history and user input
- **PDF Upload:** Add new knowledge by uploading PDF files
- **Observability:** Track user interactions and agent responses in Langsmith
- **User Feedback:** Track user feedback for each chat response in Langsmith
- **Vector Database:** Uses Qdrant database hosted as a managed service with persistent storage
- **Application Monitoring**: Using Prometheus and Grafana
- **Docker Support:** Containerized application for easy deployment
- **Chat application** has built-in support for scaling and load balancing

---

## 🧠 Agent Architecture

- **Supervisor Agent:** Orchestrates and routes queries to specialized agents (math, RAG, etc.).
- **Math Agent:** Supports math operations (add, subtract, multiply, divide)
- **Web Search Agent:** Integrates web search capabilities for general queries
- **RAG (Retrieval-Augmented Generation):** Answers questions using uploaded documents
- **LLM Agent:** Uses LLM for directly responding to queries
- **Ambiguous Queries:** Handles ambiguous queries by routing to the appropriate agent
---
""")

st.image("static/graph_image.png", caption="Agent Architecture Overview", use_container_width=True)

st.markdown("""
## ☸️ Deployment

Kubernetes YAML files are provided in the `manifests/` folder for easy deployment.

**How to use:**
1. Ensure to add environment variables as provided in `.env.example` file
2. Build and push your Docker image. This is automated using GitHub Actions
3. Pull the latest Docker image from Docker Hub:
   ```bash
   cd ragchatbot
   docker pull nagarmayank/ragchatbot:latest
   ```
4. Run the command to install Prometheus and Grafana:
   ```bash
   helm upgrade --install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false --set prometheus-node-exporter.hostRootFsMount.enabled=false --set prometheus-node-exporter.hostRootFsMount.mountPropagation='HostToContainer'
   ```
5. Apply the manifests:
   ```bash
   kubectl apply -f manifests/
   ```
6. Open [http://localhost](http://localhost) in your browser to access the Streamlit application.

---

## 📁 Project Structure

```
RAGChatbot/
├── Home.py                # UI Home page with project overview
├── pages/
│   └── 1_Add Documents.py # PDF upload and vector DB update page
│   └── 2_Chat.py # Chat interface for interacting with the chatbot
├── agents/                # Modular agent definitions (supervisor, math, rag, etc.)
├── utils/                 # Utility modules (db_config, helper_methods)
├── manifests/             # Kubernetes YAML files for deployment
├── data/                  # Uploaded PDF files
├── requirements.txt
├── Dockerfile
├── README.md
```

---

**Enjoy your chatbot with modular agent orchestration and Kubernetes support!**
""")