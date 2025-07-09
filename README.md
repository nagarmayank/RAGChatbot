# RAGChatbot

A Retrieval-Augmented Generation (RAG) chatbot with PDF document upload, vector database creation, and a modern chat UI using Streamlit.

## Features

- **Chatbot UI:** Conversational interface using [streamlit-chat](https://github.com/AI-Yash/st-chat)
- **PDF Upload:** Add new knowledge by uploading PDF files
- **Vector Database:** Uses FAISS and HuggingFace embeddings for retrieval
- **Sidebar Navigation:** Easily switch between chat and document upload pages

## Getting Started

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
faiss-cpu
tqdm
```

### 3. Run the Streamlit app

```bash
streamlit run Chat.py
```

- The chatbot UI will open in your browser.
- Use the sidebar to navigate to **Add Documents** for uploading PDFs.

### 4. Add Documents

- Go to the **Add Documents** page via the sidebar.
- Upload your PDF files.
- The app will process and update the vector database automatically.

### 5. Chat

- Return to the **Chatbot** page.
- Ask questions based on your uploaded documents.

## Project Structure

```
RAGChatbot/
├── Chat.py                # Main chatbot UI
├── pages/
│   └── 1_Add Documents.py # PDF upload and vector DB update page
├── create_vector_db.py    # (Optional) Script for batch vector DB creation
├── data/                  # Uploaded PDF files
├── vectordb/              # FAISS vector database files
├── rag_app.py             # RAG agent logic
├── requirements.txt
└── README.md
```

## Notes

- Uploaded PDFs are stored in the `data/` directory.
- The vector database is stored in the `vectordb/` directory.
- The app uses CPU for embedding generation by default.
- For best results, use high-quality, text-based PDFs.

---

**Enjoy your RAG-powered chatbot!**