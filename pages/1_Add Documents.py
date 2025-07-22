import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import logging
import os

logging.basicConfig(level=logging.INFO)

st.title("Add Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files to add to the knowledge base",
    type=["pdf"],
    accept_multiple_files=True
)

qdrant_host = "localhost"
qdrant_port = 6333
collection_name = "rag_documents"

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

client = QdrantClient(host=qdrant_host, port=qdrant_port)
# client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE),)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name=collection_name,
    embedding=embeddings,
    url=f"http://{qdrant_host}:{qdrant_port}",
)

if uploaded_files:
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully!")

    # Vector DB creation
    with st.spinner("Processing and updating vector database..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        v_path = os.path.join(os.getcwd(), "vectordb")

        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            pdf_loader = PyMuPDFLoader(file_path=file_path)
            docs = pdf_loader.load_and_split(text_splitter=splitter)
            vector_store.add_documents(docs)
    st.success("Vector database updated successfully!")

logging.info("Documents added to the collection")