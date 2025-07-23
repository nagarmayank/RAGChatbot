import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import os
from utils import db_config
from utils.db_config import DBConfig

logging.basicConfig(level=logging.INFO)

st.title("Add Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files to add to the knowledge base",
    type=["pdf"],
    accept_multiple_files=True
)

db_config = DBConfig()
vector_store = db_config.get_vector_store()

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
            print(file_path)
            pdf_loader = PyMuPDFLoader(file_path=file_path)
            docs = pdf_loader.load_and_split(text_splitter=splitter)
            print(f"Loaded {len(docs)} documents from {uploaded_file.name}")
            print(f"Starting adding to vector db...")
            vector_store.add_documents(docs)
    st.success("Vector database updated successfully!")