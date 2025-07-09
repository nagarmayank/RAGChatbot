import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

st.title("Add Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files to add to the knowledge base",
    type=["pdf"],
    accept_multiple_files=True
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
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        v_path = os.path.join(os.getcwd(), "vectordb")

        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            pdf_loader = PyMuPDFLoader(file_path=file_path)
            docs = pdf_loader.load_and_split(text_splitter=splitter)
            if os.path.exists(v_path):
                db = FAISS.load_local(folder_path=v_path, embeddings=embeddings, allow_dangerous_deserialization=True)
                db.add_documents(docs)
            else:
                db = FAISS.from_documents(docs, embedding=embeddings)
            db.save_local(folder_path=v_path)
    st.success("Vector database updated successfully!")