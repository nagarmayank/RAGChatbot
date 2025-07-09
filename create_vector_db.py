from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

path = os.getcwd()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for f in tqdm(os.listdir(os.path.join(path, 'data'))):
    print(f)
    pdf_loader = PyMuPDFLoader(file_path=os.path.join(path, 'data', f))
    docs = pdf_loader.load_and_split(text_splitter=splitter)
    v_path = os.path.join(path,'vectordb')
    if os.path.exists(v_path):
        db = FAISS.load_local(folder_path=v_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embedding=embeddings)

db.save_local(folder_path=v_path)