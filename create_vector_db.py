from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

path = os.getcwd()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

qdrant_host = "localhost"
qdrant_port = 6333
collection_name = "rag_documents"

client = QdrantClient(host=qdrant_host, port=qdrant_port)
# client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE),)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name=collection_name,
    embedding=embeddings,
    url=f"http://{qdrant_host}:{qdrant_port}",
)

for f in tqdm(os.listdir(os.path.join(path, 'data'))):
    print(f)
    pdf_loader = PyMuPDFLoader(file_path=os.path.join(path, 'data', f))
    docs = pdf_loader.load_and_split(text_splitter=splitter)
    vector_store.add_documents(docs)

print("Documents added to the collection")

# response = vector_store.similarity_search("Who is the chairman of Reliance Industries", k=5)  # Example query to test the vector store

# print(response)