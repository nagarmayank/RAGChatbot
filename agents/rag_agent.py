from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.tools import tool
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from utils import pretty_print_messages
from langchain_tavily import TavilySearch

load_dotenv()

model_name = "gemini-2.5-flash-preview-05-20" #"meta-llama/llama-4-scout-17b-16e-instruct" # 
model_provider = "google_genai" #"groq" # 

model = init_chat_model(model=model_name, model_provider=model_provider)

@tool
def rag_search(query: str) -> str:
    """Search the vector database for relevant context given a query."""
    emb_model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, cache_folder="embeddings_cache"
    )

    qdrant_host = "localhost"
    qdrant_port = 6333
    collection_name = "rag_documents"

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        url=f"http://{qdrant_host}:{qdrant_port}",
    )

    docs = vector_store.similarity_search(query)
    if not docs:
        return "No relevant documents found."
    # Return a summary of sources and content
    return "\n\n".join(
        f"Source: {doc.metadata.get('_id', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}..."
        for doc in docs
    )

def rag_agent():
    tools = [rag_search]
    return create_react_agent(
        model,
        tools, 
        prompt="You are a RAG agent skilled in Artificial Intelligence topics. \
                Use the tools provided to search for relevant context in the vector database.", 
        name='rag_agent'
    )