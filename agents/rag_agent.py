from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from utils.db_config import DBConfig
import os

load_dotenv()

model_name = os.getenv("model_name")
model_provider = os.getenv("model_provider")

model = init_chat_model(model=model_name, model_provider=model_provider)

@tool
def rag_search(query: str) -> str:
    """Search the vector database for relevant context given a query."""
    db_config = DBConfig()
    vector_db = db_config.get_vector_store()

    docs = vector_db.similarity_search(query)
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