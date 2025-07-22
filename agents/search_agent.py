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
def search_tavily(query: str, max_results=3) -> dict:
    """
    Perform a search using TavilySearch with specified parameters.
    Inputs:
        query (str): The search query to be executed.
        max_results (int): The maximum number of results to return. Default is 3.
    Outputs:
        result (dict): The search results from TavilySearch.
    """
    search = TavilySearch(max_results=max_results, include_answer=False, include_raw_content=False)
    result = search.invoke({'query':query})
    return result

def search_agent():
    """
    Search agent function that uses TavilySearch to find relevant information.
    Inputs:
        query (str): The search query to be executed.
    Outputs:
        result (dict): The search results from TavilySearch.
    """
    return create_react_agent(
        model,
        tools=[search_tavily],
        prompt="You are a search agent. Use the TavilySearch tool to find relevant information based on user queries.",
        name='search_agent'
    )