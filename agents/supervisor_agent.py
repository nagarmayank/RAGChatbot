from langchain.chat_models import init_chat_model
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from utils import pretty_print_messages

from agents.math_agent import math_agent
from agents.rag_agent import rag_agent
from agents.search_agent import search_agent
from agents.ambiguous_agent import ambiguous_agent
from agents.llm_agent import llm_agent

from dotenv import load_dotenv

load_dotenv()

model_name = "gemini-2.5-flash-preview-05-20" #"meta-llama/llama-4-scout-17b-16e-instruct" # 
model_provider = "google_genai" #"groq" # 

model = init_chat_model(model=model_name, model_provider=model_provider)

def supervisor_agent():
    memory = InMemorySaver()
    store = InMemoryStore()
    supervisor_agent = create_supervisor(
        model=model,
        agents=[math_agent(), rag_agent(), search_agent(), llm_agent(), ambiguous_agent()],
        prompt="You are a supervisor responsible for delegating tasks. \
            Assign work to one agent at a time, do not call agents in parallel. Do not do any work yourself. Make sure that either one of the agent is assigned the task. \" \
            The available agents are: \
            math_agent is for mathematical calculations, \
            rag_agent is for searching relevant context in the vector database related to Artificial Intelligence, \" \
            search_agent is for general web searches, \" \
            llm_agent is for general tasks and responding to greetings, \" \
            ambiguous_agent is for handling ambiguous requests that do not have any context.",
        output_mode="last_message",
        include_agent_name='inline'
    ).compile(checkpointer=memory, store=store)

    return supervisor_agent