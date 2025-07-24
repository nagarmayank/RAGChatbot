from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from agents.math_agent import math_agent
from agents.rag_agent import rag_agent
from agents.search_agent import search_agent
from agents.ambiguous_agent import ambiguous_agent
from agents.llm_agent import llm_agent

import os
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("model_name")
model_provider = os.getenv("model_provider")

model = init_chat_model(model=model_name, model_provider=model_provider)

def supervisor_agent():
    memory = MemorySaver()
    supervisor_agent = create_supervisor(
        model=model,
        agents=[math_agent(checkpointer=memory), rag_agent(checkpointer=memory), search_agent(checkpointer=memory), llm_agent(checkpointer=memory), ambiguous_agent(checkpointer=memory)],
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
    ).compile(checkpointer=memory)

    return supervisor_agent