from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

model_name = os.getenv("model_name")
model_provider = os.getenv("model_provider")

model = init_chat_model(model=model_name, model_provider=model_provider)

def llm_agent():
    """
    Create an agent that handles direct LLM calls.
    """
    return create_react_agent(
        model,
        tools=[],
        prompt="You are a helpful agent. Respond in short sentences.",
        name='llm_agent'
    )