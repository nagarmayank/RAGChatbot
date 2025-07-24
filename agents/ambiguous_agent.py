from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

model_name = os.getenv("model_name")
model_provider = os.getenv("model_provider")

model = init_chat_model(model=model_name, model_provider=model_provider)

def ambiguous_agent(checkpointer=None):
    """
    Create an agent that handles ambiguous requests.
    If the user request is not clear, ask for clarification.
    Do not perform any other activities.
    """
    return create_react_agent(
        model,
        tools=[],
        prompt="You are an ambiguous agent. If the user request is not clear, ask for clarification. Do not perform any other activities.",
        name='ambiguos_agent',
        checkpointer=checkpointer
    )