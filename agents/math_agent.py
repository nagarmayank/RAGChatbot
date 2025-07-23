from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

model_name = "gemini-2.5-flash-preview-05-20" #"meta-llama/llama-4-scout-17b-16e-instruct" # 
model_provider = "google_genai" #"groq" # 

model = init_chat_model(model=model_name, model_provider=model_provider)

# Define math tools
@tool
def add(a: str, b: str) -> float:
    """Add two numbers."""
    return float(a) + float(b)

@tool
def subtract(a: str, b: str) -> float:
    """Subtract b from a."""
    return float(a) - float(b)

@tool
def multiply(a: str, b: str) -> float:
    """Multiply two numbers."""
    return float(a) * float(b)

@tool
def divide(a: str, b: str) -> float:
    """Divide a by b."""
    if float(b) == 0:
        return "Error: Division by zero"
    return float(a) / float(b)

tools = [add, subtract, multiply, divide]

def math_agent(checkpointer=None):
    return create_react_agent(model, tools, prompt="You are a math agent. Use the tools provided to perform calculations.", name='math_agent', checkpointer=checkpointer)