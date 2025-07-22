from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain import hub
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
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

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    sources: List[str]

class RAGAgent:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

        graph = StateGraph(State)
        graph.add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()

    def retrieve(self, state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        sources = [f"Chunk Id: {doc.metadata['_id']}, Page: {doc.metadata['page']}" for doc in retrieved_docs]
        return {"context": retrieved_docs, "sources": sources}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.model.invoke(messages)
        return {"answer": response.content, "sources": state["sources"]}

if __name__ == "__main__":
    print("RAG application is ready to use.")
    # Test the RAG application
    model = init_chat_model("gemini-2.5-flash-preview-05-20", model_provider="google_genai")
    prompt = hub.pull("rlm/rag-prompt")
    rag_agent = RAGAgent(model, prompt)
    result = rag_agent.graph.invoke({"question": "What is the ownership structure of Reliance Industries?"})

    print(f'Answer: {result["answer"]}')
    print()
    print(f"Sources: {result["sources"]}")
    # print(result)