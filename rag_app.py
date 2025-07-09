from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain import hub
from langgraph.graph import START, StateGraph

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGAgent:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

        graph = StateGraph(State)
        graph.add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()

    def retrieve(self, state: State):
        v_path = os.path.join(os.getcwd(), 'vectordb')
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vector_store = FAISS.load_local(folder_path=v_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.model.invoke(messages)
        return {"answer": response.content}

if __name__ == "__main__":
    print("RAG application is ready to use.")
    # Test the RAG application
    model = init_chat_model("gemini-2.5-flash-preview-05-20", model_provider="google_genai")
    prompt = hub.pull("rlm/rag-prompt")
    rag_agent = RAGAgent(model, prompt)
    result = rag_agent.graph.invoke({"question": "Who is the author of the book?"})
    # result = rag_agent.graph.invoke({"question": "What are embeddings?"})

    # print(f'Context: {result["context"]}\n\n')
    print(f'Answer: {result["answer"]}')