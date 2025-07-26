from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import streamlit as st

class DBConfig:
    '''
    DBConfig is a configuration and utility class for managing connections to a Qdrant vector database and initializing embedding models.
    Attributes:
        qdrant_host (str): Hostname or IP address of the Qdrant server.
        qdrant_port (int): Port number for the Qdrant server.
        collection_name (str): Name of the Qdrant collection to use.
        model_name (str): Name of the HuggingFace embedding model.
        model_kwargs (dict): Keyword arguments for initializing the embedding model.
        encode_kwargs (dict): Keyword arguments for encoding embeddings.
    Methods:
        _get_embeddings():
            Initializes and returns a HuggingFaceEmbeddings instance using the specified model name and keyword arguments.
        get_vector_store():
            Establishes a connection to the Qdrant vector database and loads an existing collection as a vector store using the specified embedding model.
    '''
    def __init__(self):
        self.qdrant_host = "qdrant-db-svc"
        self.qdrant_port = 6333
        self.collection_name = "rag_documents"
        self.model_name = "BAAI/bge-large-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        
    @st.cache_resource
    def _get_embeddings(self):
        """
        Initializes and returns HuggingFaceEmbeddings using the specified model name and keyword arguments.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with the provided parameters.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name, model_kwargs=self.model_kwargs, encode_kwargs=self.encode_kwargs
        )
        return self.embeddings
    
    def get_vector_store(self):
        """
        Initializes and returns a vector store instance connected to a Qdrant collection.
        Establishes a connection to the Qdrant vector database using the configured host and port,
        and loads an existing collection as a vector store with the specified embedding model.
        Returns:
            QdrantVectorStore: An instance of the vector store connected to the specified Qdrant collection.
        """

        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self.vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=self.collection_name,
            embedding=self._get_embeddings(),
            url=f"{self.qdrant_host}:{self.qdrant_port}",
            timeout=120
        )
        return self.vector_store