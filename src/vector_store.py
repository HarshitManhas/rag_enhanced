from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "table_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
    def store_documents(self, documents: List[Dict[str, Any]]):
        """
        Store a list of document dicts into Chroma.
        The documents represent table rows with hierarchical headers.
        """
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(Document(page_content=doc["page_content"], metadata=doc["metadata"]))
            
        vectorstore = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            client=self.chroma_client
        )
        return vectorstore
        
    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        if search_kwargs is None:
            search_kwargs = {"k": 10}
            
        vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
