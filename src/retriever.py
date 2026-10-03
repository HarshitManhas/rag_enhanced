from typing import List, Dict, Any
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager

class TableRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        
    def get_reranked_retriever(self, top_k: int = 25, rerank_top_n: int = 5):
        """
        Creates a retriever that first fetches top_k documents using dense retrieval,
        then reranks them to return top_n documents using FlashRank.
        """
        base_retriever = self.vector_store_manager.get_retriever(search_kwargs={"k": top_k})
        
        # Initialize FlashRank compressor
        compressor = FlashrankRerank(top_n=rerank_top_n)
        
        # Combine them into a ContextualCompressionRetriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return compression_retriever

    def retrieve(self, query: str, top_k: int = 25, rerank_top_n: int = 5) -> List[Document]:
        retriever = self.get_reranked_retriever(top_k=top_k, rerank_top_n=rerank_top_n)
        results = retriever.invoke(query)
        return results
