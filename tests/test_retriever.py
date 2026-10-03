import pytest
from src.retriever import TableRetriever
from src.vector_store import VectorStoreManager

# A mock for the vector store manager to avoid needing chroma running for basic tests
class MockVectorStoreManager:
    def get_retriever(self, search_kwargs):
        class MockRetriever:
            def invoke(self, query):
                from langchain.docstore.document import Document
                return [Document(page_content="Mock doc 1"), Document(page_content="Mock doc 2")]
        return MockRetriever()

def test_table_retriever():
    vsm = MockVectorStoreManager()
    
    # We can't easily mock the FlashRank inside without complex mocking, 
    # but we can test if the class instantiates correctly.
    try:
        retriever = TableRetriever(vsm)
        # Note: testing retrieve() without actual FlashRank model might download the model or fail.
        # So we skip the actual .retrieve() call in the unit test unless we have the dependencies.
        assert retriever is not None
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")
