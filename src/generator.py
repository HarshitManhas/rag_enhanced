from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class TableGenerator:
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        # We use gemini-3.8-flash by default
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that answers questions based on the provided table data context. "
                       "The context is formatted with hierarchical table headers to provide full meaning to each cell value. "
                       "Always base your answer strictly on the provided context. If the answer is not in the context, say 'I do not know'.\n\nContext:\n{context}"),
            ("user", "{question}")
        ])
        
        self.chain = self.prompt | self.llm

    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generates an answer from the retrieved documents.
        """
        # Format the context
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(doc.page_content)
            
        context = "\n---\n".join(context_parts)
        
        response = self.chain.invoke({"context": context, "question": query})
        
        # Handle cases where response.content is a list of blocks instead of a string
        if isinstance(response.content, list):
            # Extract text from the first block
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
            return str(response.content)
            
        return str(response.content)
