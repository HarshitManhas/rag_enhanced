import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from src.data_loader import process_table_file
from src.vector_store import VectorStoreManager
from src.retriever import TableRetriever
from src.generator import TableGenerator

def ingest_data(csv_path: str, table_name: str):
    """
    Ingest a CSV table, transform it, extract hierarchical nodes, and store in vector DB.
    """
    print(f"Ingesting table: {csv_path}")
    
    # Process table file will now auto-detect indices and headers
    documents = process_table_file(csv_path, table_name)
    
    vsm = VectorStoreManager()
    vsm.store_documents(documents)
    print(f"Successfully ingested {len(documents)} row representations into ChromaDB.")

def query_system(query: str):
    """
    Retrieve relevant rows and generate an answer using the RAG pipeline.
    """
    print(f"Query: {query}")
    
    vsm = VectorStoreManager()
    retriever = TableRetriever(vsm)
    generator = TableGenerator()
    
    print("Retrieving and reranking documents...")
    docs = retriever.retrieve(query)
    
    print("Retrieved context:")
    for doc in docs:
        print(f"- {doc.page_content}")
        
    print("\nGenerating answer...")
    answer = generator.generate_answer(query, docs)
    
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Node Tabular RAG")
    subparsers = parser.add_subparsers(dest="command")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a table")
    ingest_parser.add_argument("--csv", required=True, help="Path to CSV file")
    ingest_parser.add_argument("--name", required=True, help="Table name for metadata")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("--q", required=True, help="Question to ask")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_data(args.csv, args.name)
    elif args.command == "query":
        query_system(args.q)
    else:
        parser.print_help()
