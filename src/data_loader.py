import pandas as pd
from typing import List, Dict, Any, Tuple
from src.hierarchical_node import format_table_into_documents
from src.table_parser import TableParser

def load_table(csv_path: str, index_col: List[int] = None, header: List[int] = None) -> pd.DataFrame:
    """
    Load a table using pandas. 
    If index_col and header are None, automatically detect them using the waterfall parser.
    """
    if index_col is None and header is None:
        parser = TableParser()
        structure = parser.detect_table_structure(csv_path)
        index_col = structure["index_col"]
        header = structure["header"]
        
    df = pd.read_csv(csv_path, index_col=index_col, header=header)
    return df

def classify_and_transpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify the table according to the paper (Type I, II vs Type III, IV).
    If the table index is a MultiIndex, it likely contains complex hierarchical information.
    We transpose it to make the index become the columns (headers), so we can extract hierarchical nodes.
    """
    is_complex_index = isinstance(df.index, pd.MultiIndex)
    
    # If the index is more complex than the columns, transpose.
    # For a simple heuristic: if index is multi-level and columns are single-level, definitely transpose.
    if is_complex_index:
        print("Complex index detected. Transposing the table to use indices as hierarchical headers.")
        df = df.transpose()
        
    return df

def process_table_file(csv_path: str, table_name: str, index_col: List[int] = None, header: List[int] = None) -> List[Dict[str, Any]]:
    """
    End-to-end processing of a table file into documents.
    """
    df = load_table(csv_path, index_col=index_col, header=header)
    df = classify_and_transpose(df)
    documents = format_table_into_documents(df, table_name=table_name)
    return documents

