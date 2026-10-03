import pandas as pd
from typing import List, Dict, Any

class TreeNode:
    def __init__(self, value: str):
        self.value = value
        self.children = []
        self.parent = None

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

    def print_tree(self, level=0):
        indent = "  " * level
        print(f"{indent}{self.value}")
        for child in self.children:
            child.print_tree(level + 1)

def extract_hierarchical_headers(df: pd.DataFrame) -> List[str]:
    """
    Given a dataframe, extract the hierarchical headers.
    If the table has complex indices (like Type III and IV), it should be transposed first.
    Assume df is already transposed if needed.
    This function will read the multi-index columns and generate string paths like:
    "Root: A, and Child: B, and Grandchild: C"
    """
    header_paths = []
    
    # If the dataframe has a MultiIndex for columns
    if isinstance(df.columns, pd.MultiIndex):
        for col_tuple in df.columns:
            path_parts = [str(p).strip() for p in col_tuple if pd.notna(p) and str(p).strip() != "" and not str(p).startswith("Unnamed")]
            path_str = build_path_string(path_parts)
            header_paths.append(path_str)
    else:
        # Single level columns
        for col in df.columns:
            header_paths.append(f"Root: {col}")
            
    return header_paths

def build_path_string(parts: List[str]) -> str:
    if not parts:
        return "Unknown"
    
    path = f"Root: {parts[0]}"
    if len(parts) > 1:
        path += f", and Child: {parts[1]}"
    if len(parts) > 2:
        path += f", and Grandchild: {parts[2]}"
    # If more levels exist, append them similarly or generically
    for i in range(3, len(parts)):
        path += f", and Level_{i+1}: {parts[i]}"
        
    return path

def format_table_into_documents(df: pd.DataFrame, table_name: str = "") -> List[Dict[str, Any]]:
    """
    Converts a DataFrame into a list of documents (one per row).
    Uses the hierarchical headers for columns.
    """
    header_paths = extract_hierarchical_headers(df)
    
    documents = []
    
    for idx, row in df.iterrows():
        row_content_parts = []
        row_content_parts.append(f"content: {idx}")
        
        for i, val in enumerate(row):
            if pd.isna(val) or str(val).strip() == "":
                continue
            
            col_header = header_paths[i]
            row_content_parts.append(f"{col_header}: {val}")
            
        doc_text = " ".join(row_content_parts)
        
        documents.append({
            "page_content": doc_text,
            "metadata": {
                "source": table_name,
                "type": "table_row",
                "row_index": str(idx)
            }
        })
        
    return documents
