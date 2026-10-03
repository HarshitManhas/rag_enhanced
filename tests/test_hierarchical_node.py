import pandas as pd
from src.hierarchical_node import extract_hierarchical_headers, format_table_into_documents

def test_extract_hierarchical_headers():
    row_tuples = [
        ("Total", "Outstanding Credit Guarantee Balance"),
        ("Institution", "Credit Guarantee Fund")
    ]
    index = pd.MultiIndex.from_tuples(row_tuples, names=["Level1", "Level2"])
    
    # We pretend this index was transposed into columns
    df = pd.DataFrame(columns=index)
    
    paths = extract_hierarchical_headers(df)
    assert len(paths) == 2
    assert "Root: Total, and Child: Outstanding Credit Guarantee Balance" in paths[0]
    assert "Root: Institution, and Child: Credit Guarantee Fund" in paths[1]

def test_format_table_into_documents():
    row_tuples = [
        ("Total", "Outstanding Credit Guarantee Balance"),
        ("Institution", "Credit Guarantee Fund")
    ]
    columns = pd.MultiIndex.from_tuples(row_tuples, names=["Level1", "Level2"])
    data = [
        [100, 200],
        [300, 400]
    ]
    df = pd.DataFrame(data, columns=columns, index=["Row1", "Row2"])
    
    docs = format_table_into_documents(df, "TestTable")
    
    assert len(docs) == 2
    assert "content: Row1" in docs[0]["page_content"]
    assert "Root: Total, and Child: Outstanding Credit Guarantee Balance: 100" in docs[0]["page_content"]
    assert "Root: Institution, and Child: Credit Guarantee Fund: 200" in docs[0]["page_content"]
