import csv
import json
import os
from typing import List, Tuple
from google import genai
from google.genai import types

class TableParser:
    def __init__(self):
        # We will initialize the Gemini client only when needed for the fallback
        self.client = None

    def _get_client(self):
        if self.client is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self.client = genai.Client(api_key=api_key)
        return self.client

    def _read_csv_snippet(self, csv_path: str, max_rows: int = 10) -> List[List[str]]:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [row for _, row in zip(range(max_rows), reader)]

    def heuristic_parse(self, rows: List[List[str]]) -> Tuple[int, int]:
        """
        Attempts to find a top-left block of empty cells.
        Returns (header_rows_count, index_cols_count).
        If it cannot confidently determine, returns (-1, -1).
        """
        if not rows or not rows[0]:
            return -1, -1

        # Count empty cells in the first row to determine index_cols
        index_cols = 0
        for cell in rows[0]:
            if not cell.strip():
                index_cols += 1
            else:
                break
                
        if index_cols == 0:
            # Maybe it's a simple flat table with 1 header row and 0 index cols
            # Let's verify if row 0 is strings and row 1 is numbers
            if len(rows) > 1:
                try:
                    # check if row 1 has numbers
                    [float(c) for c in rows[1] if c.strip()]
                    return 1, 0
                except ValueError:
                    return -1, -1
            return 1, 0

        # Now find how many rows have these first `index_cols` empty
        header_rows = 0
        for row in rows:
            # Check if the first `index_cols` are empty
            is_empty_block = all(not cell.strip() for cell in row[:index_cols])
            if is_empty_block:
                header_rows += 1
            else:
                break
                
        # Check for Pandas multi-index name row
        # Pandas often puts index names in the row right below the empty block,
        # where the left part has names, and the right part is empty.
        if len(rows) > header_rows:
            next_row = rows[header_rows]
            left_has_text = any(c.strip() for c in next_row[:index_cols])
            right_is_empty = all(not c.strip() for c in next_row[index_cols:])
            if left_has_text and right_is_empty:
                header_rows += 1

        if header_rows > 0 and index_cols > 0:
            return header_rows, index_cols
            
        return -1, -1

    def llm_parse(self, rows: List[List[str]]) -> Tuple[int, int]:
        """
        Fall back to Gemini to intelligently parse the table boundaries.
        """
        csv_string = "\n".join([",".join(r) for r in rows])
        
        prompt = f"""
You are an expert data analyst. Look at the following raw CSV snippet and determine the number of header rows and index columns.
Headers are the top rows that label the data columns. 
Indices are the leftmost columns that label the data rows (like multi-level indices).
Pay attention to empty cells in the top left, which usually indicate the dimensions of headers and indices.

CSV Snippet:
{csv_string}

Return a JSON object with EXACTLY two integer keys:
"header_rows": number of rows that make up the column headers
"index_cols": number of columns that make up the row indices
"""
        client = self._get_client()
        
        # Use gemini-2.5-flash with structured output
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        try:
            result = json.loads(response.text)
            return result.get("header_rows", 1), result.get("index_cols", 0)
        except Exception as e:
            print(f"LLM parsing failed: {e}. Falling back to default 1 header, 0 index.")
            return 1, 0

    def detect_table_structure(self, csv_path: str) -> dict:
        """
        Waterfall approach: Try heuristic, fallback to LLM.
        Returns a dict suitable for pandas read_csv arguments.
        """
        snippet = self._read_csv_snippet(csv_path)
        
        h_rows, i_cols = self.heuristic_parse(snippet)
        
        if h_rows == -1 or i_cols == -1:
            print("Heuristic approach ambiguous. Falling back to LLM for table parsing...")
            h_rows, i_cols = self.llm_parse(snippet)
        else:
            print(f"Heuristic successfully detected table structure.")
            
        print(f"Detected {h_rows} header row(s) and {i_cols} index column(s).")
            
        header_arg = list(range(h_rows)) if h_rows > 1 else (0 if h_rows == 1 else None)
        index_col_arg = list(range(i_cols)) if i_cols > 1 else (0 if i_cols == 1 else None)
        
        return {
            "header": header_arg,
            "index_col": index_col_arg
        }
