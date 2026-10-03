import pandas as pd
import os

def create_dummy_table():
    # Creating the MultiIndex for columns (as if the table was transposed)
    # The original table has years as columns, and complex hierarchy as rows.
    # To simulate the complex indices, we will create it with complex rows and simple columns.
    
    row_tuples = [
        ("Total", "Outstanding Credit Guarantee Balance"),
        ("Institution", "Credit Guarantee Fund"),
        ("Institution", "Technology Credit Guarantee Fund"),
        ("Institution", "Gangwon Credit Guarantee Foundation"),
        ("Guarantee Type", "Loan Guarantee"),
        ("Guarantee Type", "Deposit Bank"),
        ("Guarantee Type", "Non-Banking Institution"),
        ("Guarantee Type", "Bill Guarantee"),
        ("Guarantee Type", "Others / Miscellaneous"),
        ("Industry", "Manufacturing"),
        ("Industry", "Construction"),
        ("Industry", "Wholesale and Retail Trade"),
        ("Industry", "Food and Accommodation Services"),
        ("Industry", "Others / Miscellaneous")
    ]
    
    index = pd.MultiIndex.from_tuples(row_tuples, names=["Level1", "Level2"])
    
    data = [
        [12824, 17282, 18887, 20041, 100],
        [6211, 8070, 8334, 8735, 43.6],
        [2765, 3569, 3736, 3723, 18.6],
        [3848, 5643, 6817, 7583, 37.8],
        [12234, 16492, 18204, 19345, 96.5],
        [9048, 11279, 12034, 12724, 63.5],
        [3186, 5214, 6170, 6621, 33],
        [368, 424, 384, 364, 1.8],
        [222, 366, 298, 332, 1.7],
        [4338, 5473, 5679, 5829, 29.1],
        [1766, 2260, 2312, 2363, 11.8],
        [3904, 5362, 5906, 6283, 31.4],
        [877, 1401, 1617, 1875, 9.4],
        [1939, 2786, 3374, 3691, 18.4]
    ]
    
    columns = pd.MultiIndex.from_product([["End of"], ["2008", "2009", "2010", "October 2011", "Composition Ratio"]])
    
    df = pd.DataFrame(data, index=index, columns=columns)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/type_iv_table.csv")
    print("Created data/type_iv_table.csv")

if __name__ == "__main__":
    create_dummy_table()
