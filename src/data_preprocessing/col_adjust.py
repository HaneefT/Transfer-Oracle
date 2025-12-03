import pandas as pd

def adjust_column_names(file_path):
    d = {}
    df = pd.read_excel(file_path)
    for i in df.iloc:
        d[i[1]] = i[2]
    return d

print(adjust_column_names('../../data/AdjustedColDict_data/DF_Players_Column_Adjust.xlsx'))

