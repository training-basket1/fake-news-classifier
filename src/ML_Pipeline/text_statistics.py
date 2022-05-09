import pandas as pd 

def text_statistics(df,col):
    result = df[col].str.split().str.len()
    return result.describe()