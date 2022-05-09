import pandas as pd 

# Removed unused clumns
def remove_unused_columns(df,column_names):
    for col in column_names:
        if col in df.columns:
            df = df.drop(column_names,axis=1)
    return df

# Impute null values with None
def null_processing(feature_df):
    print("No of record with null values::", feature_df.isnull().sum() )
    columns = (feature_df.columns[feature_df.isnull().sum() > 0])
    print("Column having null values:: ", columns)
    feature_df.dropna(axis=0,inplace=True)
    return feature_df

def clean_data(df,remove_column_names):
    # remove unused column
    df = remove_unused_columns(df, remove_column_names)
    #impute null values
    feature_df = null_processing(df)
    return feature_df
