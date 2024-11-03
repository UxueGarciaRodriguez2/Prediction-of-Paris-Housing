import pandas as pd

def identify_column_types(df):
    # cat columns
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    
    # num columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    return categorical_columns, numerical_columns