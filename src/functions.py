import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def identify_column_types(df):
    # cat columns
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    
    # num columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    return categorical_columns, numerical_columns


def convert_category(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f" '{col}' does not exists")
    return df


def boxplot_outliers(data, numeric_columns, n_cols=3):
    # Calculate the number of rows required
    n_rows = int(np.ceil(len(numeric_columns) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes = axes.flatten() 
    
    for i, col in enumerate(numeric_columns):
        sns.boxplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f'Outliers {col}')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()  
    plt.show()


#MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate the MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100