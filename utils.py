import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def train_test_split_by_group(X, y, group_column, test_size=0.2, random_state=None):
    """
    Splits the data into train and test sets, ensuring that rows with the same group value
    (e.g., LOAN SEQUENCE NUMBER) remain in the same set.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature dataset.
        y (pd.Series or np.ndarray): Target dataset.
        group_column (str): The column name in X to group by (e.g., 'LOAN SEQUENCE NUMBER').
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    # Combine X and y into a single DataFrame for grouping
    data = X.copy()
    data['__target__'] = y
    data['__group__'] = X[group_column]

    # Get unique group identifiers
    unique_groups = data['__group__'].unique()

    # Split the unique groups into train and test sets
    train_groups, test_groups = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state
    )

    # Assign rows to train or test based on their group
    train_data = data[data['__group__'].isin(train_groups)]
    test_data = data[data['__group__'].isin(test_groups)]

    # Separate features and target for train and test sets
    X_train = train_data.drop(columns=['__target__', '__group__'])
    y_train = train_data['__target__']
    X_test = test_data.drop(columns=['__target__', '__group__'])
    y_test = test_data['__target__']

    return X_train, X_test, y_train, y_test


def split_dataframe_by_columns(df, columns = [
    'CREDIT SCORE',
    'ORIGINAL UPB',
    'ORIGINAL INTEREST RATE',
    'ORIGINAL LOAN-TO-VALUE (LTV)',
    'ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)',
    'ORIGINAL DEBT-TO-INCOME (DTI) RATIO',
    'ORIGINAL LOAN TERM',
    'MORTGAGE INSURANCE PERCENTAGE (MI %)',
    'ESTIMATED LOAN TO VALUE (ELTV)',
    'CURRENT ACTUAL UPB',
    'CURRENT INTEREST RATE',
    'INTEREST BEARING UPB',
    'CURRENT NON-INTEREST BEARING UPB',
    'LOAN AGE',
    'REMAINING MONTHS TO LEGAL MATURITY',
    'NUMBER OF UNITS',
    'NUMBER OF BORROWERS',
    'DELINQUENCY',
    'LAST MONTH DELINQUENCY STATUS',
    'MONTHS IN DELINQUENCY PAST 12'
]
):
    """
    Splits a DataFrame into two DataFrames:
    - One containing the specified columns
    - The other containing the remaining columns

    Parameters:
    df (pd.DataFrame): The input DataFrame
    columns (list): List of column names to include in the first DataFrame

    Returns:
    tuple: A tuple containing two DataFrames (df_with_columns, df_without_columns)
    """
    # Ensure the columns exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    # Create the two DataFrames
    df_with_columns = df[columns]
    df_without_columns = df.drop(columns=columns)
    
    return df_with_columns, df_without_columns

def reorder_dataframe(columns_order, df):
    """
    Reorders a DataFrame based on a given list of column names.
    Raises an error if there are columns in the DataFrame not in the list.

    Parameters:
    columns_order (list): List of column names in the desired order.
    df (pd.DataFrame): The input DataFrame to reorder.

    Returns:
    pd.DataFrame: A reordered DataFrame.
    """
    # Check if all columns in the DataFrame are in the provided list
    missing_columns = [col for col in df.columns if col not in columns_order]
    if missing_columns:
        raise ValueError(f"The following columns are not in the provided list: {missing_columns}")
    
    # Ensure the columns in the order list exist in the DataFrame
    columns_order = [col for col in columns_order if col in df.columns]
    
    # Reorder the DataFrame
    return df[columns_order]