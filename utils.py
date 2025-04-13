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