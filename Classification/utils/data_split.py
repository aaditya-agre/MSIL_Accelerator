from sklearn.model_selection import train_test_split

def train_test_split_data(X, y, test_size=0.2, random_state=42,**kwargs):
    """
    Splits the input features and labels into training and testing sets for classification tasks.

    Parameters:
        X (array-like or DataFrame): Feature matrix.
        y (array-like or Series): Target labels (categorical/classification).
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Controls the shuffling for reproducibility (default is 42).

    Returns:
        X_train, X_test, y_train, y_test: Split datasets with stratified class distribution.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
