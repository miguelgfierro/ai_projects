from sklearn.model_selection import train_test_split


def split_train_test(X, y, test_size=0.2):
    """Split a dataset into train and test sets.
    Args:
        X (np.array or pd.DataFrame): Features.
        y (np.array or pd.DataFrame): Labels.
        test_size (float): Percentage in the test set.
    Returns:
        X_train, X_test, y_train, y_test (list): List with the dataset splitted.
    Example:
        >>> import numpy as np
        >>> X = np.random.randint(0,10, (100,5))
        >>> y = np.random.randint(0,1, 100)
        >>> X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
        >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        (80, 5) (20, 5) (80,) (20,)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

