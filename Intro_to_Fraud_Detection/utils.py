from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, log_loss, precision_score, recall_score
import numpy as np


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



def classification_metrics_binary(y_true, y_pred):
    """Returns a report with different metrics for a binary classification problem.
    - Accuracy: Number of correct predictions made as a ratio of all predictions. Useful when there are equal number
    of observations in each class and all predictions and prediction errors are equally important.
    - Confusion matrix: C_ij where observations are known to be in group i but predicted to be in group j. In binary
    classification true negatives is C_00, false negatives is C_10, true positives is C_11 and false positives is C_01.
    - Precision: Number of true positives divided by the number of true and false positives. It is the ability of the
    classifier not to label as positive a sample that is negative.
    - Recall: Number of true positives divided by the number of true positives and false negatives. It is the ability
    of the classifier to find all the positive samples.
    High Precision and low Recall will return few positive results but most of them will be correct. 
    High Recall and low Precision will return many positive results but most of them will be incorrect.
    - F1 Score: 2*((precision*recall)/(precision+recall)). It measures the balance between precision and recall.
    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels (binary).
    Returns:
        report (dict): Dictionary with metrics.
    Examples:
        >>> from collections import OrderedDict
        >>> y_true = [0,1,0,0,1]
        >>> y_pred = [0,1,0,1,1]
        >>> result = classification_metrics_binary(y_true, y_pred)
        >>> OrderedDict(sorted(result.items()))
        OrderedDict([('Accuracy', 0.8), ('Confusion Matrix', array([[2, 1],
               [0, 2]])), ('F1', 0.8), ('Precision', 0.6666666666666666), ('Recall', 1.0)])
    """
    m_acc = accuracy_score(y_true, y_pred)
    m_f1 = f1_score(y_true, y_pred)
    m_precision = precision_score(y_true, y_pred)
    m_recall = recall_score(y_true, y_pred)
    m_conf = confusion_matrix(y_true, y_pred)
    report = {'Accuracy': m_acc, 'Precision': m_precision, 'Recall': m_recall, 'F1': m_f1, 'Confusion Matrix': m_conf}
    return report


def classification_metrics_binary_prob(y_true, y_prob):
    """Returns a report with different metrics for a binary classification problem.
    - AUC: The Area Under the Curve represents the ability to discriminate between positive and negative classes. An
    area of 1 represent perfect scoring and an area of 0.5 means random guessing.
    - Log loss: Also called logistic regression loss or cross-entropy loss. It quantifies the performance by
    penalizing false classifications. Minimizing the Log Loss is equivalent to minimizing the squared error but using
    probabilistic predictions. Log loss penalize heavily classifiers that are confident about incorrect classifications.
    Args:
        y_true (list or array): True labels.
        y_prob (list or array): Predicted labels (probability).
    Returns:
        report (dict): Dictionary with metrics.
    Examples:
        >>> from collections import OrderedDict
        >>> y_true = [0,1,0,0,1]
        >>> y_prob = [0.2,0.7,0.4,0.3,0.2]
        >>> result = classification_metrics_binary_prob(y_true, y_prob)
        >>> OrderedDict(sorted(result.items()))
        OrderedDict([('AUC', 0.5833333333333333), ('Log loss', 0.6113513950783531)])
        >>> y_prob = [0.2,0.7,0.4,0.3,0.3]
        >>> result = classification_metrics_binary_prob(y_true, y_prob)
        >>> OrderedDict(sorted(result.items()))
        OrderedDict([('AUC', 0.75), ('Log loss', 0.5302583734567203)])
    """
    m_auc = roc_auc_score(y_true, y_prob)
    m_logloss = log_loss(y_true, y_prob)
    report = {'AUC': m_auc, 'Log loss': m_logloss}
    return report


def binarize_prediction(y, threshold=0.5):
    y_pred = np.where(y > threshold, 1, 0)
    return y_pred
