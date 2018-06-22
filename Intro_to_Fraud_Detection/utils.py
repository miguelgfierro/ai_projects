from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, log_loss, precision_score, recall_score
import numpy as np
import itertools
import matplotlib.pyplot as plt
import asyncio
import json
import sqlite3
from sqlite3 import Error


# Constants
BASELINE_MODEL = 'baseline.model'
BAD_REQUEST = 400
STATUS_OK = 200
NOT_FOUND = 404
SERVER_ERROR = 500
PORT = 5000
FRAUD_THRESHOLD = 0.5
DATABASE_FILE = 'db.sqlite3'


def connect_to_database(database=None):
    """
    Connect to a sqlite database. Don't forget to close the connection at the
    end of the routine with `conn.close()`.
    Args:
        database (str): Database filename.
    Returns:
        conn (object): Connector object.

    """
    if database is None:
        database = ':memory:'
    try:
        conn = sqlite3.connect(database)
    except Error as e:
        print(e)
        raise
    return conn


def create_table(cursor, table_name, table_instruction):
    """Create a table and drop it if it exists.
    Args:
        cursor (object): sqlite cursor.
        table_name (str): Table name.
        table_instruction (str): Table variable definition.
    Example:
        >>> conn = sqlite3.connect(':memory:')
        >>> cur = conn.cursor()
        >>> instruction = '''num INT(11) NULL, float_num FLOAT(8,6) NOT NULL DEFAULT "0",letter CHAR(1),input_date DATE'''
        >>> create_table('', cur, "table_name", instruction)
        >>> c = cur.execute('PRAGMA table_info([table_name]);')
        >>> cur.fetchall()
        [(0, 'num', 'INT(11)', 0, None, 0), (1, 'float_num', 'FLOAT(8,6)', 1, '"0"', 0), (2, 'letter', 'CHAR(1)', 0, None, 0), (3, 'input_date', 'DATE', 0, None, 0)]


    """
    query = "DROP TABLE IF EXISTS " + table_name + ";"
    cursor.execute(query)
    query = " CREATE TABLE " + table_name
    query += " (" + table_instruction + " );"
    cursor.execute(query)


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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
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
    report = {'Accuracy': m_acc, 'Precision': m_precision,
              'Recall': m_recall, 'F1': m_f1, 'Confusion Matrix': m_conf}
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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plots a confusion matrix.
    Args:
        cm (numpy array): The confusion matrix array.
        classes (list): List wit the classes names.
        normalize (bool): Flag to normalize data.
        title (str): Title of the plot.
        cmap (matplotlib.cm): Matplotlib colormap https://matplotlib.org/api/cm_api.html
    Examples:
        >>> import numpy as np
        >>> a = np.array([[10, 3, 0],[1, 2, 3],[1, 5, 9]])
        >>> classes = ['cl1', 'cl2', 'cl3']
        >>> plot_confusion_matrix(a, classes, normalize=False)
        >>> plot_confusion_matrix(a, classes, normalize=True)

    """
    cm_max = cm.max()
    cm_min = cm.min()
    if cm_min > 0:
        cm_min = 0
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_max = 1
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm_max / 2.
    plt.clim(cm_min, cm_max)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,
                 round(cm[i, j], 3),  # round to 3 decimals if they are float
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def chunked_http_client(num_chunks, s):
    # Use semaphore to limit number of requests
    semaphore = asyncio.Semaphore(num_chunks)

    @asyncio.coroutine
    # Return co-routine that will work asynchronously and respect locking of semaphore
    def http_get(url, payload, verbose):
        nonlocal semaphore
        with (yield from semaphore):
            headers = {'content-type': 'application/json'}
            response = yield from s.request('post', url, data=json.dumps(payload), headers=headers)
            if verbose:
                print("Response status:", response.status)
            body = yield from response.json()
            if verbose:
                print(body)
            yield from response.wait_for_close()
        return body
    return http_get


def run_load_test(url, payloads, _session, concurrent, verbose):
    http_client = chunked_http_client(num_chunks=concurrent, s=_session)

    # http_client returns futures, save all the futures to a list
    tasks = [http_client(url, payload, verbose) for payload in payloads]

    dfs_route = []
    # wait for futures to be ready then iterate over them
    for future in asyncio.as_completed(tasks):
        data = yield from future
        try:
            dfs_route.append(data)
        except Exception as err:
            print("Error {0}".format(err))
    return dfs_route
