# Source of part of this code:
# https://github.com/Azure/fast_retraining/
# https://github.com/miguelgfierro/codebase/
# https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html


import os
import numpy as np
import pandas as pd
from timeit import default_timer
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_airline():
    """Loads airline data.

    The dataset consists of a large amount of records, containing flight arrival and departure details for all the
    commercial flights within the USA, from October 1987 to April 2008. Its size is around 116 million records and
    5.76 GB of memory.

    There are 13 attributes, each represented in a separate column: Year (1987-2008), Month (1-12), Day of Month (1-31),
    Day of Week (1:Monday - 7:Sunday), CRS Departure Time (local time as hhmm), CRS Arrival Time (local time as hhmm),
    Unique Carrier, Flight Number, Actual Elapsed Time (in min), Origin, Destination, Distance (in miles), and Diverted
    (1=yes, 0=no).

    The target attribute is Arrival Delay, it is a positive or negative value measured in minutes.
    Link to the source: http://kt.ijs.si/elena_ikonomovska/data.html

    Returns:
        pd.DataFrame: A DataFrame.
    """
    cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    return pd.read_csv(os.path.join("data", "airline_14col.data"), names=cols)


def _get_nominal_integer_dict(nominal_vals):
    """Convert nominal values in integers, starting at 0.

    Args:
        nominal_vals (pd.Series): A series.

    Returns:
        dict: A dictionary with numeric values.
    """
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max + 1
    return d


def _convert_to_integer(srs, d):
    """Convert series to integer, given a dictionary.

    Args:
        srs (pd.Series): A series.
        d (dict): A dictionary mapping values to integers

    Returns:
        pd.Series: An series with numeric values.
    """
    return srs.map(lambda x: d[x])


def convert_related_cols_categorical_to_numeric(df, col_list):
    """Convert categorical columns, that are related between each other,
    to numeric and leave numeric columns as they are.

    Args:
        df (pd.DataFrame): Dataframe.
        col_list (list): List of columns.

    Returns:
        pd.DataFrame: A dataframe with numeric values.

    Examples:
        >>> df = pd.DataFrame({'letters':['a','b','c'],'letters2':['c','d','e'],'numbers':[1,2,3]})
        >>> df_numeric = convert_related_cols_categorical_to_numeric(df, col_list=['letters','letters2'])
        >>> print(df_numeric)
           letters  letters2  numbers
        0        0         2        1
        1        1         3        2
        2        2         4        3
    """
    ret = pd.DataFrame()
    values = None
    for c in col_list:
        values = pd.concat([values, df[c]], axis=0)
        values = pd.Series(values.unique())
    col_dict = _get_nominal_integer_dict(values)
    for column_name in df.columns:
        column = df[column_name]
        if column_name in col_list:
            ret[column_name] = _convert_to_integer(column, col_dict)
        else:
            ret[column_name] = column
    return ret


def convert_cols_categorical_to_numeric(df, col_list=None):
    """Convert categorical columns to numeric and leave numeric columns
    as they are. You can force to convert a numerical column if it is
    included in `col_list`.

    Args:
        df (pd.DataFrame): Dataframe.
        col_list (list): List of columns.

    Returns:
        pd.DataFrame: An dataframe with numeric values.

    Examples:
        >>> df = pd.DataFrame({'letters':['a','b','c'],'numbers':[1,2,3]})
        >>> df_numeric = convert_cols_categorical_to_numeric(df)
        >>> print(df_numeric)
           letters  numbers
        0        0        1
        1        1        2
        2        2        3
    """
    if col_list is None:
        col_list = []
    ret = pd.DataFrame()
    for column_name in df.columns:
        column = df[column_name]
        if column.dtype == "object" or column_name in col_list:
            col_dict = _get_nominal_integer_dict(column)
            ret[column_name] = _convert_to_integer(column, col_dict)
        else:
            ret[column_name] = column
    return ret


class Timer(object):
    """Timer class.

    Examples:
        >>> big_num = 100000
        >>> t = Timer()
        >>> t.start()
        >>> for i in range(big_num):
        >>>     r = 1
        >>> t.stop()
        >>> print(t.interval)
        0.0946876304844
        >>> with Timer() as t:
        >>>     for i in range(big_num):
        >>>         r = 1
        >>> print(t.interval)
        0.0766928562442
        >>> try:
        >>>     with Timer() as t:
        >>>         for i in range(big_num):
        >>>             r = 1
        >>>             raise(Exception("Get out!"))
        >>> finally:
        >>>     print(t.interval)
        0.0757778924471
    """

    def __init__(self):
        self._timer = default_timer

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start the timer."""
        self.start = self._timer()

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        self.interval = self.end - self.start


def binarize_prediction(y, threshold=0.5):
    return np.where(y > threshold, 1, 0)


def classification_metrics_binary(y_true, y_pred):
    m_acc = accuracy_score(y_true, y_pred)
    m_f1 = f1_score(y_true, y_pred)
    m_precision = precision_score(y_true, y_pred)
    m_recall = recall_score(y_true, y_pred)
    report = {
        "Accuracy": m_acc,
        "Precision": m_precision,
        "Recall": m_recall,
        "F1": m_f1,
    }
    return report


def classification_metrics_binary_prob(y_true, y_prob):
    m_auc = roc_auc_score(y_true, y_prob)
    report = {"AUC": m_auc}
    return report


def visualize_classifier(model, X, y, ax=None, cmap="rainbow"):
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(
        X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3
    )
    ax.axis("tight")
    ax.axis("off")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Predict with the estimator
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        levels=np.arange(n_classes + 1) - 0.5,
        cmap=cmap,
        clim=(y.min(), y.max()),
        zorder=1,
    )

    ax.set(xlim=xlim, ylim=ylim)


def visualize_tree(model, feature_names, class_names=None, figsize=(7, 7)):
    if class_names is None:
        class_names = list(map(str, feature_names))
    fig = plt.figure(figsize=figsize)
    _ = tree.plot_tree(
        model, feature_names=feature_names, class_names=class_names, filled=True
    )
