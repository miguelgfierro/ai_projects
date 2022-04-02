# Source of part of this code: https://github.com/Azure/fast_retraining
import os
import pandas as pd


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