"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""
from typing import Dict, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le


def split_data(model_input_table: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets

    Args:
        model_input_table (pd.DataFrame): Data containing features and target
        parameters (Dict): Parameters defined in yml

    Returns:
        Tuple: Splitted data
    """
    # features to predict label
    X = model_input_table[parameters["features"]]
    # label to predict
    y = le().fit_transform(model_input_table[parameters["target"]])
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = parameters["test_size"], random_state = parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test