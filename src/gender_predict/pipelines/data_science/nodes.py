"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.sklearn import autolog, eval_and_log_metrics


def split_data(model_input_table: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets

    Args:
        model_input_table (pd.DataFrame): Data containing features and target
        parameters (Dict): Parameters defined in yml

    Returns:
        Tuple: Splitted data
    """
    # define features / label
    X = model_input_table[parameters["features"]].to_numpy()
    y = le().fit_transform(model_input_table[parameters["target"]])
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = parameters["test_size"], random_state = parameters["random_state"], stratify = y
    )
    return X_train, X_test, y_train, y_test


def __print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def fit_model(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, parameters: Dict) -> Tuple[Pipeline, eval_and_log_metrics]:
    """train classifier

    Args:
        X_train (np.array): train features
        X_test (np.array): train label
        y_train (np.array): test features
        y_test (np.array): test label
        parameters (Dict): yaml conf

    Returns:
        Pipeline: Fitted Pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            analyzer=parameters["analyzer"],
            ngram_range=(parameters["ngram_range_start"],parameters["ngram_range_end"]))
        ),
        ('clf', LinearSVC(max_iter=3000))])


    # create experiment
    mlflow.create_experiment(parameters["name_experiment"])
    # enable autologging from mlflow
    autolog()
    # launch mlflow run
    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        metrics = eval_and_log_metrics(pipeline, X_test, y_test, prefix="test_")

    # fetch the auto logged parameters and metrics for ended run
    __print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    return pipeline, metrics