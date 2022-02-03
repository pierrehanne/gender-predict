"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    split_data,
    fit_model
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="splitted_data_node",
            ),
            node(
                func=fit_model,
                inputs=["X_train", "X_test", "y_train", "y_test", "parameters"],
                outputs=["fit_model", "test_metrics"],
                name="fit_model_node",
            ),
        ]
    )