"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    split_data,
    init_device,
    init_neural_network
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
                func=init_device,
                inputs="parameters",
                outputs="device",
                name="init_device_node",
            ),
            node(
                func=init_neural_network,
                inputs=["device", "parameters"],
                outputs="model",
                name="init_neural_network_node",
            ),
        ]
    )
