"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node

from .nodes import(
    preprocess_belgium_name,
    preprocess_canadian_name,
    preprocess_french_name,
    preprocess_american_name,
    create_model_input_table
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_belgium_name,
                inputs="belgium_name",
                outputs="preprocessed_belgium_name",
                name="preprocess_belgium_node",
            ),
            node(
                func=preprocess_canadian_name,
                inputs="canadian_name",
                outputs="preprocessed_canadian_name",
                name="preprocess_canadian_node",
            ),
            node(
                func=preprocess_french_name,
                inputs=["fr_french_name", "idf_french_name"],
                outputs="preprocessed_french_name",
                name="preprocess_french_node",
            ),
            node(
                func=preprocess_american_name,
                inputs=["nyc_american_name", "usa_american_name"],
                outputs="preprocessed_american_name",
                name="preprocess_american_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_belgium_name", "preprocessed_canadian_name", "preprocessed_french_name", "preprocessed_american_name", "parameters"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            )
        ]
    )