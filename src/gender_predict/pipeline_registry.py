"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import data_engineering as de

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    # instantiate data engineering pipeline
    pipeline_data_engineering = de.create_pipeline()
    return {
        "__default__": Pipeline([pipeline_data_engineering]),
        "de": pipeline_data_engineering
    }