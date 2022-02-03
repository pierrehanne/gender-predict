"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import data_engineering as de, data_science as ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    # instantiate data engineering pipeline
    pipeline_data_engineering = de.create_pipeline()
    #instantiate data science pipeline
    pipeline_data_science = ds.create_pipeline()
    # pipelines
    return {
        "__default__": Pipeline([pipeline_data_engineering, pipeline_data_science]),
        "de": pipeline_data_engineering,
        "ds": pipeline_data_science,
        "train": pipeline_data_engineering + pipeline_data_science
    }