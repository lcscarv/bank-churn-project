import pandas as pd

from finance_churn_project.processing.data_processing import processing_pipeline


def test_processing_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    return processing_pipeline(data)
