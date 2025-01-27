import pandas as pd

from src.processing.data_loader import obtain_most_recent_file
from src.processing.data_processing import processing_pipeline


def test_processing_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    return processing_pipeline(data)


def test_obtain_most_recent_file() -> str:
    return obtain_most_recent_file(folder_path='./tests/data/raw/')
