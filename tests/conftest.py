import pickle

import pandas as pd
import pytest

from typing import Any


def read_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


@pytest.fixture
def training_data() -> pd.DataFrame:
    return read_pickle('./data/processed/training_data.pkl')


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.read_csv('./data/raw/data.csv')
