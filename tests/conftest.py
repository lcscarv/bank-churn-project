import csv
import pickle

import pandas as pd
import pytest

from typing import Any


def get_delimiter(file_path: str) -> str:
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(4096)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


def read_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


@pytest.fixture
def training_data() -> pd.DataFrame:
    return read_pickle('./tests/data/processed/training_data.pkl')


@pytest.fixture
def data() -> pd.DataFrame:
    delimiter = get_delimiter('./tests/data/raw/data.csv')
    return pd.read_csv('./tests/data/raw/data.csv', delimiter=delimiter)
