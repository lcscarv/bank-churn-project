import csv
import os
import glob
import logging

import pandas as pd

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


def get_delimiter(file_path: str) -> str:
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(4096)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


def csv_loader(file_path: str) -> pd.DataFrame:
    delimiter = get_delimiter(file_path)
    df = pd.read_csv(file_path, sep=delimiter)
    return df


def obtain_most_recent_file(folder_path: str) -> str:
    file_list = glob.glob(os.path.join(folder_path, '*'))
    if not file_list:
        raise ValueError(f"No files in directory {folder_path}. Check your path or files")
    latest_file = max(file_list, key=os.path.getctime)
    return latest_file


def data_load_pipeline(folder_path: str) -> pd.DataFrame | None:
    logger.debug('Starting data load')
    most_recent_file = obtain_most_recent_file(folder_path)
    logger.debug(f'Most recent file path: {most_recent_file}. Loading data')
    data = csv_loader(most_recent_file)
    logger.debug('Data loaded')
    return data
