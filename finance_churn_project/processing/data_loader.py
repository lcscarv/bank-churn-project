import csv
import logging
import pandas as pd
from utils.general_utils import obtain_most_recent_file

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


def data_load_pipeline(folder_path: str) -> pd.DataFrame | None:
    logger.debug('Starting data load')
    most_recent_file = obtain_most_recent_file(folder_path)
    logger.debug(f'Most recent file path: {most_recent_file}. Loading data')
    data = csv_loader(most_recent_file)
    logger.debug('Data loaded')
    return data
