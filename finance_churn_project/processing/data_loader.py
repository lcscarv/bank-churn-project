import csv
import pandas as pd


def get_delimiter(file_path: str) -> str:
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(4096)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


def csv_loader(file_path: str) -> pd.DataFrame:
    delimiter = get_delimiter(file_path)
    df = pd.read_csv(file_path, sep=delimiter)
    return df
