import csv
import os
import glob
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


def obtain_most_recent_file(folder_path: str) -> str:
    file_list = glob.glob(os.path.join(folder_path, '*'))
    latest_file = max(file_list, key=os.path.getctime)
    return latest_file
