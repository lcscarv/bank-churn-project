import os
import re
import glob


def get_file_list(folder_path: str) -> list[str] | None:
    file_list = glob.glob(os.path.join(folder_path, '*'))
    if not file_list:
        raise ValueError(f"No files in directory {folder_path}. Check your path or files")
    return file_list


def obtain_most_recent_file(folder_path: str) -> str:
    file_list = get_file_list(folder_path)
    latest_file = max(file_list, key=os.path.getctime)  # type: ignore
    return latest_file


def obtain_score_from_name(file_name: str) -> float | None:
    regex_pattern = r'_(\d+\.\d+)\.pkl$'
    match = re.match(regex_pattern, file_name)
    if match:
        score = float(match.group('score'))
        return score


def get_best_model_path(file_paths: list[str]) -> str:
    def sorting_function(x: str) -> float:
        score = obtain_score_from_name(x)
        if score is None:
            return float('inf')
        return score

    sorted_paths = sorted(file_paths, key=sorting_function)
    return sorted_paths[0]
