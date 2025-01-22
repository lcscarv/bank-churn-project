import re


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
