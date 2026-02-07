import os
import shutil
from datetime import datetime
from typing import Iterable, Tuple


def farm_path(base_dir: str, farm_slug: str) -> str:
    return os.path.join(base_dir, "farms", farm_slug)


def turbine_path(base_dir: str, farm_slug: str, turbine_slug: str) -> str:
    return os.path.join(farm_path(base_dir, farm_slug), "turbines", turbine_slug)


def ensure_turbine_dir(base_dir: str, farm_slug: str, turbine_slug: str) -> str:
    path = turbine_path(base_dir, farm_slug, turbine_slug)
    os.makedirs(path, exist_ok=True)
    return path


def data_csv_path(base_dir: str, farm_slug: str, turbine_slug: str) -> str:
    return os.path.join(turbine_path(base_dir, farm_slug, turbine_slug), "data.csv")


def feature_csv_path(base_dir: str, farm_slug: str, turbine_slug: str) -> str:
    return os.path.join(turbine_path(base_dir, farm_slug, turbine_slug), "feature_description.csv")


def rename_farm_dir(base_dir: str, old_slug: str, new_slug: str) -> None:
    if old_slug == new_slug:
        return
    old_path = farm_path(base_dir, old_slug)
    new_path = farm_path(base_dir, new_slug)
    if os.path.exists(old_path):
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)


def rename_turbine_dir(base_dir: str, farm_slug: str, old_slug: str, new_slug: str) -> None:
    if old_slug == new_slug:
        return
    old_path = turbine_path(base_dir, farm_slug, old_slug)
    new_path = turbine_path(base_dir, farm_slug, new_slug)
    if os.path.exists(old_path):
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)


def save_initial_csv(file_obj, dest_path: str) -> int:
    with open(dest_path, "wb") as handle:
        shutil.copyfileobj(file_obj, handle)
    return count_rows(dest_path)


def save_feature_csv(file_obj, dest_path: str) -> None:
    with open(dest_path, "wb") as handle:
        shutil.copyfileobj(file_obj, handle)


def count_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def append_row_and_trim(path: str, row: str, max_rows: int) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError("CSV data file not found")

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    header = lines[0] if lines else ""
    data_lines = lines[1:]
    data_lines.append(row.rstrip("\n") + "\n")

    if max_rows is not None and max_rows >= 0:
        while len(data_lines) > max_rows:
            data_lines.pop(0)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.writelines(data_lines)


def read_recent_rows(path: str, limit: int = 100) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    data_lines = lines[1:]
    return [line.rstrip("\n") for line in data_lines[-limit:]]
