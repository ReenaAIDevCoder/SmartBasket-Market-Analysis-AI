from pathlib import Path
import pandas as pd

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

# Directory Creation

def create_project_directories():
    """
    Create required project directories if they
    do not already exist.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

# CSV Loader

def load_csv(
    file_path,
    **kwargs
):
    """
    Load a CSV file into a pandas DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        **kwargs
    )

# CSV Saver

def save_csv(
    dataframe,
    file_path
):
    """
    Save a DataFrame to CSV.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        file_path,
        index=False
    )

    print(
        f"File saved successfully: {file_path}"
    )

# Dataset Information

def dataset_summary(
    dataframe,
    dataset_name="Dataset"
):
    """
    Display basic dataset information.
    """

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"{dataset_name} Summary"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        "Rows:",
        f"{len(dataframe):,}"
    )

    print(
        "Columns:",
        len(dataframe.columns)
    )

    print(
        "\nColumns:"
    )

    print(
        dataframe.columns.tolist()
    )

    print(
        "\nMissing Values:"
    )

    print(
        dataframe.isnull().sum()
    )

# Memory Usage

def get_memory_usage(
    dataframe
):
    """
    Return DataFrame memory usage in MB.
    """

    memory_mb = (
        dataframe.memory_usage(
            deep=True
        ).sum()
        / (1024 ** 2)
    )

    return memory_mb