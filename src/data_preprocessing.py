from pathlib import Path
import pandas as pd

# Paths

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

RAW_DIR = (
    PROJECT_ROOT /
    "data" /
    "raw"
)

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

# Load Small Datasets

def load_small_datasets():
    """
    Load manageable Instacart datasets.
    """

    aisles = pd.read_csv(
        RAW_DIR / "aisles.csv"
    )

    departments = pd.read_csv(
        RAW_DIR / "departments.csv"
    )

    products = pd.read_csv(
        RAW_DIR / "products.csv"
    )

    orders = pd.read_csv(
        RAW_DIR / "orders.csv"
    )

    return (
        aisles,
        departments,
        products,
        orders
    )

# Optimize Data Types

def optimize_dataframe(
    dataframe
):
    """
    Reduce memory usage of numeric columns.
    """

    df = dataframe.copy()

    for column in df.select_dtypes(
        include=["int64"]
    ).columns:

        df[column] = pd.to_numeric(
            df[column],
            downcast="integer"
        )

    for column in df.select_dtypes(
        include=["float64"]
    ).columns:

        df[column] = pd.to_numeric(
            df[column],
            downcast="float"
        )

    return df

# Read Prior Transactions in Chunks

def read_prior_chunks(
    chunksize=500_000
):
    """
    Generator for memory-efficient reading
    of order_products__prior.csv.
    """

    file_path = (
        RAW_DIR /
        "order_products__prior.csv"
    )

    return pd.read_csv(
        file_path,
        chunksize=chunksize,
        usecols=[
            "order_id",
            "product_id",
            "add_to_cart_order",
            "reordered"
        ],
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "add_to_cart_order": "int16",
            "reordered": "int8"
        }
    )

# Product Frequency

def calculate_product_frequency(
    chunksize=500_000
):
    """
    Calculate product purchase frequency
    using chunk-based processing.
    """

    frequency = {}

    for chunk in read_prior_chunks(
        chunksize=chunksize
    ):

        counts = (
            chunk["product_id"]
            .value_counts()
        )

        for product_id, count in counts.items():

            frequency[product_id] = (
                frequency.get(
                    product_id,
                    0
                )
                + int(count)
            )

    result = pd.DataFrame(
        frequency.items(),
        columns=[
            "product_id",
            "purchase_count"
        ]
    )

    result = result.sort_values(
        "purchase_count",
        ascending=False
    ).reset_index(
        drop=True
    )

    return result

# Save Product Frequency

def save_product_frequency(
    product_frequency
):
    """
    Save product frequency to processed data.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DIR /
        "product_frequency.csv"
    )

    product_frequency.to_csv(
        output_path,
        index=False
    )

    return output_path


# Reorder Statistics

def calculate_reorder_statistics(
    chunksize=500_000
):
    """
    Calculate reordered vs non-reordered
    product purchase statistics.
    """

    reordered = 0

    not_reordered = 0

    for chunk in pd.read_csv(
        RAW_DIR /
        "order_products__prior.csv",
        chunksize=chunksize,
        usecols=["reordered"],
        dtype={
            "reordered": "int8"
        }
    ):

        counts = (
            chunk["reordered"]
            .value_counts()
        )

        reordered += int(
            counts.get(1, 0)
        )

        not_reordered += int(
            counts.get(0, 0)
        )

    total = (
        reordered +
        not_reordered
    )

    return {
        "reordered": reordered,
        "not_reordered": not_reordered,
        "total": total,
        "reorder_rate": (
            reordered / total
            if total > 0
            else 0
        )
    }