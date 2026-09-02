from pathlib import Path

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules
)

# Project Paths

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

# Select Frequent Products

def select_frequent_products(
    product_frequency,
    top_n=2000
):
    """
    Select top N frequently purchased products.
    """

    return (
        product_frequency
        .sort_values(
            "purchase_count",
            ascending=False
        )
        .head(top_n)
        ["product_id"]
        .tolist()
    )

# Create Sparse Basket Matrix

def create_sparse_basket_matrix(
    transactions
):
    """
    Create a sparse order-product matrix.

    Rows    -> Orders
    Columns -> Products
    Values  -> Product presence
    """

    order_codes, order_ids = pd.factorize(
        transactions["order_id"]
    )

    product_codes, product_ids = pd.factorize(
        transactions["product_id"]
    )

    matrix = csr_matrix(
        (
            np.ones(
                len(transactions),
                dtype=np.int8
            ),
            (
                order_codes,
                product_codes
            )
        ),
        shape=(
            len(order_ids),
            len(product_ids)
        )
    )

    basket_matrix = (
        pd.DataFrame
        .sparse
        .from_spmatrix(
            matrix,
            index=order_ids,
            columns=product_ids
        )
    )

    basket_matrix = (
        basket_matrix
        .astype(
            pd.SparseDtype(
                bool,
                fill_value=False
            )
        )
    )

    return basket_matrix

# Generate Frequent Itemsets

def generate_frequent_itemsets(
    basket_matrix,
    min_support=0.01
):
    """
    Generate frequent itemsets using FP-Growth.
    """

    return fpgrowth(
        basket_matrix,
        min_support=min_support,
        use_colnames=True
    )

# Generate Association Rules

def generate_association_rules(
    frequent_itemsets,
    min_lift=1.0
):
    """
    Generate association rules using lift.
    """

    if frequent_itemsets.empty:

        return pd.DataFrame()

    return association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=min_lift
    )

# Filter Strong Rules

def filter_strong_rules(
    rules,
    min_confidence=0.30,
    min_lift=1.20
):
    """
    Filter association rules by confidence and lift.
    """

    if rules.empty:

        return rules.copy()

    result = rules[
        (
            rules["confidence"]
            >= min_confidence
        )
        &
        (
            rules["lift"]
            >= min_lift
        )
    ].copy()

    return (
        result
        .sort_values(
            [
                "lift",
                "confidence",
                "support"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

# Convert Rules for Saving

def prepare_rules_for_saving(
    rules
):
    """
    Convert frozensets into comma-separated strings.
    """

    result = rules.copy()

    result["antecedents"] = (
        result["antecedents"]
        .apply(
            lambda x:
            ", ".join(
                sorted(x)
            )
        )
    )

    result["consequents"] = (
        result["consequents"]
        .apply(
            lambda x:
            ", ".join(
                sorted(x)
            )
        )
    )

    return result

# Save Association Rules

def save_association_rules(
    rules,
    filename="association_rules.csv"
):
    """
    Save association rules to processed directory.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DIR /
        filename
    )

    rules_to_save = (
        prepare_rules_for_saving(
            rules
        )
    )

    rules_to_save.to_csv(
        output_path,
        index=False
    )

    return output_path