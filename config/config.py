from pathlib import Path

# Project Root

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)
# Main Directories

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"

# Raw Dataset Files

AISLES_FILE = (
    RAW_DATA_DIR / "aisles.csv"
)

DEPARTMENTS_FILE = (
    RAW_DATA_DIR / "departments.csv"
)

PRODUCTS_FILE = (
    RAW_DATA_DIR / "products.csv"
)

ORDERS_FILE = (
    RAW_DATA_DIR / "orders.csv"
)

ORDER_PRODUCTS_PRIOR_FILE = (
    RAW_DATA_DIR /
    "order_products__prior.csv"
)

ORDER_PRODUCTS_TRAIN_FILE = (
    RAW_DATA_DIR /
    "order_products__train.csv"
)

# Processed Files

PRODUCT_FREQUENCY_FILE = (
    PROCESSED_DATA_DIR /
    "product_frequency.csv"
)

ASSOCIATION_RULES_FILE = (
    PROCESSED_DATA_DIR /
    "association_rules.csv"
)

RECOMMENDATION_RULES_FILE = (
    PROCESSED_DATA_DIR /
    "recommendation_rules.csv"
)

CLEANED_ORDERS_FILE = (
    PROCESSED_DATA_DIR /
    "cleaned_orders.csv"
)

BASKET_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "basket_data.csv"
)

# Model Files

ASSOCIATION_RULES_MODEL = (
    MODELS_DIR /
    "association_rules.pkl"
)

PRODUCT_ENCODER_MODEL = (
    MODELS_DIR /
    "product_encoder.pkl"
)

RECOMMENDATION_MODEL = (
    MODELS_DIR /
    "recommendation_model.pkl"
)

# Market Basket Parameters

TOP_N_PRODUCTS = 2000

MIN_SUPPORT = 0.01

MIN_CONFIDENCE = 0.30

MIN_LIFT = 1.20


# Processing Parameters

CHUNK_SIZE = 500_000

MAX_BASKET_ORDERS = 300_000

# Recommendation Parameters

DEFAULT_TOP_N_RECOMMENDATIONS = 5

# Create Required Directories

def create_directories():

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":

    create_directories()

    print(
        "SmartBasket AI directories "
        "verified successfully."
    )