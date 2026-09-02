import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Page Configuration

st.set_page_config(
    page_title="SmartBasket AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# Paths

PRODUCTS_PATH = "../data/raw/products.csv"
ORDERS_PATH = "../data/raw/orders.csv"

PRODUCT_FREQUENCY_PATH = (
    "../data/processed/product_frequency.csv"
)

RULES_PATH = (
    "../data/processed/association_rules.csv"
)

# Load Data

@st.cache_data
def load_products():

    return pd.read_csv(
        PRODUCTS_PATH
    )


@st.cache_data
def load_orders():

    return pd.read_csv(
        ORDERS_PATH
    )


@st.cache_data
def load_product_frequency():

    return pd.read_csv(
        PRODUCT_FREQUENCY_PATH
    )


@st.cache_data
def load_rules():

    return pd.read_csv(
        RULES_PATH
    )

# Load Dataset Safely

try:

    products = load_products()

    orders = load_orders()

    product_frequency = (
        load_product_frequency()
    )

    rules = load_rules()

except FileNotFoundError as error:

    st.error(
        "Required dataset file is missing."
    )

    st.code(str(error))

    st.stop()

# Header

st.title(
    "📊 SmartBasket AI Dashboard"
)

st.markdown(
    """
    ### Intelligent Grocery Shopping Analytics

    Explore customer ordering behavior, product popularity,
    reorder patterns, grocery departments, and association-rule
    insights generated from the Instacart Online Grocery Dataset.
    """
)

st.divider()

# Sidebar

st.sidebar.title(
    "🎛️ Dashboard Controls"
)

st.sidebar.info(
    """
    Use the dashboard to explore:

    • Customer behavior
    • Product popularity
    • Shopping time
    • Grocery departments
    • Association rules
    """
)

# KPI SECTION

st.header(
    "📌 Key Performance Indicators"
)

total_customers = (
    orders["user_id"].nunique()
)

total_orders = (
    orders["order_id"].nunique()
)

total_products = (
    products["product_id"].nunique()
)

total_departments = (
    products["department_id"].nunique()
)

total_aisles = (
    products["aisle_id"].nunique()
)

total_rules = (
    len(rules)
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

with col2:

    st.metric(
        "🛒 Orders",
        f"{total_orders:,}"
    )

with col3:

    st.metric(
        "📦 Products",
        f"{total_products:,}"
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "🏬 Departments",
        f"{total_departments:,}"
    )

with col5:

    st.metric(
        "🗂️ Aisles",
        f"{total_aisles:,}"
    )

with col6:

    st.metric(
        "🔗 Association Rules",
        f"{total_rules:,}"
    )


st.divider()

# PRODUCT POPULARITY

st.header(
    "🏆 Top Purchased Products"
)

top_n = st.slider(
    "Select number of products",
    min_value=5,
    max_value=20,
    value=10
)

top_products = (
    product_frequency
    .sort_values(
        "purchase_count",
        ascending=False
    )
    .head(top_n)
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    top_products["product_name"][::-1],
    top_products["purchase_count"][::-1]
)

ax.set_title(
    "Most Purchased Grocery Products"
)

ax.set_xlabel(
    "Purchase Count"
)

ax.set_ylabel(
    "Product"
)

plt.tight_layout()

st.pyplot(fig)

# SHOPPING TIME ANALYSIS

st.header(
    "⏰ Shopping Time Analysis"
)

orders_by_hour = (
    orders["order_hour_of_day"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    orders_by_hour.index,
    orders_by_hour.values,
    marker="o"
)

ax.set_title(
    "Orders by Hour of Day"
)

ax.set_xlabel(
    "Hour of Day"
)

ax.set_ylabel(
    "Number of Orders"
)

ax.set_xticks(
    range(0, 24)
)

plt.tight_layout()

st.pyplot(fig)

# DAY OF WEEK ANALYSIS

st.header(
    "📅 Orders by Day of Week"
)

day_names = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}

orders_by_day = (
    orders["order_dow"]
    .value_counts()
    .sort_index()
)

day_labels = [
    day_names[i]
    for i in orders_by_day.index
]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    day_labels,
    orders_by_day.values
)

ax.set_title(
    "Orders by Day of Week"
)

ax.set_xlabel(
    "Day"
)

ax.set_ylabel(
    "Number of Orders"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

st.pyplot(fig)

# DEPARTMENT ANALYSIS

st.header(
    "🏬 Grocery Department Analysis"
)

department_counts = (
    products
    .groupby("department_id")
    ["product_id"]
    .count()
    .sort_values(
        ascending=False
    )
)

department_counts = (
    department_counts
    .reset_index()
    .merge(
        products[
            [
                "department_id"
            ]
        ].drop_duplicates(),
        on="department_id",
        how="left"
    )
)


# Load department names
DEPARTMENT_PATH = (
    "../data/raw/departments.csv"
)

departments = pd.read_csv(
    DEPARTMENT_PATH
)

department_counts = (
    department_counts
    .merge(
        departments,
        on="department_id",
        how="left"
    )
)


department_counts = (
    department_counts
    .sort_values(
        "product_id",
        ascending=False
    )
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.barh(
    department_counts["department"][::-1],
    department_counts["product_id"][::-1]
)

ax.set_title(
    "Number of Products by Department"
)

ax.set_xlabel(
    "Number of Products"
)

ax.set_ylabel(
    "Department"
)

plt.tight_layout()

st.pyplot(fig)


# PRODUCT FREQUENCY TABLE

st.header(
    "📦 Product Purchase Frequency"
)

display_columns = [
    "product_id",
    "product_name",
    "purchase_count"
]

st.dataframe(
    product_frequency[
        display_columns
    ].head(20),
    use_container_width=True,
    hide_index=True
)

# ASSOCIATION RULE ANALYSIS

st.header(
    "🔗 Market Basket Association Rules"
)

st.markdown(
    """
    Association rules identify products that tend to appear
    together in customer shopping baskets.
    """
)


# Rule filters

col1, col2 = st.columns(2)

with col1:

    confidence_threshold = st.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )

with col2:

    lift_threshold = st.slider(
        "Minimum Lift",
        min_value=1.0,
        max_value=5.0,
        value=1.20,
        step=0.10
    )


filtered_rules = rules[
    (rules["confidence"] >= confidence_threshold)
    &
    (rules["lift"] >= lift_threshold)
].copy()


filtered_rules = (
    filtered_rules
    .sort_values(
        [
            "lift",
            "confidence"
        ],
        ascending=False
    )
)


st.write(
    f"Strong rules found: "
    f"**{len(filtered_rules):,}**"
)


st.dataframe(
    filtered_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(50),
    use_container_width=True,
    hide_index=True
)

# CONFIDENCE VS LIFT

st.header(
    "📈 Association Rule Strength"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.scatter(
    rules["confidence"],
    rules["lift"],
    alpha=0.5
)

ax.set_title(
    "Confidence vs Lift"
)

ax.set_xlabel(
    "Confidence"
)

ax.set_ylabel(
    "Lift"
)

plt.tight_layout()

st.pyplot(fig)

# TOP ASSOCIATION RULES

st.header(
    "🏆 Strongest Product Associations"
)

top_rules = (
    rules
    .sort_values(
        [
            "lift",
            "confidence"
        ],
        ascending=False
    )
    .head(10)
)


for index, row in top_rules.iterrows():

    st.markdown(
        f"""
        **{index + 1}.**
        `{row['antecedents']}`
        → `{row['consequents']}`

        Support: **{row['support']:.2%}**  
        Confidence: **{row['confidence']:.2%}**  
        Lift: **{row['lift']:.2f}**
        """
    )

    st.divider()


# BUSINESS INSIGHTS

st.header(
    "💡 Business Insights"
)

most_popular_product = (
    product_frequency
    .sort_values(
        "purchase_count",
        ascending=False
    )
    .iloc[0]
)

peak_hour = (
    orders_by_hour.idxmax()
)

peak_day = (
    orders_by_day.idxmax()
)

best_rule = (
    rules
    .sort_values(
        "lift",
        ascending=False
    )
    .iloc[0]
)


st.success(
    f"""
    🏆 **Most Purchased Product:** 
    {most_popular_product['product_name']}

    ⏰ **Peak Shopping Hour:** 
    {peak_hour}:00

    📅 **Most Active Shopping Day:** 
    {day_names[peak_day]}

    🔗 **Strongest Association:**
    {best_rule['antecedents']}
    → {best_rule['consequents']}

    📈 **Strongest Rule Lift:**
    {best_rule['lift']:.2f}
    """
)

# Footer

st.divider()

st.caption(
    "SmartBasket AI | "
    "Instacart Grocery Analytics | "
    "Market Basket Analysis | "
    "Intelligent Recommendation System"
)