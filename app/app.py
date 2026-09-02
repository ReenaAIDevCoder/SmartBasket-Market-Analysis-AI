from pathlib import Path
import sys

import streamlit as st
import pandas as pd

# 1. PROJECT ROOT

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

# 2. PROJECT DIRECTORIES

SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"


# 3. ADD SRC TO PYTHON PATH

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )

# 4. IMPORT RECOMMENDATION ENGINE

from recommendation import (
    recommend_products,
    prepare_rules
)

# 5. PAGE CONFIGURATION

st.set_page_config(
    page_title="SmartBasket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 6. CUSTOM CSS

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# 7. LOAD PRODUCTS

@st.cache_data
def load_products():

    file_path = (
        RAW_DIR /
        "products.csv"
    )

    return pd.read_csv(
        file_path,
        usecols=[
            "product_id",
            "product_name",
            "aisle_id",
            "department_id"
        ],
        dtype={
            "product_id": "int32",
            "aisle_id": "int16",
            "department_id": "int16"
        }
    )

# 8. LOAD PRODUCT FREQUENCY

@st.cache_data
def load_product_frequency():

    file_path = (
        PROCESSED_DIR /
        "product_frequency.csv"
    )

    if not file_path.exists():

        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "purchase_count"
            ]
        )

    return pd.read_csv(
        file_path
    )

# 9. LOAD ASSOCIATION RULES

@st.cache_data
def load_rules():

    file_path = (
        PROCESSED_DIR /
        "association_rules.csv"
    )

    if not file_path.exists():

        return pd.DataFrame()

    rules = pd.read_csv(
        file_path
    )

    # Normalize column names

    rules.columns = [
        str(column).strip()
        for column in rules.columns
    ]


    # Handle singular/plural column names

    if (
        "antecedent" in rules.columns
        and "antecedents" not in rules.columns
    ):

        rules = rules.rename(
            columns={
                "antecedent":
                    "antecedents"
            }
        )


    if (
        "consequent" in rules.columns
        and "consequents" not in rules.columns
    ):

        rules = rules.rename(
            columns={
                "consequent":
                    "consequents"
            }
        )

    # Prepare rules

    if (
        "antecedents" in rules.columns
        and "consequents" in rules.columns
    ):

        try:

            rules = prepare_rules(
                rules
            )

        except Exception:

            pass


    return rules

# 10. LOAD DEPARTMENTS

@st.cache_data
def load_departments():

    file_path = (
        RAW_DIR /
        "departments.csv"
    )

    if not file_path.exists():

        return pd.DataFrame()

    return pd.read_csv(
        file_path
    )

# 11. MEMORY-SAFE ORDER SUMMARY

@st.cache_data
def load_order_summary():

    file_path = (
        RAW_DIR /
        "orders.csv"
    )

    total_orders = 0

    unique_users = set()

    orders_by_hour = {}

    orders_by_day = {}

    # Read orders in chunks

    for chunk in pd.read_csv(

        file_path,

        usecols=[
            "order_id",
            "user_id",
            "order_dow",
            "order_hour_of_day"
        ],

        dtype={
            "order_id": "int32",
            "user_id": "int32",
            "order_dow": "int8",
            "order_hour_of_day": "int8"
        },

        chunksize=500_000
    ):

        # Total orders

        total_orders += len(
            chunk
        )

        # Unique customers

        unique_users.update(
            chunk[
                "user_id"
            ].unique()
        )

        # Orders by hour

        hour_counts = (
            chunk[
                "order_hour_of_day"
            ]
            .value_counts()
        )

        for hour, count in (
            hour_counts.items()
        ):

            orders_by_hour[
                int(hour)
            ] = (
                orders_by_hour.get(
                    int(hour),
                    0
                )
                + int(count)
            )

        # Orders by day

        day_counts = (
            chunk[
                "order_dow"
            ]
            .value_counts()
        )

        for day, count in (
            day_counts.items()
        ):

            orders_by_day[
                int(day)
            ] = (
                orders_by_day.get(
                    int(day),
                    0
                )
                + int(count)
            )


        del chunk


    return {

        "total_orders":
            total_orders,

        "total_customers":
            len(unique_users),

        "orders_by_hour":
            pd.Series(
                orders_by_hour
            ).sort_index(),

        "orders_by_day":
            pd.Series(
                orders_by_day
            ).sort_index()
    }
    
# 12. LOAD ALL DATA

try:

    products = load_products()

    product_frequency = (
        load_product_frequency()
    )

    rules = load_rules()

    departments = load_departments()

    order_summary = (
        load_order_summary()
    )

except FileNotFoundError as error:

    st.error(
        "❌ Required project file was not found."
    )

    st.code(
        str(error)
    )

    st.stop()

except Exception as error:

    st.error(
        "❌ Error while loading SmartBasket data."
    )

    st.exception(
        error
    )

    st.stop()

# 13. SESSION STATE

if "basket" not in st.session_state:

    st.session_state.basket = []

# 14. SIDEBAR

st.sidebar.title(
    "🛒 SmartBasket AI"
)

st.sidebar.caption(
    "Intelligent Grocery Recommendation System"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🛒 Smart Basket",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

# 15. HOME PAGE

if page == "🏠 Home":

    st.markdown(
        """
        <div class="main-title">
            🛒 SmartBasket AI
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
            Intelligent Shopping & Market Basket
            Recommendation System
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()

    # KPI

    total_products = (
        products[
            "product_id"
        ]
        .nunique()
    )


    total_orders = (
        order_summary[
            "total_orders"
        ]
    )


    total_customers = (
        order_summary[
            "total_customers"
        ]
    )


    total_rules = len(
        rules
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "📦 Products",
            f"{total_products:,}"
        )


    with col2:

        st.metric(
            "🛒 Orders",
            f"{total_orders:,}"
        )


    with col3:

        st.metric(
            "👥 Customers",
            f"{total_customers:,}"
        )


    with col4:

        st.metric(
            "🔗 Rules",
            f"{total_rules:,}"
        )


    st.divider()

    # PROJECT OVERVIEW

    st.header(
        "🚀 What is SmartBasket AI?"
    )


    st.write(
        """
        SmartBasket AI is an intelligent grocery shopping
        recommendation system that analyzes historical
        purchasing patterns and recommends products that
        customers may want to purchase together.
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            ### 🛒 Smart Basket

            - Search grocery products
            - Build your shopping basket
            - Get AI-powered recommendations
            - View confidence and lift
            """
        )


    with col2:

        st.markdown(
            """
            ### 📊 Analytics

            - Product popularity
            - Association rules
            - Shopping behavior
            - Market basket insights
            """
        )


    st.divider()


    st.info(
        """
        💡 Go to **Smart Basket** from the sidebar
        and add products to receive recommendations.
        """
    )

# 16. SMART BASKET PAGE

elif page == "🛒 Smart Basket":

    st.title(
        "🛒 Smart Basket"
    )


    st.write(
        """
        Build your grocery basket and let SmartBasket AI
        recommend products based on historical shopping
        patterns.
        """
    )


    st.divider()

    # PRODUCT SEARCH

    st.subheader(
        "➕ Add Products"
    )


    search_text = st.text_input(
        "🔎 Search grocery product",
        placeholder=(
            "Try Sugar, Rice, Noodles, "
            "Milk, Bread..."
        )
    )

    # SEARCH PRODUCTS

    if search_text.strip():

        matching_products = (
            products[
                products[
                    "product_name"
                ]
                .str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]
            .sort_values(
                "product_name"
            )
            .head(100)
        )

    else:

        # Show popular products when
        # search box is empty

        if not product_frequency.empty:

            matching_products = (
                product_frequency
                .head(100)
                [
                    [
                        "product_id",
                        "product_name"
                    ]
                ]
            )

        else:

            matching_products = (
                products
                .head(100)
                [
                    [
                        "product_id",
                        "product_name"
                    ]
                ]
            )

    # PRODUCT SELECTBOX

    if matching_products.empty:

        st.warning(
            "❌ No product found. "
            "Try another search."
        )

    else:

        selected_product = st.selectbox(

            "Select a grocery product",

            matching_products[
                "product_name"
            ]
            .drop_duplicates()
            .tolist()
        )

        # ADD PRODUCT

        if st.button(
            "➕ Add to Basket",
            use_container_width=True
        ):

            if (
                selected_product
                not in st.session_state.basket
            ):

                st.session_state.basket.append(
                    selected_product
                )

                st.success(
                    f"✅ Added: {selected_product}"
                )

                st.rerun()

            else:

                st.warning(
                    "⚠️ This product is already "
                    "in your basket."
                )


    st.divider()

    # CURRENT BASKET

    st.subheader(
        "🛍️ Current Basket"
    )


    if st.session_state.basket:

        for index, product in enumerate(
            st.session_state.basket
        ):

            col1, col2 = st.columns(
                [8, 1]
            )


            with col1:

                st.write(
                    f"**{index + 1}.** "
                    f"{product}"
                )


            with col2:

                if st.button(
                    "❌",
                    key=f"remove_{index}"
                ):

                    st.session_state.basket.pop(
                        index
                    )

                    st.rerun()


        if st.button(
            "🗑️ Clear Basket",
            use_container_width=True
        ):

            st.session_state.basket = []

            st.rerun()


    else:

        st.info(
            "🛒 Your basket is empty. "
            "Add products to start."
        )


    st.divider()

    # RECOMMENDATION ENGINE

    st.subheader(
        "🤖 Smart Recommendations"
    )


    if st.session_state.basket:

        try:

            recommendations = (
                recommend_products(

                    basket=
                        st.session_state.basket,

                    rules=rules,

                    top_n=5,

                    min_confidence=0.05,

                    min_lift=1.05
                )
            )


        except Exception as error:

            st.error(
                "❌ Recommendation engine error."
            )

            st.exception(
                error
            )

            recommendations = (
                pd.DataFrame()
            )

        # DISPLAY RECOMMENDATIONS

        if (
            recommendations is None
            or len(recommendations) == 0
        ):

            st.info(
                """
                💡 No strong recommendations were
                found for your current basket.

                Try adding another product.
                """
            )


        else:

            st.success(
                "✨ Products you may also want to buy:"
            )


            for _, row in (
                recommendations.iterrows()
            ):

                # Product
                

                product_name = row.get(
                    "Product",
                    row.get(
                        "product",
                        "Unknown Product"
                    )
                )


                # Confidence

                confidence = row.get(
                    "Confidence",
                    row.get(
                        "confidence",
                        0
                    )
                )


                # Lift
                

                lift = row.get(
                    "Lift",
                    row.get(
                        "lift",
                        0
                    )
                )


                # Support
                

                support = row.get(
                    "Support",
                    row.get(
                        "support",
                        0
                    )
                )


                col1, col2, col3 = (
                    st.columns(
                        [4, 2, 2]
                    )
                )


                with col1:

                    st.markdown(
                        f"### 🛒 {product_name}"
                    )


                with col2:

                    st.metric(
                        "Confidence",
                        f"{float(confidence):.1%}"
                    )


                with col3:

                    st.metric(
                        "Lift",
                        f"{float(lift):.2f}"
                    )


                st.caption(
                    f"Support: "
                    f"{float(support):.2%}"
                )


                st.divider()


    else:

        st.info(
            "Add products to your basket "
            "to receive AI recommendations."
        )


# 17. ANALYTICS PAGE

elif page == "📊 Analytics":

    st.title(
        "📊 SmartBasket Analytics"
    )


    st.write(
        """
        Explore grocery product popularity,
        shopping behavior and association-rule insights.
        """
    )


    st.divider()

    # KPI

    total_products = (
        products[
            "product_id"
        ]
        .nunique()
    )


    total_orders = (
        order_summary[
            "total_orders"
        ]
    )


    total_customers = (
        order_summary[
            "total_customers"
        ]
    )


    total_rules = len(
        rules
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "📦 Products",
            f"{total_products:,}"
        )


    with col2:

        st.metric(
            "🛒 Orders",
            f"{total_orders:,}"
        )


    with col3:

        st.metric(
            "👥 Customers",
            f"{total_customers:,}"
        )


    with col4:

        st.metric(
            "🔗 Rules",
            f"{total_rules:,}"
        )


    st.divider()

    # TOP PRODUCTS
    

    st.subheader(
        "🏆 Top Purchased Products"
    )


    if product_frequency.empty:

        st.warning(
            "Product frequency data is unavailable."
        )

    else:

        top_n = st.slider(
            "Number of products",
            min_value=5,
            max_value=min(
                20,
                len(product_frequency)
            ),
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


        st.dataframe(

            top_products[
                [
                    "product_id",
                    "product_name",
                    "purchase_count"
                ]
            ],

            use_container_width=True,

            hide_index=True
        )


    st.divider()


    # ORDERS BY HOUR

    st.subheader(
        "⏰ Shopping Time Analysis"
    )


    orders_by_hour = (
        order_summary[
            "orders_by_hour"
        ]
    )


    st.line_chart(
        orders_by_hour
    )


    st.caption(
        "Number of orders by hour of day."
    )


    st.divider()

    # ORDERS BY DAY


    st.subheader(
        "📅 Orders by Day of Week"
    )


    orders_by_day = (
        order_summary[
            "orders_by_day"
        ]
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


    day_data = pd.DataFrame(
        {
            "Day": [
                day_names.get(
                    int(day),
                    str(day)
                )

                for day
                in orders_by_day.index
            ],

            "Orders":
                orders_by_day.values
        }
    )


    st.dataframe(
        day_data,
        use_container_width=True,
        hide_index=True
    )


    st.divider()

    # ASSOCIATION RULE EXPLORER
    

    st.subheader(
        "🔗 Association Rule Explorer"
    )


    if rules.empty:

        st.warning(
            "No association rules available."
        )

    else:

        col1, col2 = st.columns(2)


        with col1:

            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.05
            )


        with col2:

            min_lift = st.slider(
                "Minimum Lift",
                min_value=0.0,
                max_value=5.0,
                value=1.05,
                step=0.10
            )

        # FILTER RULES

        filtered_rules = rules.copy()


        if "confidence" in (
            filtered_rules.columns
        ):

            filtered_rules = (
                filtered_rules[
                    filtered_rules[
                        "confidence"
                    ]
                    >= min_confidence
                ]
            )


        if "lift" in (
            filtered_rules.columns
        ):

            filtered_rules = (
                filtered_rules[
                    filtered_rules[
                        "lift"
                    ]
                    >= min_lift
                ]
            )


        # DISPLAY COLUMN NORMALIZATION


        antecedent_column = None

        consequent_column = None


        if "antecedents" in (
            filtered_rules.columns
        ):

            antecedent_column = (
                "antecedents"
            )

        elif "antecedent" in (
            filtered_rules.columns
        ):

            antecedent_column = (
                "antecedent"
            )


        if "consequents" in (
            filtered_rules.columns
        ):

            consequent_column = (
                "consequents"
            )

        elif "consequent" in (
            filtered_rules.columns
        ):

            consequent_column = (
                "consequent"
            )


        # Convert rule values to readable strings


        def rule_to_string(value):

            if isinstance(
                value,
                (list, tuple, set)
            ):

                return ", ".join(
                    map(
                        str,
                        value
                    )
                )

            return str(
                value
            )


        if antecedent_column:

            filtered_rules[
                "Antecedent"
            ] = (
                filtered_rules[
                    antecedent_column
                ]
                .apply(
                    rule_to_string
                )
            )


        if consequent_column:

            filtered_rules[
                "Consequent"
            ] = (
                filtered_rules[
                    consequent_column
                ]
                .apply(
                    rule_to_string
                )
            )


        # Sort rules

        sort_columns = []

        if "lift" in filtered_rules.columns:

            sort_columns.append(
                "lift"
            )

        if "confidence" in filtered_rules.columns:

            sort_columns.append(
                "confidence"
            )


        if sort_columns:

            filtered_rules = (
                filtered_rules
                .sort_values(
                    sort_columns,
                    ascending=False
                )
            )


        st.write(
            f"Strong rules found: "
            f"**{len(filtered_rules):,}**"
        )


        # Display rules

        display_columns = []


        for column in [
            "Antecedent",
            "Consequent",
            "support",
            "confidence",
            "lift"
        ]:

            if column in (
                filtered_rules.columns
            ):

                display_columns.append(
                    column
                )


        if display_columns:

            st.dataframe(

                filtered_rules[
                    display_columns
                ]
                .head(50),

                use_container_width=True,

                hide_index=True
            )

        else:

            st.warning(
                "Association-rule columns "
                "could not be detected."
            )


# 18. ABOUT PAGE

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About SmartBasket AI"
    )


    st.markdown(
        """
        ## 🛒 SmartBasket AI

        **Intelligent Shopping & Market Basket
        Recommendation System**

        ---

        ### 📊 Dataset

        Instacart Online Grocery Basket Dataset.

        ---

        ### 🧠 Core Methodology

        - Data Preprocessing
        - Exploratory Data Analysis
        - Product Frequency Analysis
        - Market Basket Analysis
        - FP-Growth
        - Association Rule Mining
        - Recommendation Engine

        ---

        ### 🛠️ Technologies

        - Python
        - Pandas
        - NumPy
        - SciPy
        - MLxtend
        - Streamlit

        ---

        ### 📈 Recommendation Metrics

        **Support**

        Measures how frequently an itemset occurs
        in the transaction dataset.

        **Confidence**

        Measures how often the consequent occurs
        when the antecedent is present.

        **Lift**

        Measures the strength of association between
        products compared with random occurrence.

        ---

        ### 🔄 Project Pipeline

        Raw Dataset  
        ↓  
        Data Preprocessing  
        ↓  
        Exploratory Data Analysis  
        ↓  
        Product Frequency Analysis  
        ↓  
        FP-Growth  
        ↓  
        Association Rules  
        ↓  
        Recommendation Engine  
        ↓  
        SmartBasket AI Application
        """
    )


    st.divider()


    st.success(
        "🚀 SmartBasket AI is ready!"
    )


# 19. SIDEBAR FOOTER

st.sidebar.divider()

st.sidebar.caption(
    "SmartBasket AI"
)

st.sidebar.caption(
    "Market Basket Analysis + "
    "Intelligent Recommendations"
)