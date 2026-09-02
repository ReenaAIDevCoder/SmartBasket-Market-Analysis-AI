import streamlit as st
import pandas as pd

# Recommendation Engine

def recommend_products(
    basket,
    rules,
    top_n=5,
    min_confidence=0.30,
    min_lift=1.20
):
    """
    Generate product recommendations using
    association rules.
    """

    # Normalize basket
    basket_normalized = {
        item.strip().lower()
        for item in basket
    }

    recommendations = []

    # Process association rules
    for _, rule in rules.iterrows():

        antecedents = {
            item.strip().lower()
            for item in str(
                rule["antecedents"]
            ).split(",")
            if item.strip()
        }

        consequents = [
            item.strip()
            for item in str(
                rule["consequents"]
            ).split(",")
            if item.strip()
        ]

        # Check whether the current basket
        # contains the rule antecedents
        if antecedents.issubset(
            basket_normalized
        ):

            # Quality filtering
            if rule["confidence"] < min_confidence:
                continue

            if rule["lift"] < min_lift:
                continue

            # Generate recommendations
            for product in consequents:

                # Don't recommend products
                # already in the basket
                if product.lower() not in basket_normalized:

                    recommendations.append({

                        "Product": product,

                        "Confidence":
                            float(rule["confidence"]),

                        "Lift":
                            float(rule["lift"]),

                        "Support":
                            float(rule["support"])

                    })

    # No recommendations
    if not recommendations:

        return pd.DataFrame(
            columns=[
                "Product",
                "Confidence",
                "Lift",
                "Support"
            ]
        )

    # Convert to DataFrame
    recommendations_df = pd.DataFrame(
        recommendations
    )

    # Multiple rules may recommend
    # the same product
    recommendations_df = (
        recommendations_df
        .groupby(
            "Product",
            as_index=False
        )
        .agg({
            "Confidence": "max",
            "Lift": "max",
            "Support": "max"
        })
    )

    # Rank recommendations
    recommendations_df = (
        recommendations_df
        .sort_values(
            [
                "Lift",
                "Confidence",
                "Support"
            ],
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return recommendations_df

# Recommendation Cards

def display_recommendation_cards(
    recommendations
):
    """
    Display recommendations as
    user-friendly Streamlit cards.
    """

    if recommendations.empty:

        st.info(
            "💡 No strong recommendations found "
            "for your current basket."
        )

        return

    st.success(
        "✨ SmartBasket AI found products "
        "you may also want to buy!"
    )

    for index, row in recommendations.iterrows():

        st.markdown(
            f"""
            ### 🛒 {index + 1}. {row['Product']}
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Confidence",
                f"{row['Confidence']:.1%}"
            )

        with col2:

            st.metric(
                "Lift",
                f"{row['Lift']:.2f}"
            )

        with col3:

            st.metric(
                "Support",
                f"{row['Support']:.2%}"
            )

        st.divider()

# Recommendation Section
def render_recommendation_section(
    basket,
    rules,
    top_n=5
):
    """
    Render complete SmartBasket recommendation
    section in the Streamlit application.
    """

    st.header(
        "🤖 SmartBasket AI Recommendations"
    )

    # Empty basket
    if not basket:

        st.info(
            "🛒 Add products to your basket "
            "to receive AI-powered recommendations."
        )

        return pd.DataFrame()

    # Display current basket
    st.subheader(
        "🛍️ Your Current Basket"
    )

    basket_columns = st.columns(
        min(len(basket), 4)
    )

    for index, item in enumerate(basket):

        with basket_columns[
            index % len(basket_columns)
        ]:

            st.write(
                f"✓ **{item}**"
            )

    st.divider()

    # Generate recommendations
    recommendations = recommend_products(
        basket=basket,
        rules=rules,
        top_n=top_n,
        min_confidence=0.30,
        min_lift=1.20
    )

    # Display recommendations
    display_recommendation_cards(
        recommendations
    )

    return recommendations

# Recommendation Summary

def display_recommendation_summary(
    recommendations
):
    """
    Display a compact recommendation summary.
    """

    if recommendations.empty:

        return

    st.subheader(
        "📊 Recommendation Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Recommendations",
            len(recommendations)
        )

    with col2:

        st.metric(
            "Best Lift",
            f"{recommendations['Lift'].max():.2f}"
        )

    with col3:

        st.metric(
            "Best Confidence",
            f"{recommendations['Confidence'].max():.1%}"
        )

# Test Mode

if __name__ == "__main__":

    print(
        "recommendation_ui.py loaded successfully."
    )