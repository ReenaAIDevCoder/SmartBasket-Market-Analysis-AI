# SmartBasket AI
# Recommendation Engine

import pandas as pd

# PREPARE ASSOCIATION RULES

def prepare_rules(rules):

    rules = rules.copy()

    # Support both old and new column names


    if "antecedents" in rules.columns:

        rules["antecedent"] = (
            rules["antecedents"]
            .astype(str)
        )

    elif "antecedent" in rules.columns:

        rules["antecedent"] = (
            rules["antecedent"]
            .astype(str)
        )

    else:

        raise ValueError(
            "Association rules file does not contain "
            "'antecedent' or 'antecedents' column."
        )


    if "consequents" in rules.columns:

        rules["consequent"] = (
            rules["consequents"]
            .astype(str)
        )

    elif "consequent" in rules.columns:

        rules["consequent"] = (
            rules["consequent"]
            .astype(str)
        )

    else:

        raise ValueError(
            "Association rules file does not contain "
            "'consequent' or 'consequents' column."
        )


    # Convert metrics to numeric


    for column in [
        "support",
        "confidence",
        "lift"
    ]:

        if column in rules.columns:

            rules[column] = pd.to_numeric(
                rules[column],
                errors="coerce"
            )

    # Remove invalid rows


    rules = rules.dropna(
        subset=[
            "antecedent",
            "consequent"
        ]
    )

    # Clean product names


    rules["antecedent"] = (
        rules["antecedent"]
        .astype(str)
        .str.strip()
    )

    rules["consequent"] = (
        rules["consequent"]
        .astype(str)
        .str.strip()
    )


    return rules

# RECOMMEND PRODUCTS

def recommend_products(
    basket_products,
    rules,
    top_n=10,
    min_confidence=0.05,
    min_lift=1.10
):

    # Empty basket

    if not basket_products:

        return []

    # Prepare basket


    basket = [
        str(product).strip().lower()
        for product in basket_products
    ]

    # Prepare rules

    rules = prepare_rules(
        rules
    )

    # Filter strong rules

    if "confidence" in rules.columns:

        rules = rules[
            rules["confidence"]
            >= min_confidence
        ]


    if "lift" in rules.columns:

        rules = rules[
            rules["lift"]
            >= min_lift
        ]

    # Find recommendations

    recommendations = []


    for _, rule in rules.iterrows():

        antecedent = (
            str(rule["antecedent"])
            .strip()
            .lower()
        )

        consequent = (
            str(rule["consequent"])
            .strip()
        )

        # Match current basket
    

        if antecedent in basket:

            # Don't recommend products
            # already in basket

            if consequent.lower() in basket:

                continue


            confidence = (
                float(rule["confidence"])
                if "confidence" in rule
                else 0.0
            )

            lift = (
                float(rule["lift"])
                if "lift" in rule
                else 0.0
            )

            support = (
                float(rule["support"])
                if "support" in rule
                else 0.0
            )


            recommendations.append(
                {
                    "product": consequent,
                    "confidence": confidence,
                    "lift": lift,
                    "support": support
                }
            )

    # Remove duplicate products


    unique_recommendations = {}

    for item in recommendations:

        product = item["product"]

        if (
            product not in
            unique_recommendations
        ):

            unique_recommendations[
                product
            ] = item

        else:

            old = unique_recommendations[
                product
            ]

            if (
                item["lift"],
                item["confidence"]
            ) > (
                old["lift"],
                old["confidence"]
            ):

                unique_recommendations[
                    product
                ] = item

    # Sort recommendations
    

    recommendations = sorted(
        unique_recommendations.values(),
        key=lambda x: (
            x["lift"],
            x["confidence"],
            x["support"]
        ),
        reverse=True
    )


    return recommendations[
        :top_n
    ]