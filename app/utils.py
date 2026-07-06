"""
Utility Functions
"""


def risk_category(probability):

    if probability < 0.40:
        return "🟢 Low Risk"

    elif probability < 0.70:
        return "🟡 Medium Risk"

    else:
        return "🔴 High Risk"


def recommendation(probability):

    if probability < 0.40:
        return "Shipment is likely to arrive on time."

    elif probability < 0.70:
        return (
            "Monitor shipment and keep customer informed."
        )

    else:
        return (
            "Prioritize shipment, notify customer, "
            "and consider faster shipping."
        )