"""
Visualization Utility Functions
Machine Learning-based Late Delivery Risk Prediction
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# Create images folder automatically
# ---------------------------------------------------
os.makedirs("reports/images", exist_ok=True)

# Professional chart style
plt.style.use("ggplot")


# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------
def save_bar_chart(
    data,
    title,
    xlabel,
    ylabel,
    filename,
    color="steelblue",
    rotation=45
):
    plt.figure(figsize=(10, 6))

    data.plot(
        kind="bar",
        color=color,
        edgecolor="black"
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=rotation)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        f"reports/images/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------
# HISTOGRAM
# ---------------------------------------------------
def save_histogram(
    data,
    title,
    xlabel,
    ylabel,
    filename,
    bins=30,
    color="skyblue"
):
    plt.figure(figsize=(10, 6))

    plt.hist(
        data,
        bins=bins,
        color=color,
        edgecolor="black"
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        f"reports/images/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------
# COUNT PLOT
# ---------------------------------------------------
def save_countplot(
    df,
    column,
    title,
    filename,
    color="steelblue",
    rotation=45
):
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x=column,
        color=color,
        edgecolor="black",
        order=df[column].value_counts().index
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(rotation=rotation)

    plt.tight_layout()

    plt.savefig(
        f"reports/images/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------
# BOXPLOT
# ---------------------------------------------------
def save_boxplot(
    df,
    column,
    title,
    filename
):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        x=df[column],
        color="orange"
    )

    plt.title(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    plt.savefig(
        f"reports/images/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------
def save_heatmap(df, filename):
    plt.figure(figsize=(14, 10))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        annot=False
    )

    plt.title(
        "Correlation Heatmap",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"reports/images/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------
# GROUPED BAR CHART
# ---------------------------------------------------

def save_grouped_bar_chart(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))

    data.plot(
        kind="bar",
        figsize=(10, 6),
        edgecolor="black"
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=45)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        f"reports/images/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()