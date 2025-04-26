import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

FONTSIZE = 18

import scienceplots

plt.style.use(["science", "ieee"])


def extract_keywords(reason_str):
    """Extract keywords from exclusion reasons, ignoring location modifiers."""
    if pd.isna(reason_str):
        return []

    keywords = []
    parts = str(reason_str).split(";")
    for part in parts:
        part = part.strip()
        # Remove location context (in abstract/in title)
        if "(" in part:
            keyword = part.split("(")[0].strip()
        else:
            keyword = part
        keywords.append(keyword)
    return keywords


def create_visualization(excluded_df, included_df, manual_exclusions={}):
    """
    Create 1x3 subplots for citation data analysis.

    Parameters:
    -----------
    excluded_df : DataFrame
        DataFrame containing excluded citations
    included_df : DataFrame
        DataFrame containing included citations
    manual_exclusions : dict
        Dictionary of additional exclusion counts to add {filter_type: count}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Plot 1: Excluded papers by filter type
    filter_counts = excluded_df["exclusion_filter"].value_counts().to_dict()

    # Add manual exclusions to the counts
    for filter_type, count in manual_exclusions.items():
        if filter_type in filter_counts:
            filter_counts[filter_type] += count
        else:
            filter_counts[filter_type] = count

    filter_types = list(filter_counts.keys())
    filter_values = list(filter_counts.values())

    filter_types = [f.replace("_", " ").title() for f in filter_types]
    axes[0].bar(filter_types, filter_values, color="salmon")
    axes[0].set_xlabel("Filter Type", fontsize=FONTSIZE)
    axes[0].set_ylabel("Number of Papers", fontsize=FONTSIZE)
    axes[0].tick_params(axis="x", rotation=90)

    # Plot 2: Exclusion keywords for entries with exclusion_filter='keyword'
    keyword_filtered = excluded_df[excluded_df["exclusion_filter"] == "keyword"]

    # Extract all keywords, ignoring whether they're in title or abstract
    all_keywords = []
    for reason in keyword_filtered["exclusion_reason"]:
        keywords = extract_keywords(reason)
        all_keywords.extend(keywords)

    keyword_counts = pd.Series(all_keywords).value_counts()

    # Limit to top keywords if there are too many

    axes[1].bar(
        [k.title() for k in keyword_counts.index],
        keyword_counts.values,
        color="lightblue",
    )
    axes[1].set_xlabel("Keyword", fontsize=FONTSIZE)
    # axes[1].set_ylabel("Number of Papers", fontsize=12)
    axes[1].tick_params(axis="x", rotation=90)

    # Plot 3: 2x2 heatmap (included/excluded vs open-access/not open-access)
    heatmap_data = pd.DataFrame(
        {
            "Open Access": [
                sum(included_df["is_open_access"] == True),
                sum(excluded_df["is_open_access"] == True),
            ],
            "Not Open Access": [
                sum(included_df["is_open_access"] == False),
                sum(excluded_df["is_open_access"] == False),
            ],
        },
        index=["Included", "Excluded"],
    )

    # Plot the heatmap with percentages
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        ax=axes[2],
        annot_kws={"size": FONTSIZE},
    )

    for ax in axes:
        ax.tick_params(axis="both", labelsize=FONTSIZE - 2)  # Setting tick label size

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False, which="both", direction="in")
        ax.grid(False)
        if ax == axes[2]:
            ax.tick_params(bottom=False, left=False, which="both", direction="in")

    labels = ["a.", "b.", "c."]
    for ax, lbl in zip((axes[0], axes[1], axes[2]), labels):
        ax.text(
            -0.04,
            1.016,  # x,y in Axes fraction units: just inside top‐left
            lbl,  # the label text
            transform=ax.transAxes,  # interpret x,y in [0,1]×[0,1] of the Axes
            fontsize=14,  # tweak as you like
            fontweight="bold",
            va="top",  # vertical alignment at the top of the text
            ha="left",  # horizontal alignment at the left of the text
        )
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    excluded_df = pd.read_csv("data/csvs/info_citations_excluded.csv")
    included_df = pd.read_csv("data/csvs/info_citations_included.csv")

    # Example manual exclusions (replace with your actual values)
    manual_exclusions = {"Manual*": 5, "Duplicate": 119}

    fig = create_visualization(excluded_df, included_df, manual_exclusions)
    plt.savefig("citation_analysis.pdf", format="pdf", dpi=300, bbox_inches="tight")
