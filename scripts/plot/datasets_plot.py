#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

import scienceplots

plt.style.use(["science", "ieee", "grid"])

FONTSIZE = 18
plt.rcParams.update({"font.size": FONTSIZE})


def load_data(excluded_file, included_file):
    """Load and combine data from excluded and included CSVs."""
    # Load data
    excluded_df = pd.read_csv(excluded_file)
    included_df = pd.read_csv(included_file)

    # Add inclusion status column
    excluded_df["status"] = "excluded"
    included_df["status"] = "included"

    # Combine datasets
    combined_df = pd.concat([excluded_df, included_df])

    return combined_df


def extract_datasets(df):
    """Extract dataset mentions from the cited_dataset column."""
    all_datasets = []

    for _, row in df.iterrows():
        if pd.notna(row["cited_dataset"]):
            # Strip quotes if present and split the datasets
            cited_dataset = str(row["cited_dataset"])
            if cited_dataset.startswith('"') and cited_dataset.endswith('"'):
                cited_dataset = cited_dataset[1:-1]

            datasets = [ds.strip() for ds in cited_dataset.split(",")]

            # Try to convert citation_count to float
            try:
                citation_count = (
                    float(row.get("citation_count", 0))
                    if not pd.isna(row.get("citation_count", 0))
                    else 0
                )
            except (ValueError, TypeError):
                citation_count = 0

            # Try to convert year to float
            try:
                year = (
                    float(row.get("year", None))
                    if not pd.isna(row.get("year", None))
                    else None
                )
            except (ValueError, TypeError):
                year = None

            # Add each dataset to the list
            for dataset in datasets:
                all_datasets.append(
                    {
                        "dataset": dataset,
                        "paper_id": row["paper_id"],
                        "status": row["status"],
                        "citation_count": citation_count,
                        "year": year,
                    }
                )

    # Convert to DataFrame
    datasets_df = pd.DataFrame(all_datasets)

    return datasets_df


def return_dataset_counts(datasets_df):
    """
    Compute and return a list of dataset names sorted by their total paper-count (included+excluded).
    """
    # count how many times each dataset appears, split by status
    counts = datasets_df.groupby(["dataset", "status"]).size().unstack(fill_value=0)
    # ensure both columns exist
    for col in ("included", "excluded"):
        if col not in counts:
            counts[col] = 0

    # add a 'total' column and sort by it
    counts["total"] = counts["included"] + counts["excluded"]
    counts = counts.sort_values("total", ascending=True)

    # return the sorted dataset names
    return counts.index.tolist()


def plot_dataset_counts(
    ax, datasets_df, ordered_datasets, color_included, color_excluded, y_positions
):
    """
    Create horizontal bar plot of dataset counts at the given y_positions.
    """
    # (1) compute counts
    dataset_counts = (
        datasets_df.groupby(["dataset", "status"]).size().unstack(fill_value=0)
    )
    # ensure both columns exist
    for col in ("included", "excluded"):
        if col not in dataset_counts:
            dataset_counts[col] = 0

    # (2) pull out our ordered lists
    inc = dataset_counts.loc[ordered_datasets, "included"]
    exc = dataset_counts.loc[ordered_datasets, "excluded"]

    # (3) draw bars at the y_positions
    ax.barh(y_positions, inc, color=color_included, height=1.2, label="Included")
    ax.barh(
        y_positions, exc, left=inc, color=color_excluded, height=1.2, label="Excluded"
    )

    # (4) fix ticks & grid
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_datasets)
    ax.set_ylim(y_positions[0] - 0.6, y_positions[-1] + 0.6)
    ax.set_xlabel("Number of Papers")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_title("Dataset Counts", fontsize=FONTSIZE, fontweight="bold")


def plot_citation_boxplot(
    ax, datasets_df, ordered_datasets, color_included, color_excluded, y_positions
):
    """
    Create horizontal boxplots of citations, offset at each y_position.
    """
    for i, dataset in enumerate(ordered_datasets):
        y = y_positions[i]
        # included
        data_inc = datasets_df.query("dataset==@dataset & status=='included'")[
            "citation_count"
        ]
        # excluded
        data_exc = datasets_df.query("dataset==@dataset & status=='excluded'")[
            "citation_count"
        ]

        if len(data_inc):
            bp = ax.boxplot(
                data_inc,
                positions=[y - 0.9],
                vert=False,
                widths=1.5,
                patch_artist=True,
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=color_included, alpha=0.5)
            for el in ["whiskers", "caps", "medians", "fliers"]:
                plt.setp(bp[el], color=color_included)

        if len(data_exc):
            bp = ax.boxplot(
                data_exc,
                positions=[y + 0.9],
                vert=False,
                widths=1.5,
                patch_artist=True,
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=color_excluded, alpha=0.5)
            for el in ["whiskers", "caps", "medians", "fliers"]:
                plt.setp(bp[el], color=color_excluded)

    # same ticks & limits
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_datasets)
    ax.set_ylim(y_positions[0] - 0.6, y_positions[-1] + 0.6)
    ax.set_xlabel("Number of Citations")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_title("Citation Counts", fontsize=FONTSIZE, fontweight="bold")


def plot_year_bubbles(
    ax, datasets_df, ordered_datasets, color_included, color_excluded, y_positions
):
    """
    Create a bubble/pie plot at each (year, y_position).
    """
    # group once
    grouped = (
        datasets_df.groupby(["year", "dataset", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    max_count = 0
    for _, row in grouped.iterrows():
        year, dataset = row["year"], row["dataset"]
        included = row.get("included", 0)
        excluded = row.get("excluded", 0)
        total = included + excluded
        if total == 0 or np.isnan(year):
            continue

        # lookup our fixed y
        y = y_positions[ordered_datasets.index(dataset)]
        max_count = max(max_count, total)

        r = 0.15 * np.sqrt(total)  # radius
        ax.pie(
            [included, excluded],
            colors=[color_included, color_excluded],
            center=(year, y),
            radius=r,
            wedgeprops=dict(width=r, edgecolor="white"),
        )

    # manual ticks & limits
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_datasets)
    ax.set_ylim(y_positions[0] - 0.6, y_positions[-1] + 0.6)
    ax.set_xlim(datasets_df["year"].min() - 1, datasets_df["year"].max() + 1)
    ax.set_xlabel("Year")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_title("Papers per Year", fontsize=FONTSIZE, fontweight="bold")

    # manual ticks & limits
    min_year = int(datasets_df["year"].min())
    max_year = int(datasets_df["year"].max())
    ax.set_xlim(min_year - 1, max_year + 1)
    from matplotlib.ticker import MultipleLocator

    # tick every 10 years
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xticklabels(ax.get_xticks().astype(int))  # if you want integer labels
    ax.set_xlabel("Year")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    # bubble-size legend
    sizes = sorted({5, 10, 20, max(30, int(max_count / 2)), max_count})
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=(0.15 * np.sqrt(c) * 10),
            label=str(c),
        )
        for c in sizes
    ]
    ax.legend(
        handles=handles,
        title="Paper Count",
        loc="center right",
        bbox_to_anchor=(1.33, 0.5),
        fontsize=FONTSIZE - 2,
        title_fontsize=FONTSIZE - 2,
    )
    return max_count


def main():
    """Main function to generate and save the visualizations."""
    # Set the style for the plots
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Define file paths
    excluded_file = "data/csvs/info_citations_excluded.csv"
    included_file = "data/csvs/info_citations_included.csv"
    output_dir = "data/plots"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define colors
    color_included = "#2ca02c"  # Green
    color_excluded = "#d62728"  # Red

    # Load and process data
    combined_df = load_data(excluded_file, included_file)
    datasets_df = extract_datasets(combined_df)

    # Create figure and subplots with shared y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    ax1, ax2, ax3 = axes

    # Plot 1: Dataset Counts
    ordered_datasets = return_dataset_counts(datasets_df)

    spacing = 4
    y_positions = np.arange(len(ordered_datasets)) * spacing
    plot_dataset_counts(
        ax1,
        datasets_df,
        ordered_datasets,
        color_included,
        color_excluded,
        y_positions,
    )
    # Plot 2: Citation Boxplot
    plot_citation_boxplot(
        ax2,
        datasets_df,
        ordered_datasets,
        color_included,
        color_excluded,
        y_positions,
    )

    # Plot 3: Year Bubbles
    plot_year_bubbles(
        ax3, datasets_df, ordered_datasets, color_included, color_excluded, y_positions
    )
    # give each of the first two axes 10% extra y-padding
    for ax in (ax1, ax2, ax3):
        pad = spacing * 0.5  # e.g. if spacing==4, pad==2
        ax.set_ylim(y_positions[0] - pad, y_positions[-1] + pad)
    # Create common legend for included/excluded
    common_legend_handles = [
        mpatches.Patch(color=color_included, label="Included"),
        mpatches.Patch(color=color_excluded, label="Excluded"),
    ]

    # Add the common legend at the bottom of the figure
    fig.legend(
        handles=common_legend_handles,
        loc="lower center",
        ncol=2,  # Two columns for the legend
        bbox_to_anchor=(0.5, 0.02),  # Position at the bottom center
        frameon=True,
    )

    # right after you create your subplots (or after plotting everything,
    # but before saving), do:

    labels = ["a.", "b.", "c."]
    for ax, lbl in zip((ax1, ax2, ax3), labels):
        ax.text(
            -0.05,
            1.016,  # x,y in Axes fraction units: just inside top‐left
            lbl,  # the label text
            transform=ax.transAxes,  # interpret x,y in [0,1]×[0,1] of the Axes
            fontsize=FONTSIZE,  # tweak as you like
            fontweight="bold",
            va="top",  # vertical alignment at the top of the text
            ha="left",  # horizontal alignment at the left of the text
        )

        ax.tick_params(axis="both", labelsize=FONTSIZE - 2)  # Setting tick label size

    plt.subplots_adjust(top=1.0, bottom=0.2)  # Adjust these values to control spacing

    # Adjust layout to make room for the common legend at the bottom
    plt.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )  # Leave space for the suptitle and bottom legend
    # Save figure
    output_file = os.path.join(output_dir, "dataset_analysis.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)  # Close the figure to free up memory
    print(f"Figure saved to {output_file}")


if __name__ == "__main__":
    main()
