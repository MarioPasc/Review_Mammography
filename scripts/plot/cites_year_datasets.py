"""
Enhanced visualization of dataset citation data with custom styling and annotations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator


# Load the dataset citation data
def load_dataset_data(json_path):
    """Load and preprocess the dataset citation data."""
    with open(json_path, "r") as file:
        data = json.load(file)

    # Remove datasets with duplicate paper_id to avoid duplication
    seen_paper_ids = set()
    unique_datasets = []

    for item in data:
        if "paper_id" in item and item["paper_id"] not in seen_paper_ids:
            seen_paper_ids.add(item["paper_id"])
            unique_datasets.append(item)

    # Filter out datasets with 0 citations across all years
    filtered_data = []
    for dataset in unique_datasets:
        if dataset.get("total_citations", 0) > 0:
            filtered_data.append(dataset)

    return filtered_data


def create_enhanced_visualization(data, output_path, vertical_heights=None):
    """Create an enhanced visualization with custom styling and annotations."""
    # Set Arial Bold as font
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    # Default vertical heights dictionary if none provided
    if vertical_heights is None:
        vertical_heights = {}

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate the overall year range for all datasets
    all_years = []
    for dataset in data:
        years = [int(y) for y in dataset.get("citations_by_year", {}).keys()]
        if years:
            all_years.extend(years)
        if "publication_year" in dataset and dataset["publication_year"]:
            all_years.append(dataset["publication_year"])

    year_range = range(
        min(all_years) if all_years else 1990, max(all_years) + 1 if all_years else 2025
    )

    # Color palette for different datasets
    colors = plt.cm.tab10.colors + plt.cm.Set2.colors

    # Calculate y-axis limits
    max_citations = max(
        [
            max(
                [int(count) for count in d.get("citations_by_year", {}).values()],
                default=0,
            )
            for d in data
        ],
        default=100,
    )
    y_limit = int(max_citations * 1.3)  # 30% extra space for annotations

    # Track horizontal positions for annotations to avoid overlap
    year_annotations = {}

    # Plot each dataset's citations by year
    for i, dataset in enumerate(data):
        dataset_name = dataset.get("dataset", "Unknown")
        pub_year = dataset.get("publication_year")
        if not pub_year:
            continue

        first_author = dataset.get("first_author", "Unknown").split(" ")[
            -1
        ]  # Get last name
        citations = dataset.get("citations_by_year", {})

        # Sort years and get citation counts
        years = sorted([int(y) for y in citations.keys()])
        if not years:
            continue

        counts = [citations.get(str(y), 0) for y in years]

        # Plot citation trend
        color = colors[i % len(colors)]
        ax.plot(
            years,
            counts,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=6,
            label=dataset_name,
            color=color,
        )

        # Add vertical marker at publication year with customizable height
        default_height = y_limit * 0.6  # Default 60% of y-limit

        # Get custom height for this publication year if specified
        custom_height_factor = vertical_heights.get(
            pub_year, 0.6
        )  # Default to 0.6 if not specified
        vertical_height = y_limit * custom_height_factor

        # Create vertical line with discontinuous style
        dash_pattern = [4, 2]  # 4 points on, 2 points off
        ax.plot(
            [pub_year, pub_year],
            [0, vertical_height],
            color=color,
            linewidth=1.5,
            alpha=0.5,
            dashes=dash_pattern,
        )

        # Use the same vertical height for annotation placement
        text_y = vertical_height
        text_offset = 0

        # Check if we need to adjust for overlapping annotations in the same year
        if pub_year in year_annotations:
            text_offset = year_annotations[pub_year] * 30
            year_annotations[pub_year] += 1
        else:
            year_annotations[pub_year] = 1

        # Add annotation text at the top of the vertical line
        annotation_text = f"{dataset_name}\n{first_author} et al.\n{int(pub_year)}"
        ax.annotate(
            annotation_text,
            xy=(pub_year, text_y),
            xytext=(pub_year, text_y),  # No additional offset
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=color,
        )

    # Set title and labels with Arial font
    ax.set_title("Dataset Citations by Year", fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Citations", fontsize=14, fontweight="bold", labelpad=10)

    # Style the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Add grid lines but only on the y-axis
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Ensure x-axis uses integer years and covers all relevant years
    ax.set_xlim(min(year_range) - 1, max(year_range) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set y-axis limit to accommodate annotations
    ax.set_ylim(0, y_limit)

    # Add legend without box
    legend = ax.legend(loc="upper left", frameon=False, fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    # File paths
    json_file = (
        "/home/mariopasc/Python/Projects/Review_Mammography/data/csvs/info_raw.json"
    )
    output_file = "/home/mariopasc/Python/Projects/Review_Mammography/data/plots/dataset_citations_enhanced.svg"

    # Create custom vertical heights for specific publication years
    # Values are proportions of the y-axis (0.0 to 1.0)
    custom_heights = {
        1995: 0.5,  # UCSF/LLNL
        1998: 0.5,  # DDSM
        2011: 0.6,  # BancoWeb
        2012: 0.5,  # INbreast
        2017: 0.85,  # CBIS-DDSM
        2019: 0.8,  # CSAW
        2020: 0.7,  # OPTIMAM
        2022.2: 0.9,
        2022.4: 0.8,
        2022: 0.65,
        2023: 0.45,  # CMMD
        # Add more as needed
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data and create visualization with custom heights
    dataset_data = load_dataset_data(json_file)
    create_enhanced_visualization(
        dataset_data, output_file, vertical_heights=custom_heights
    )
