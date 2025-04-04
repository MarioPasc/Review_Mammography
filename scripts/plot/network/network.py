import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import numpy as np


def create_keyword_network(csv_file, keywords, output_file="keyword_network.png"):
    """
    Create a network visualization of keyword co-occurrences in abstracts and titles.

    Args:
        csv_file: Path to the CSV file containing the articles
        keywords: List of keywords to search for
        output_file: Path to save the visualization
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create dictionaries to store keyword occurrences and co-occurrences
    keyword_counts = Counter()
    co_occurrence_counts = defaultdict(Counter)

    # Process each article
    for _, row in df.iterrows():
        # Combine title and abstract for searching
        text = f"{row['title']} {row['abstract']}".lower()

        # Find keywords in the text
        found_keywords = set()
        for keyword in keywords:
            # Use word boundary to match whole words only
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text):
                found_keywords.add(keyword)
                keyword_counts[keyword] += 1

        # Count co-occurrences
        if len(found_keywords) > 1:
            for kw1 in found_keywords:
                for kw2 in found_keywords:
                    if kw1 != kw2:
                        co_occurrence_counts[kw1][kw2] += 1

    # Create a network graph
    G = nx.Graph()

    # Add nodes with sizes based on occurrence counts
    max_count = max(keyword_counts.values()) if keyword_counts else 1
    min_size = 300  # Minimum node size
    size_scale = 1500  # Scaling factor for node sizes

    for keyword, count in keyword_counts.items():
        # Only add nodes that appear at least once
        if count > 0:
            # Scale node size based on count
            node_size = min_size + (count / max_count) * size_scale
            G.add_node(keyword, size=node_size, count=count)

    # Add edges with weights based on co-occurrence counts
    max_co_occurrence = 1
    for kw1, co_occurrences in co_occurrence_counts.items():
        for kw2, count in co_occurrences.items():
            if count > 0:
                G.add_edge(kw1, kw2, weight=count)
                max_co_occurrence = max(max_co_occurrence, count)

    # If no keywords were found
    if not G.nodes():
        print("No keywords found in the dataset")
        return

    # Create the visualization
    plt.figure(figsize=(14, 10))

    # Use spring layout to position nodes
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Get node sizes
    node_sizes = [G.nodes[node]["size"] for node in G.nodes()]

    # Get edge weights for line width
    edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]

    # Draw the network
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title("Keyword Co-occurrence Network in Mammography Literature", fontsize=16)
    plt.axis("off")

    # Add legend for node and edge sizes
    min_count = min(keyword_counts.values())
    legend_text = (
        f"Node size: proportional to keyword frequency (min={min_count}, max={max_count})\n"
        f"Edge width: proportional to co-occurrence frequency (max={max_co_occurrence})"
    )
    plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=10)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Network visualization saved to {output_file}")

    # Return statistics for further analysis
    return {
        "keyword_counts": dict(keyword_counts),
        "co_occurrence_counts": {k: dict(v) for k, v in co_occurrence_counts.items()},
        "total_keywords_found": len(keyword_counts),
        "total_co_occurrences": sum(
            sum(v.values()) for v in co_occurrence_counts.values()
        )
        // 2,
    }


def generate_analysis_report(stats, output_file="keyword_analysis.txt"):
    """Generate a text report of the keyword analysis"""
    keyword_counts = stats["keyword_counts"]
    co_occurrence_counts = stats["co_occurrence_counts"]

    with open(output_file, "w") as f:
        f.write("Keyword Analysis Report\n")
        f.write("======================\n\n")

        # Keyword occurrence statistics
        f.write("Keyword Occurrences:\n")
        f.write("-------------------\n")
        for keyword, count in sorted(
            keyword_counts.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"{keyword}: {count}\n")

        f.write("\nKeyword Co-occurrences:\n")
        f.write("----------------------\n")

        # Find all unique co-occurrence pairs
        co_occurrences = []
        for kw1, counters in co_occurrence_counts.items():
            for kw2, count in counters.items():
                if kw1 < kw2:  # Avoid counting pairs twice
                    co_occurrences.append((kw1, kw2, count))

        # Sort by count and print
        for kw1, kw2, count in sorted(co_occurrences, key=lambda x: x[2], reverse=True):
            f.write(f"{kw1} - {kw2}: {count}\n")

    print(f"Analysis report saved to {output_file}")


if __name__ == "__main__":
    # List of keywords to search for
    keywords = [
        "DDSM",
        "CBIS-DDSM",
        "MIAS",
        "BancoWeb",
        "LAPIMO",
        "UCSF",
        "LLNL",
        "INbreast",
        "VinDr",
        "VinDr-Mammo",
        "CMMD",
        "RSNA",
        "OPTIMAM",
        "CSAW",
        "EMBED",
        "ADMANI",
        "BCDR",
        "BCDR-FM",
        "BCDR-DM",
        "deep learning",
        "convolutional neural network",
        "neural network",
        "machine learning",
        "artificial intelligence",
        "CNN",
        "NN",
        "DL",
        "ML",
    ]

    # Path to the CSV file
    csv_file = (
        "/home/mariopasc/Python/Projects/Review_Mammography/data/ncbi_results.csv"
    )

    # Create the network visualization
    stats = create_keyword_network(csv_file, keywords, "keyword_network.png")

    # Generate detailed analysis report
    generate_analysis_report(stats, "keyword_analysis.txt")
