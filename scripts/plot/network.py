import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import numpy as np
import yaml  # type: ignore
import os


def load_keywords_from_yaml(yaml_file):
    """
    Load keywords from a YAML configuration file, flattening any nested keyword sets.

    Args:
        yaml_file: Path to the YAML configuration file

    Returns:
        List of all keywords found in the file
    """
    try:
        # Read and parse the YAML file
        with open(yaml_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract keyword sets
        if "keyword_sets" in config:
            # Flatten all keyword sets into a single list
            all_keywords = []
            for keyword_set in config["keyword_sets"]:
                all_keywords.extend(keyword_set)
            return all_keywords
        elif "keywords" in config:
            # Handle old format with a flat keywords list
            return config["keywords"]
        else:
            print(f"Warning: No keywords found in {yaml_file}")
            return []

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {yaml_file}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {str(e)}")
        return []


def create_synonym_mappings(synonym_dict):
    """
    Create a mapping from each synonym to its preferred term.

    Args:
        synonym_dict: Dictionary where keys are preferred terms and values are
                     tuples/lists of synonyms

    Returns:
        Dictionary mapping each synonym (and preferred term) to its preferred term
    """
    mapping = {}

    # For each preferred term and its synonyms
    for preferred_term, synonyms in synonym_dict.items():
        # Map the preferred term to itself
        mapping[preferred_term.lower()] = preferred_term

        # Map each synonym to the preferred term
        for synonym in synonyms:
            mapping[synonym.lower()] = preferred_term

    return mapping


def generate_analysis_report(
    stats, output_file="keyword_analysis.txt", synonym_dict=None
):
    """
    Generate a text report of the keyword analysis

    Args:
        stats: Dictionary containing keyword statistics
        output_file: Path to save the report
        synonym_dict: Dictionary of synonyms used in the analysis
    """
    if not stats:
        print("No statistics available to generate report")
        return

    keyword_counts = stats["keyword_counts"]
    co_occurrence_counts = stats["co_occurrence_counts"]

    with open(output_file, "w") as f:
        f.write("Keyword Analysis Report\n")
        f.write("======================\n\n")

        # If synonyms were used, include that information
        if synonym_dict:
            f.write("Synonym Mappings Used:\n")
            f.write("--------------------\n")
            for preferred, synonyms in synonym_dict.items():
                f.write(f"{preferred}: {', '.join(synonyms)}\n")
            f.write("\n")

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


def create_keyword_network(
    csv_file,
    keywords,
    output_file="keyword_network.png",
    synonym_dict=None,
    dataset_terms=None,
):
    """
    Create a network visualization of keyword co-occurrences in abstracts and titles.

    Args:
        csv_file: Path to the CSV file containing the articles
        keywords: List of keywords to search for
        output_file: Path to save the visualization
        synonym_dict: Dictionary mapping preferred terms to their synonyms
        dataset_terms: Set of terms that represent datasets (for special styling)
    """
    # Create synonym mappings if provided
    synonym_mapping = {}
    if synonym_dict:
        synonym_mapping = create_synonym_mappings(synonym_dict)
        print(f"Created synonym mappings for {len(synonym_dict)} preferred terms")

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
        found_preferred_terms = set()

        for keyword in keywords:
            # Use word boundary to match whole words only
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text):
                # Map to preferred term if it's in the synonym mapping
                if synonym_mapping and keyword.lower() in synonym_mapping:
                    preferred_term = synonym_mapping[keyword.lower()]
                    found_preferred_terms.add(preferred_term)
                    keyword_counts[preferred_term] += 1
                else:
                    found_preferred_terms.add(keyword)
                    keyword_counts[keyword] += 1

        # Count co-occurrences using preferred terms
        if len(found_preferred_terms) > 1:
            for term1 in found_preferred_terms:
                for term2 in found_preferred_terms:
                    if term1 != term2:
                        co_occurrence_counts[term1][term2] += 1

    # Create a network graph
    G = nx.Graph()

    # Add nodes with sizes based on occurrence counts
    max_count = max(keyword_counts.values()) if keyword_counts else 1
    min_size = 300  # Minimum node size
    size_scale = 1500  # Scaling factor for node sizes

    # Add nodes
    for keyword, count in keyword_counts.items():
        # Only add nodes that appear at least once
        if count > 0:
            # Scale node size based on count
            node_size = min_size + (count / max_count) * size_scale
            # Add a flag for dataset nodes
            is_dataset = dataset_terms and keyword in dataset_terms
            G.add_node(keyword, size=node_size, count=count, is_dataset=is_dataset)

    # Add edges with weights based on co-occurrence counts
    max_co_occurrence = 1
    for kw1, co_occurrences in co_occurrence_counts.items():
        for kw2, count in co_occurrences.items():
            if count > 0:
                # Check if both nodes are datasets
                is_dataset_edge = False
                if dataset_terms:
                    if kw1 in dataset_terms and kw2 in dataset_terms:
                        is_dataset_edge = True

                G.add_edge(kw1, kw2, weight=count, is_dataset_edge=is_dataset_edge)
                max_co_occurrence = max(max_co_occurrence, count)

    # If no keywords were found
    if not G.nodes():
        print("No keywords found in the dataset")
        return None

    # Create the visualization
    plt.figure(figsize=(14, 10))

    # Use spring layout to position nodes
    pos = nx.spring_layout(G, k=6, iterations=100, seed=42)

    # Separate nodes by type (dataset vs non-dataset)
    dataset_nodes = [
        node for node in G.nodes() if G.nodes[node].get("is_dataset", False)
    ]
    other_nodes = [
        node for node in G.nodes() if not G.nodes[node].get("is_dataset", False)
    ]

    # Get node sizes
    dataset_node_sizes = [G.nodes[node]["size"] for node in dataset_nodes]
    other_node_sizes = [G.nodes[node]["size"] for node in other_nodes]

    # Get all node sizes for legend
    all_sizes = sorted([G.nodes[node]["size"] for node in G.nodes()])
    all_counts = sorted([G.nodes[node]["count"] for node in G.nodes()])

    # Define distinct colors for dataset and non-dataset nodes
    dataset_color = "#1f77b4"  # Blue
    other_color = "#ff7f0e"  # Orange

    # Draw nodes by type
    if dataset_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=dataset_nodes,
            node_size=dataset_node_sizes,
            node_color=dataset_color,
            alpha=0.8,
            label="Dataset",
        )

    if other_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=other_nodes,
            node_size=other_node_sizes,
            node_color=other_color,
            alpha=0.8,
            label="Non-Dataset",
        )

    # Choose a colormap for edges
    edge_cmap = plt.cm.plasma

    # Get edge weights for coloring
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    # Draw all edges with curved style
    nx.draw_networkx_edges(
        G,
        pos,
        width=3,  # Fixed width for all edges
        edge_color=edge_weights,
        edge_cmap=edge_cmap,
        alpha=0.7,
        edge_vmin=0,
        edge_vmax=max_co_occurrence,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title("Keyword Co-occurrence Network in Mammography Literature", fontsize=16)
    plt.axis("off")

    # Add the node type legend automatically through matplotlib
    plt.legend(loc="lower right", fontsize=10)

    # Add edge colorbar for co-occurrence
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(0, max_co_occurrence)),
        ax=plt.gca(),
        orientation="vertical",
        pad=0.05,
        shrink=0.5,
    )
    cbar.set_label("Co-occurrence Frequency")

    # Save the main figure
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


if __name__ == "__main__":
    # Path to the YAML configuration file
    yaml_file = "/home/mariopasc/Python/Projects/Review_Mammography/scripts/fetch/parameters.yaml"

    # Path to the CSV file
    csv_file = (
        "/home/mariopasc/Python/Projects/Review_Mammography/data/combined_results.csv"
    )

    # Output files
    network_output = "data/plots/keyword_network.png"
    report_output = "data/plots/keyword_analysis.txt"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(network_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load keywords from YAML file
    keywords = load_keywords_from_yaml(yaml_file)

    # Define synonym dictionary
    # Format: {preferred_term: (synonym1, synonym2, ...)}
    synonym_dict = {
        "Machine Learning": (
            "machine learning",
            "ML",
            "machine learning model",
            "model",
        ),
        "Deep Learning": ("deep learning", "DL", "deep neural network"),
        "Neural Network": ("neural network", "NN", "artificial neural network"),
        "Convolutional Neural Network": ("convolutional neural network", "CNN"),
        "Artificial Intelligence": ("artificial intelligence", "AI"),
        "Computer Vision": ("computer vision",),
        "Classification": ("image classification", "classification"),
        "Segmentation": ("image segmentation", "segmentation", "segment"),
        "Detection": ("object detection", "detection", "localization"),
        "DDSM": ("DDSM", "Digital Database for Screening Mammography"),
        "CBIS-DDSM": ("CBIS-DDSM", "Curated Breast Imaging Subset of DDSM"),
        "MIAS": ("MIAS", "Mammographic Image Analysis Society database"),
        "BancoWeb": ("BancoWeb", "LAPIMO", "Online Mammographic Images Database"),
        "INbreast": ("INbreast",),
        "VinDr-Mammo": ("VinDr", "VinDr-Mammo"),
        "BCDR": ("BCDR", "BCDR-FM", "BCDR-DM", "Breast Cancer Digital Repository"),
        "RSNA": ("RSNA", "RSNA Screening Mammography Breast Cancer Detection"),
        "OPTIMAM": ("OPTIMAM", "OPTIMAM Mammography Image Database"),
        "CSAW": ("CSAW", "Cohort of Screen-Aged Women"),
        "EMBED": ("EMBED", "EMory BrEast imaging Dataset"),
        "ADMANI": (
            "ADMANI",
            "Annotated Digital Mammograms and Associated Non-Image Datasets",
        ),
    }

    # Define which terms are dataset-related
    dataset_terms = {
        "DDSM",
        "CBIS-DDSM",
        "MIAS",
        "UCSF",
        "CMMD",
        "BancoWeb",
        "INbreast",
        "VinDr-Mammo",
        "OPTIMAM",
        "BCDR",
        "RSNA",
        "OPTIMAM",
        "CSAW",
        "EMBED",
        "ADMANI",
    }

    if not keywords:
        print("No keywords loaded from configuration file. Exiting.")
    else:
        print(f"Loaded {len(keywords)} keywords from configuration file")

        # Create the network visualization with synonym handling and dataset edge styling
        stats = create_keyword_network(
            csv_file,
            keywords,
            network_output,
            synonym_dict=synonym_dict,
            dataset_terms=dataset_terms,
        )

        # Generate detailed analysis report
        if stats:
            generate_analysis_report(stats, report_output, synonym_dict=synonym_dict)
        else:
            print("No statistics generated. No keywords found in the articles.")
