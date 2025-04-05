import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import numpy as np
import yaml  # type: ignore
import os
import seaborn as sns  # type: ignore


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
    config_file=None,
):
    """
    Create a network visualization of keyword co-occurrences in abstracts and titles.

    Args:
        csv_file: Path to the CSV file containing the articles
        keywords: List of keywords to search for
        output_file: Path to save the visualization
        synonym_dict: Dictionary mapping preferred terms to their synonyms
        config_file: Path to the config file with keyword categories
    """
    # Create synonym mappings if provided
    synonym_mapping = {}
    if synonym_dict:
        synonym_mapping = create_synonym_mappings(synonym_dict)
        print(f"Created synonym mappings for {len(synonym_dict)} preferred terms")

    # Load keyword categories from config file
    keyword_categories = {}
    if config_file:
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            if "keyword_sets" in config:
                # Extract the first 3 keyword sets (Dataset, CS, Medical)
                category_names = [
                    "Dataset",
                    "Computer Science",
                    "Medical Terms",
                    "Computer Science",
                ]

                for i, (category_name, keyword_set) in enumerate(
                    zip(category_names, config["keyword_sets"][:4])
                ):
                    for kw in keyword_set:
                        # Map each keyword to its category
                        # If using synonyms, map both the keyword and its preferred term
                        if synonym_mapping and kw.lower() in synonym_mapping:
                            preferred_term = synonym_mapping[kw.lower()]
                            keyword_categories[preferred_term] = category_name
                        else:
                            keyword_categories[kw] = category_name
        except Exception as e:
            print(f"Error loading categories from config: {e}")

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
            # Assign the category
            category = keyword_categories.get(keyword, "Other")
            G.add_node(keyword, size=node_size + 20, count=count, category=category)

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
        return None

    # Create the visualization
    plt.figure(figsize=(14, 10))

    # Use spring layout to position nodes
    pos = nx.spring_layout(G, k=8, iterations=100, seed=1)

    # Separate nodes by category
    category_colors = {
        "Dataset": "#DDAA33",  # Blue
        "Computer Science": "#BB5566",  # Orange
        "Medical Terms": "#6699CC",  # Green
        "Other": "#d62728",  # Red for uncategorized terms
    }

    # Group nodes by category
    categorized_nodes = {category: [] for category in category_colors}
    for node in G.nodes():
        category = G.nodes[node].get("category", "Other")
        categorized_nodes[category].append(node)

    # Draw nodes by category
    for category, nodes in categorized_nodes.items():
        if nodes:
            # Get node sizes
            node_sizes = [G.nodes[node]["size"] for node in nodes]

            # Draw nodes
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes,
                node_size=node_sizes,
                node_color=category_colors[category],
                alpha=0.8,
                label=category,
            )

    # Draw edges
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_cmap = sns.color_palette("rocket_r", as_cmap=True)

    # Draw all edges with curved style
    nx.draw_networkx_edges(
        G,
        pos,
        width=3,
        edge_color=edge_weights,
        edge_cmap=edge_cmap,
        alpha=0.5,
        edge_vmin=0,
        edge_vmax=max_co_occurrence,
    )

    # Calculate font sizes based on node sizes
    font_sizes = {}
    for node in G.nodes():
        # Get the original node size
        node_size = G.nodes[node]["size"]
        # Scale the font size based on the node size
        # You can adjust these values to get the desired scaling effect
        base_font_size = 8
        font_scale_factor = 0.005
        font_sizes[node] = base_font_size + (node_size * font_scale_factor)

    # Draw labels with variable font sizes
    nx.draw_networkx_labels(
        G, pos, font_size=font_sizes, font_weight="normal", font_family="arial black"
    )

    # plt.title("Keyword Co-occurrence Network in Mammography Literature", fontsize=16)
    plt.axis("off")

    # Create a custom legend with consistent marker sizes
    legend_elements = []
    for category, color in category_colors.items():
        if categorized_nodes[category]:  # Only add categories that have nodes
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,  # Consistent size for all legend markers
                    label=category,
                )
            )

    # Add custom legend
    plt.legend(handles=legend_elements, loc="lower left", fontsize=10)

    # Add edge colorbar for co-occurrence
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(0, max_co_occurrence)),
        ax=plt.gca(),
        orientation="horizontal",
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
    csv_file = "/home/mariopasc/Python/Projects/Review_Mammography/data/csvs/info_citations.csv"

    # Output files
    network_output = "data/plots/keyword_network.svg"
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
        "AUC": ("AUC", "area under the curve", "area under the ROC curve"),
        "ROC": ("ROC", "receiver operating characteristic", "ROC curve", "ROC-AUC"),
        "F1-Score": ("F1-Score", "F1 score", "F1"),
        "Sensitivity": ("sensitivity", "true positive rate"),
        "Specificity": ("specificity", "true negative rate"),
        "Accuracy": ("accuracy",),
        "Precision": (
            "precision",
            "positive predictive value",
            "positive predictive rate",
            "PPV",
        ),
        "Recall": ("recall", "sensitivity", "true positive rate"),
        "Precision-Recall Curve": (
            "precision-recall curve",
            "PR curve",
            "precision-recall",
        ),
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
        "CMMD": ("CMMD", "Chinese Mammography Database"),
        "Breast Cancer": ("breast cancer", "BC"),
        "Mammography": (
            "mammography",
            "screening mammography",
            "digital mammography",
            "breast imaging",
        ),
    }
    if not keywords:
        print("No keywords loaded from configuration file. Exiting.")
    else:
        print(f"Loaded {len(keywords)} keywords from configuration file")

        # Create the network visualization with the config file for categories
        stats = create_keyword_network(
            csv_file,
            keywords,
            network_output,
            synonym_dict=synonym_dict,
            config_file=yaml_file,  # Pass the config file
        )

        # Generate detailed analysis report
        if stats:
            generate_analysis_report(stats, report_output, synonym_dict=synonym_dict)
        else:
            print("No statistics generated. No keywords found in the articles.")
