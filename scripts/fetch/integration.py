"""
Integration tool for multiple literature search APIs.
Combines results from NCBI, arXiv, and potentially other sources into a single dataset.
"""

import os
import csv
import time
import pandas as pd
from typing import Dict, Any, List, Callable, Tuple
import importlib
import sys
from datetime import datetime


# Add the parent directory to the Python path to import sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import arxiv_fetch  # type: ignore
from fetch import ncbi_fetch as ncbi
from fetch import semantic_scholar_fetch


def fetch_from_all_sources(
    config_file: str, sources: Dict[str, Callable], output_dir: str = "data"
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    Fetch articles from all configured sources using the same configuration file.

    Args:
        config_file: Path to the YAML configuration file
        sources: Dictionary mapping source names to their search functions
        output_dir: Directory to store individual source results

    Returns:
        Tuple containing:
        - Dictionary mapping source names to their results
        - Dictionary mapping source names to their CSV output files
    """
    results = {}
    csv_files = {}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Current timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Starting fetches from {len(sources)} sources using config: {config_file}")

    # Fetch from each source
    for source_name, search_function in sources.items():
        print(f"\n--- Fetching from {source_name} ---")
        start_time = time.time()

        # Call the search function
        try:
            source_results = search_function(config_file)
            results[source_name] = source_results

            # Save individual source results
            if source_results:
                csv_path = os.path.join(
                    output_dir, f"{source_name.lower()}_{timestamp}.csv"
                )

                # If the search function doesn't save a CSV, we'll do it here
                if not any(
                    file.endswith(".csv")
                    for file in os.listdir(output_dir)
                    if os.path.isfile(os.path.join(output_dir, file))
                    and os.path.getmtime(os.path.join(output_dir, file)) > start_time
                ):

                    print(
                        f"Saving {len(source_results)} results from {source_name} to {csv_path}"
                    )

                    # Get all fields from all articles
                    all_fields = set()
                    for article in source_results:
                        all_fields.update(article.keys())

                    # Write to CSV
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(list(all_fields)))
                        writer.writeheader()

                        for article in source_results:
                            # Convert lists to strings for CSV
                            processed_article = {}
                            for key, value in article.items():
                                if isinstance(value, list):
                                    processed_article[key] = "; ".join(
                                        str(v) for v in value
                                    )
                                else:
                                    processed_article[key] = value
                            writer.writerow(processed_article)

                # Record the CSV file path
                csv_files[source_name] = csv_path
            else:
                print(f"No results returned from {source_name}")

            print(
                f"Completed {source_name} fetch in {time.time() - start_time:.2f} seconds"
            )

        except Exception as e:
            print(f"Error fetching from {source_name}: {str(e)}")

    return results, csv_files


def combine_results(
    results: Dict[str, List[Dict[str, Any]]],
    output_file: str = "data/combined_results.csv",
) -> None:
    """
    Combine results from multiple sources into a single CSV file.

    Args:
        results: Dictionary mapping source names to their results
        output_file: Path for the output CSV file
    """
    combined_data = []

    # Process each source's results
    for source_name, source_results in results.items():
        for article in source_results:
            # Add source information
            article_copy = article.copy()
            article_copy["source"] = source_name
            combined_data.append(article_copy)

    if not combined_data:
        print("No data to combine")
        return

    # Get all fields from all articles
    all_fields: set = set()
    for article in combined_data:
        all_fields.update(article.keys())

    # Ensure 'source' is included in the fields
    all_fields.add("source")

    # Write to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(list(all_fields)))
        writer.writeheader()

        for article in combined_data:
            # Convert lists to strings for CSV
            processed_article = {}
            for key, value in article.items():
                if isinstance(value, list):
                    processed_article[key] = "; ".join(str(v) for v in value)
                else:
                    processed_article[key] = value
            writer.writerow(processed_article)

    print(
        f"Combined {len(combined_data)} articles from {len(results)} sources to {output_file}"
    )


def combine_from_csv_files(
    csv_files: Dict[str, str], output_file: str = "data/combined_results.csv"
) -> None:
    """
    Combine results from multiple CSV files into a single CSV file.

    Args:
        csv_files: Dictionary mapping source names to their CSV files
        output_file: Path for the output CSV file
    """
    combined_df = None

    # Process each source's CSV
    for source_name, csv_path in csv_files.items():
        try:
            # Read the CSV
            df = pd.read_csv(csv_path)

            # Add source information
            df["source"] = source_name

            # Append to combined dataframe
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            print(f"Added {len(df)} rows from {source_name}")

        except Exception as e:
            print(f"Error reading CSV from {source_name}: {str(e)}")

    if combined_df is None or combined_df.empty:
        print("No data to combine")
        return

    # Save combined CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(
        f"Combined {len(combined_df)} articles from {len(csv_files)} sources to {output_file}"
    )


def main(config_file: str, output_file: str = "data/combined_results.csv") -> None:
    """
    Main function to fetch and combine results from all sources.

    Args:
        config_file: Path to the YAML configuration file
        output_file: Path for the output CSV file
    """
    # Define available sources with their search functions
    sources = {}

    # Add arXiv
    sources["arXiv"] = arxiv_fetch.search_from_yaml
    sources["Pubmed"] = ncbi.search_from_yaml
    sources["SemanticScholar"] = semantic_scholar_fetch.search_from_yaml

    # Fetch from all sources
    results, csv_files = fetch_from_all_sources(config_file, sources)

    # Combine results
    if any(len(result_list) > 0 for result_list in results.values()):
        combine_results(results, output_file)
    elif csv_files:
        # Fallback to combining from CSV files if direct results aren't available
        combine_from_csv_files(csv_files, output_file)
    else:
        print("No results to combine")


if __name__ == "__main__":
    # Default configuration and output files
    config_file = "./scripts/fetch/parameters.yaml"
    output_file = "./data/combined_results.csv"

    # Run the main function
    main(config_file, output_file)
