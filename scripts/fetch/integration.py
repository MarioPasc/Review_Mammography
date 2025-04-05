"""
Integration tool for multiple literature search APIs.
Combines results from NCBI, arXiv, and potentially other sources into a single dataset.
"""

import os
import csv
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable, Tuple
import importlib
import sys
from datetime import datetime
from difflib import SequenceMatcher
import yaml  # type: ignore

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


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate articles from combined results based on multiple criteria.

    Args:
        df: DataFrame containing combined article data

    Returns:
        Deduplicated DataFrame
    """
    print(f"Starting deduplication process on {len(df)} articles...")

    # First pass: deduplicate based on DOI if available
    if "doi" in df.columns and not df["doi"].isna().all():
        # Remove NaN values for comparison
        df["doi_clean"] = df["doi"].fillna("")
        # Make DOIs lowercase for comparison
        df["doi_clean"] = df["doi_clean"].str.lower()
        # Group by DOI and keep row with most non-null values
        doi_groups = df.groupby("doi_clean")
        to_keep = []

        for _, group in doi_groups:
            if len(group) > 1:
                # Keep the row with most non-null values
                non_null_counts = group.count(axis=1)
                to_keep.append(group.iloc[non_null_counts.argmax()].name)
            else:
                # If only one entry, keep it
                to_keep.append(group.iloc[0].name)

        df_dedupe = df.loc[to_keep].copy()
        df_dedupe.drop(columns=["doi_clean"], inplace=True)
        print(f"After DOI deduplication: {len(df_dedupe)} articles")
    else:
        df_dedupe = df.copy()

    # Second pass: deduplicate based on title similarity for remaining entries
    if "title" in df_dedupe.columns:
        # Fill missing titles with empty strings for comparison
        df_dedupe["title_clean"] = df_dedupe["title"].fillna("").str.strip().str.lower()

        # Initialize list to track indices to drop
        to_drop = []

        # Loop through DataFrame to find similar titles
        for i, row_i in df_dedupe.iterrows():
            if i in to_drop:
                continue

            title_i = row_i["title_clean"]
            if not title_i:  # Skip empty titles
                continue

            for j, row_j in df_dedupe.iloc[i + 1 :].iterrows():  # type: ignore
                if j in to_drop:
                    continue

                title_j = row_j["title_clean"]
                if not title_j:  # Skip empty titles
                    continue

                # Calculate title similarity
                similarity = SequenceMatcher(None, title_i, title_j).ratio()

                # If titles are similar, consider them duplicates
                if similarity > 0.85:
                    # Keep the row with more non-null values
                    row_i_count = row_i.count()
                    row_j_count = row_j.count()

                    if row_j_count > row_i_count:
                        to_drop.append(i)
                        break
                    else:
                        to_drop.append(j)

        # Drop duplicates
        df_dedupe = df_dedupe.drop(to_drop).copy()
        df_dedupe.drop(columns=["title_clean"], inplace=True)
        print(f"After title similarity deduplication: {len(df_dedupe)} articles")

    # Reset index for clean output
    df_dedupe.reset_index(drop=True, inplace=True)

    return df_dedupe


def combine_results(
    results: Dict[str, List[Dict[str, Any]]],
    output_file: str = "data/combined_results.csv",
    config_file: str = "./scripts/fetch/parameters.yaml",
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

    # Convert to DataFrame for deduplication
    combined_df = pd.DataFrame(combined_data)

    # Remove duplicates
    deduplicated_df = remove_duplicates(combined_df)

    # Filter by dataset keywords
    filtered_df = filter_by_dataset_keywords(deduplicated_df, config_file)

    # Write to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    print(
        f"Combined {len(combined_data)} articles from {len(results)} sources to {output_file}"
    )
    print(f"After deduplication: {len(deduplicated_df)} unique articles")
    print(
        f"After dataset keyword filtering: {len(filtered_df)} articles saved to {output_file}"
    )


def combine_from_csv_files(
    csv_files: Dict[str, str],
    output_file: str = "data/combined_results.csv",
    config_file: str = "./scripts/fetch/parameters.yaml",
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

    # Remove duplicates
    deduplicated_df = remove_duplicates(combined_df)

    # Filter by dataset keywords
    filtered_df = filter_by_dataset_keywords(deduplicated_df, config_file)

    # Save combined CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    print(
        f"Combined {len(combined_df)} articles from {len(csv_files)} sources to {output_file}"
    )
    print(f"After deduplication: {len(deduplicated_df)} unique articles")
    print(
        f"After dataset keyword filtering: {len(filtered_df)} articles saved to {output_file}"
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
    # sources["arXiv"] = arxiv_fetch.search_from_yaml
    sources["Pubmed"] = ncbi.search_from_yaml
    sources["SemanticScholar"] = semantic_scholar_fetch.search_from_yaml

    # Fetch from all sources
    results, csv_files = fetch_from_all_sources(config_file, sources)

    # Combine results
    if any(len(result_list) > 0 for result_list in results.values()):
        combine_results(results, output_file, config_file)
    elif csv_files:
        # Fallback to combining from CSV files if direct results aren't available
        combine_from_csv_files(csv_files, output_file, config_file)
    else:
        print("No results to combine")


def filter_by_dataset_keywords(df: pd.DataFrame, config_file: str) -> pd.DataFrame:
    """
    Filter the dataframe to include only entries that mention at least one dataset keyword.

    Args:
        df: DataFrame containing combined article data
        config_file: Path to the YAML configuration file with dataset keywords

    Returns:
        Filtered DataFrame
    """
    # Read the configuration file to get dataset keywords
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        return df

    # Get the dataset keywords (first set in keyword_sets)
    if (
        "keyword_sets" not in config
        or not config["keyword_sets"]
        or not config["keyword_sets"][0]
    ):
        print("No dataset keywords found in config file")
        return df

    dataset_keywords = config["keyword_sets"][0]
    print(f"Filtering by {len(dataset_keywords)} dataset keywords")

    # Fields to check for keywords
    fields_to_check = ["title", "abstract", "keywords", "mesh_terms"]

    # Filter function that returns True if any dataset keyword is found in relevant fields
    def contains_dataset_keyword(row):
        for field in fields_to_check:
            if field in row and isinstance(row[field], str):
                text = row[field].lower()
                if any(keyword.lower() in text for keyword in dataset_keywords):
                    return True
        return False

    # Apply filter
    original_count = len(df)
    filtered_df = df[df.apply(contains_dataset_keyword, axis=1)]
    print(
        f"Dataset keyword filter: kept {len(filtered_df)} of {original_count} articles"
    )

    return filtered_df


if __name__ == "__main__":
    # Default configuration and output files
    config_file = "./scripts/fetch/parameters.yaml"
    output_file = "./data/combined_results.csv"

    # Run the main function
    main(config_file, output_file)
