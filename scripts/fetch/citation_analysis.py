"""
Dataset Citation Analysis

This script analyzes citations for mammography datasets:
1. Fetches publication year and first author for each dataset
2. Retrieves citation counts by year for each dataset
3. Produces statistics on dataset usage over time
"""

import time
import yaml  # type: ignore
import os
import json
import pandas as pd
import requests  # type: ignore
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback

# API constants
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_CITATIONS_ENDPOINT = "/paper"
API_SLEEP_TIME = 1  # seconds

# Dataset ID to name mapping (from papers_semantic_scholar.py)
DATASET_MAPPING = {
    "10.1007/978-94-011-5318-8_75": "DDSM",
    "10.1007/s10278-010-9297-2": "BancoWeb",
    "10.17863/CAM.105113": "MIAS",
    "10.1109/IEMBS.1995.575239": "UCSF/LLNL",
    "10.1038/sdata.2017.177": "CBIS-DDSM",
    "10.1016/j.acra.2011.09.014": "INbreast",
    "10.1101/2022.03.07.22272009": "VinDr-Mammo",
    "10.1038/s41597-023-02025-1": "CMMD",
    "10.1148/ryai.2020200103": "OPTIMAM",
    "10.1007/s10278-019-00278-0": "CSAW",
    "10.48550/arXiv.2202.04073": "EMBED",
    "corpusid:246680284": "EMBED",
    "10.1148/ryai.220072": "ADMANI",
    "10.48550/arXiv.2411.02710": "BCDR",
    "corpusid:273821203": "BCDR",
}

# Corpus ID prefixes
CORPUS_ID_PREFIXES = {
    "arxiv": "arXiv:",
    "pubmed": "PMID:",
    "mag": "MAG:",
    "corpusid": "CorpusID:",
}


def configure_api_key(api_key: Optional[str] = None) -> Dict[str, str]:
    """Configure API headers with optional API key"""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def get_paper_id_from_doi(
    doi: str, api_key: Optional[str] = None, retries: int = 3
) -> Optional[str]:
    """Convert a DOI to a Semantic Scholar Paper ID"""
    headers = configure_api_key(api_key)

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/DOI:{doi}",
                headers=headers,
                params={"fields": "paperId"},
            )

            response.raise_for_status()
            data = response.json()
            return data.get("paperId")

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error fetching paper ID for DOI {doi}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)
            else:
                print(
                    f"Failed to fetch paper ID for DOI {doi} after {retries} attempts"
                )

    return None


def get_paper_id_from_corpus_id(
    corpus_type: str, corpus_id: str, api_key: Optional[str] = None, retries: int = 3
) -> Optional[str]:
    """Convert a corpus ID to a Semantic Scholar Paper ID"""
    headers = configure_api_key(api_key)

    prefix = CORPUS_ID_PREFIXES.get(corpus_type.lower(), "")
    if not prefix:
        print(f"Unknown corpus type: {corpus_type}")
        return None

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{prefix}{corpus_id}",
                headers=headers,
                params={"fields": "paperId"},
            )

            response.raise_for_status()
            data = response.json()
            return data.get("paperId")

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error fetching paper ID for {corpus_type} ID {corpus_id}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)
            else:
                print(
                    f"Failed to fetch paper ID for {corpus_type} ID {corpus_id} after {retries} attempts"
                )

    return None


def get_paper_details(
    paper_id: str, api_key: Optional[str] = None, retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information about a paper by its Semantic Scholar ID

    Args:
        paper_id: Semantic Scholar Paper ID
        api_key: Optional API key
        retries: Number of retry attempts

    Returns:
        Dictionary containing paper details
    """
    headers = configure_api_key(api_key)
    fields = (
        "paperId,title,authors,year,publicationDate,venue,citationCount,isOpenAccess"
    )

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}",
                headers=headers,
                params={"fields": fields},
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error fetching paper details for {paper_id}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)
            else:
                print(
                    f"Failed to fetch paper details for {paper_id} after {retries} attempts"
                )

    return None


def get_paper_id_from_identifier(
    identifier: str, api_key: Optional[str] = None
) -> Optional[str]:
    """
    Convert various identifier types to Semantic Scholar Paper ID

    Args:
        identifier: DOI, corpus ID, or Semantic Scholar Paper ID
        api_key: Optional API key

    Returns:
        Semantic Scholar Paper ID
    """
    if identifier.startswith("10."):  # DOI
        paper_id = get_paper_id_from_doi(identifier, api_key)
        if not paper_id:
            print(f"Could not get paper ID for DOI: {identifier}")
        return paper_id

    elif ":" in identifier:  # Corpus ID
        parts = identifier.split(":", 1)
        if len(parts) != 2:
            print(f"Invalid corpus ID format: {identifier}")
            return None

        corpus_type, corpus_id = parts
        paper_id = get_paper_id_from_corpus_id(corpus_type, corpus_id, api_key)
        if not paper_id:
            print(f"Could not get paper ID for {corpus_type} ID: {corpus_id}")
        return paper_id

    # Assume it's already a Semantic Scholar paper ID
    return identifier


def fetch_citations_count_by_year(
    paper_id: str, api_key: Optional[str] = None, retries: int = 3
) -> Dict[str, int]:
    """
    Fetch all citations and group them by year

    Args:
        paper_id: Semantic Scholar Paper ID
        api_key: Optional API key
        retries: Number of retry attempts

    Returns:
        Dictionary mapping years to citation counts
    """
    headers = configure_api_key(api_key)
    citations_by_year: dict = {}

    # Set up pagination
    limit = 100
    offset = 0
    total_citations = 0

    while True:
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}/citations",
                    headers=headers,
                    params={
                        "fields": "citingPaper.year",
                        "limit": limit,
                        "offset": offset,
                    },
                )

                response.raise_for_status()
                data = response.json()

                if "data" not in data or not isinstance(data["data"], list):
                    print(f"Unexpected response format for paper {paper_id}")
                    return citations_by_year

                citations = data["data"]
                if not citations:
                    # No more citations
                    return citations_by_year

                # Process this batch of citations
                for citation in citations:
                    citing_paper = citation.get("citingPaper", {})
                    year = citing_paper.get("year")

                    if year is not None:
                        year_str = str(year)
                        if year_str in citations_by_year:
                            citations_by_year[year_str] += 1
                        else:
                            citations_by_year[year_str] = 1

                total_citations += len(citations)
                print(f"Processed {len(citations)} citations, total: {total_citations}")

                if len(citations) < limit:
                    # This was the last page
                    return citations_by_year

                # Continue to next page
                offset += limit
                time.sleep(API_SLEEP_TIME)
                break

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(
                        f"Error fetching citations for {paper_id}, attempt {attempt + 1}: {str(e)}"
                    )
                    time.sleep(API_SLEEP_TIME * 2)
                else:
                    print(
                        f"Failed to fetch citations for {paper_id} after {retries} attempts"
                    )
                    return citations_by_year

    return citations_by_year


def analyze_dataset_citations(config_file: str) -> None:
    """
    Analyze citations for each dataset defined in the configuration

    Args:
        config_file: Path to YAML configuration file
    """
    try:
        # Read configuration
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        api_key = config.get("semantic_scholar_api_key", None)
        output_prefix = config.get("output_prefix", "data/dataset_citations")

        # Results storage
        dataset_results = []

        # Create output directory
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process each dataset
        for identifier, dataset_name in DATASET_MAPPING.items():
            print(f"\n===== Processing dataset: {dataset_name} ({identifier}) =====")

            # Get Semantic Scholar paper ID
            paper_id = get_paper_id_from_identifier(identifier, api_key)
            if not paper_id:
                print(f"Skipping {dataset_name}: could not get paper ID")
                continue

            # Get dataset paper details
            paper_details = get_paper_details(paper_id, api_key)
            if not paper_details:
                print(f"Skipping {dataset_name}: could not get paper details")
                continue

            # Extract publication year and first author
            publication_year = paper_details.get("year")
            authors = paper_details.get("authors", [])
            first_author = authors[0].get("name") if authors else "Unknown"

            print(f"Dataset: {dataset_name}")
            print(f"Publication year: {publication_year}")
            print(f"First author: {first_author}")
            print(
                f"Total citation count: {paper_details.get('citationCount', 'Unknown')}"
            )

            # Get citations by year
            print(f"Fetching citation data by year...")
            citations_by_year = fetch_citations_count_by_year(paper_id, api_key)

            # Add to results
            dataset_result = {
                "dataset": dataset_name,
                "paper_id": paper_id,
                "publication_year": publication_year,
                "first_author": first_author,
                "total_citations": paper_details.get("citationCount", 0),
                "citations_by_year": citations_by_year,
            }

            dataset_results.append(dataset_result)

        # Generate CSV of results
        if dataset_results:
            # Create a DataFrame for the summary
            summary_data = []
            years_set = set()

            for result in dataset_results:
                # Collect all years across all datasets
                years_set.update(result["citations_by_year"].keys())

                summary_data.append(
                    {
                        "dataset": result["dataset"],
                        "publication_year": result["publication_year"],
                        "first_author": result["first_author"],
                        "total_citations": result["total_citations"],
                    }
                )

            # Sort years
            all_years = sorted(list(years_set))

            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data)

            # Create citations by year DataFrame
            citations_data = []
            for result in dataset_results:
                row = {"dataset": result["dataset"]}
                for year in all_years:
                    row[f"citations_{year}"] = result["citations_by_year"].get(year, 0)
                citations_data.append(row)

            citations_df = pd.DataFrame(citations_data)

            # Save results
            summary_file = f"{output_prefix}_summary.csv"
            citations_file = f"{output_prefix}_by_year.csv"

            summary_df.to_csv(summary_file, index=False)
            citations_df.to_csv(citations_file, index=False)

            print(f"\nResults saved to:")
            print(f"  - Summary: {summary_file}")
            print(f"  - Citations by year: {citations_file}")

            # Save raw data as JSON
            json_file = f"{output_prefix}_raw.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(dataset_results, f, indent=2)

            print(f"  - Raw data: {json_file}")

            # Generate visualization
            try:
                plt.figure(figsize=(12, 8))

                for result in dataset_results:
                    dataset = result["dataset"]
                    years = sorted([int(y) for y in result["citations_by_year"].keys()])
                    counts = [result["citations_by_year"].get(str(y), 0) for y in years]

                    # Only plot if we have data
                    if years and counts:
                        plt.plot(
                            years, counts, marker="o", linestyle="-", label=dataset
                        )

                plt.title("Dataset Citations by Year")
                plt.xlabel("Year")
                plt.ylabel("Citations")
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend(loc="upper left")

                plot_file = f"{output_prefix}_plot.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                print(f"  - Citation plot: {plot_file}")

            except Exception as e:
                print(f"Error generating visualization: {str(e)}")

    except Exception as e:
        print(f"Error analyzing dataset citations: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    analyze_dataset_citations("./scripts/fetch/parameters.yaml")
