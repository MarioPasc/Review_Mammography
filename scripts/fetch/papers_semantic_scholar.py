"""
Semantic Scholar citation fetcher module for retrieving papers that cite specific
papers or repositories, and filtering them based on AI-related keywords.
"""

import time
import yaml  # type: ignore
import os
import csv
import json
from typing import List, Dict, Any, Optional, Tuple
import requests  # type: ignore
from datetime import datetime
import pandas as pd

# Import common functions from the existing semantic scholar module
from semantic_scholar_fetch import (  # type: ignore
    configure_api_key,
    extract_article_metadata,
    save_results_to_csv,
    API_SLEEP_TIME,
    SEMANTIC_SCHOLAR_API_URL,
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manipulate import filter_citations

# Citation endpoint
PAPER_CITATIONS_ENDPOINT = "/paper"

# Corpus ID prefixes
CORPUS_ID_PREFIXES = {
    "arxiv": "arXiv:",
    "pubmed": "PMID:",
    "mag": "MAG:",
    "corpusid": "CorpusID:",
}

# Dataset ID to name mapping
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
    "10.1117/12.2041674": "OMI-DB",
    "10.1007/s10278-019-00278-0": "CSAW",
    "10.48550/arXiv.2202.04073": "EMBED",
    "corpusid:246680284": "EMBED",
    "10.1148/ryai.220072": "ADMANI",
    "10.48550/arXiv.2411.02710": "BCDR",
    "corpusid:273821203": "BCDR",
}


def check_doi_exists(doi: str, api_key: Optional[str] = None, retries: int = 3) -> bool:
    """
    Check if a DOI exists in Semantic Scholar's database.

    Args:
        doi: Digital Object Identifier
        api_key: Optional API key
        retries: Number of retries for failed requests

    Returns:
        True if DOI exists in Semantic Scholar, False otherwise
    """
    headers = configure_api_key(api_key)

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/DOI:{doi}",
                headers=headers,
                params={"fields": "paperId"},
            )

            # 200 status means the DOI exists
            if response.status_code == 200:
                return True

            # 404 means not found
            if response.status_code == 404:
                return False

            # Other status codes - retry
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"Error checking DOI {doi}, attempt {attempt + 1}: {str(e)}")
                time.sleep(API_SLEEP_TIME * 2)
            else:
                print(f"Failed to check DOI {doi} after {retries} attempts")

    return False


def check_corpus_id_exists(
    corpus_type: str, corpus_id: str, api_key: Optional[str] = None, retries: int = 3
) -> bool:
    """
    Check if a corpus ID exists in Semantic Scholar's database.

    Args:
        corpus_type: Type of corpus ID (arxiv, pubmed, etc.)
        corpus_id: Corpus identifier
        api_key: Optional API key
        retries: Number of retries for failed requests

    Returns:
        True if corpus ID exists in Semantic Scholar, False otherwise
    """
    headers = configure_api_key(api_key)

    prefix = CORPUS_ID_PREFIXES.get(corpus_type.lower(), "")
    if not prefix:
        print(f"Unknown corpus type: {corpus_type}")
        return False

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{prefix}{corpus_id}",
                headers=headers,
                params={"fields": "paperId"},
            )

            # 200 status means the corpus ID exists
            if response.status_code == 200:
                return True

            # 404 means not found
            if response.status_code == 404:
                return False

            # Other status codes - retry
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error checking {corpus_type} ID {corpus_id}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)
            else:
                print(
                    f"Failed to check {corpus_type} ID {corpus_id} after {retries} attempts"
                )

    return False


def get_paper_id_from_doi(
    doi: str, api_key: Optional[str] = None, retries: int = 3
) -> Optional[str]:
    """
    Convert a DOI to a Semantic Scholar Paper ID.

    Args:
        doi: Digital Object Identifier
        api_key: Optional API key
        retries: Number of retries for failed requests

    Returns:
        Semantic Scholar Paper ID or None if not found
    """
    headers = configure_api_key(api_key)

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/DOI:{doi}",
                headers=headers,
                params={"fields": "paperId"},
            )

            # Check for successful response
            response.raise_for_status()
            data = response.json()
            return data.get("paperId")

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error fetching paper ID for DOI {doi}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)  # Longer sleep on failure
            else:
                print(
                    f"Failed to fetch paper ID for DOI {doi} after {retries} attempts"
                )

    return None


def get_paper_id_from_corpus_id(
    corpus_type: str, corpus_id: str, api_key: Optional[str] = None, retries: int = 3
) -> Optional[str]:
    """
    Convert a corpus ID to a Semantic Scholar Paper ID.

    Args:
        corpus_type: Type of corpus ID (arxiv, pubmed, etc.)
        corpus_id: Corpus identifier
        api_key: Optional API key
        retries: Number of retries for failed requests

    Returns:
        Semantic Scholar Paper ID or None if not found
    """
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

            # Check for successful response
            response.raise_for_status()
            data = response.json()
            return data.get("paperId")

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error fetching paper ID for {corpus_type} ID {corpus_id}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)  # Longer sleep on failure
            else:
                print(
                    f"Failed to fetch paper ID for {corpus_type} ID {corpus_id} after {retries} attempts"
                )

    return None


def fetch_paper_citations(
    paper_id: str,
    fields: List[str],
    api_key: Optional[str] = None,
    max_results: int = 1000,
    retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetch papers that cite the given paper.

    Args:
        paper_id: Semantic Scholar Paper ID
        fields: Fields to retrieve
        api_key: Optional API key
        max_results: Maximum number of citations to retrieve
        retries: Number of retry attempts

    Returns:
        List of citation papers
    """
    headers = configure_api_key(api_key)
    all_citations: List[Dict[str, Any]] = []

    # Build parameters
    params = {
        "fields": ",".join(fields),
        "limit": 100,  # Max per page
    }

    # For pagination
    offset = 0

    while len(all_citations) < max_results:
        params["offset"] = offset

        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{SEMANTIC_SCHOLAR_API_URL}{PAPER_CITATIONS_ENDPOINT}/{paper_id}/citations",
                    params=params,
                    headers=headers,
                )

                # Check for successful response
                response.raise_for_status()
                data = response.json()

                if "data" in data and isinstance(data["data"], list):
                    batch = data["data"]
                    if not batch:  # No more citations
                        return all_citations

                    # Extract the cited paper from each citation
                    papers = [
                        item.get("citingPaper", {})
                        for item in batch
                        if "citingPaper" in item
                    ]

                    # Only take what we need if this batch would exceed max_results
                    remaining = max_results - len(all_citations)
                    papers = papers[:remaining]
                    all_citations.extend(papers)

                    print(
                        f"Retrieved {len(papers)} citations, total: {len(all_citations)}/{max_results}"
                    )

                    # Check if we need more results
                    if (
                        len(all_citations) < max_results
                        and len(batch) == params["limit"]
                    ):
                        offset += params["limit"]  # type: ignore
                        # Sleep to avoid hitting rate limits
                        time.sleep(API_SLEEP_TIME)
                        break  # Success, continue to next page
                    else:
                        return all_citations  # No more results or we have enough
                else:
                    print("Unexpected response format")
                    return all_citations

            except requests.exceptions.RequestException as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(API_SLEEP_TIME * 2)
                else:
                    print(f"Failed after {retries} attempts")
                    return all_citations  # Return what we have

    return all_citations


def filter_papers_by_keywords(
    papers: List[Dict[str, Any]], keywords: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter papers based on keywords appearing in title or abstract.

    Args:
        papers: List of paper dictionaries
        keywords: List of keywords to match

    Returns:
        Filtered list of papers
    """
    filtered_papers = []

    # Convert keywords to lowercase for case-insensitive matching
    keywords_lower = [kw.lower() for kw in keywords if kw is not None]

    for paper in papers:
        if paper is None:
            continue

        # Handle possible None values by using empty strings as fallback
        title = (paper.get("title", "") or "").lower()
        abstract = (paper.get("abstract", "") or "").lower()

        # Check if any keyword appears in title or abstract
        if any(kw in title or kw in abstract for kw in keywords_lower):
            filtered_papers.append(paper)

    return filtered_papers


def extract_article_metadata_with_dataset(
    papers: List[Dict[str, Any]], dataset_name: str
) -> List[Dict[str, Any]]:
    """
    Extract relevant metadata from Semantic Scholar paper objects and add dataset name.

    Args:
        papers: List of paper dictionaries from Semantic Scholar API
        dataset_name: Name of the dataset that this paper cites

    Returns:
        List of metadata dictionaries
    """
    metadata_list = []

    for paper in papers:
        if not paper:
            continue

        # Extract basic information
        metadata = {
            "paper_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
            "citation_count": paper.get("citationCount", 0),
            "is_open_access": paper.get("isOpenAccess", False),
            "cited_dataset": dataset_name,  # Add the dataset name
        }

        # Extract authors (as a comma-separated string)
        authors = paper.get("authors", [])
        author_names = [author.get("name", "") for author in authors if author]
        metadata["authors"] = ", ".join(author_names)

        # Extract publication venue
        venue = paper.get("venue", "")
        metadata["venue"] = venue

        # Extract fields of study
        fields_of_study = paper.get("fieldsOfStudy", [])
        metadata["fields_of_study"] = (
            ", ".join(fields_of_study) if fields_of_study else ""
        )

        # Extract external IDs
        external_ids = paper.get("externalIds", {})
        metadata["doi"] = external_ids.get("DOI", "")
        metadata["arxiv_id"] = external_ids.get("ArXiv", "")
        metadata["pubmed_id"] = external_ids.get("PubMed", "")

        # Extract publication types
        pub_types = paper.get("publicationTypes", [])
        metadata["publication_types"] = ", ".join(pub_types) if pub_types else ""

        # Add to results
        metadata_list.append(metadata)

    return metadata_list


def process_citation_papers(config_file: str) -> List[Dict[str, Any]]:
    """
    Process a list of citation papers from a YAML configuration file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        List of filtered citation papers
    """
    try:
        # Read and parse the YAML file
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Check if citation papers are defined
        if "citation_papers" not in config or not config["citation_papers"]:
            print("No citation papers defined in the configuration file.")
            return []

        # Get the list of paper DOIs or IDs
        paper_identifiers = config["citation_papers"]

        # Get AI terms for filtering
        ai_terms = []
        if "keyword_sets" in config:
            # Find the AI terms set (usually the second set)
            for keyword_set in config["keyword_sets"]:
                if any(
                    kw in keyword_set
                    for kw in [
                        "deep learning",
                        "machine learning",
                        "artificial intelligence",
                    ]
                ):
                    ai_terms = keyword_set
                    break

        if not ai_terms:
            print("AI terms not found in keyword sets.")
            return []

        # Extract other parameters
        max_results = config.get("max_results", 500)
        output_prefix = config.get("output_prefix", "data/citation_results")
        api_key = config.get("semantic_scholar_api_key", None)

        # Fields to request from the API
        fields = [
            "paperId",
            "url",
            "title",
            "abstract",
            "venue",
            "year",
            "publicationDate",
            "authors",
            "externalIds",
            "fieldsOfStudy",
            "isOpenAccess",
            "citationCount",
            "publicationTypes",
            "journal",
        ]

        all_citations = []
        filtered_citations = []
        processed_papers = []
        failed_papers = []
        paper_id_to_dataset = {}  # Map paper IDs to their dataset names
        start_time = time.time()

        # Process each paper identifier
        for identifier in paper_identifiers:
            paper_id = identifier

            # Get dataset name for this identifier
            dataset_name = DATASET_MAPPING.get(identifier, "Unknown")

            # Handle different identifier types
            if identifier.startswith("10."):  # DOI
                print(f"Checking if DOI {identifier} exists in Semantic Scholar...")
                if not check_doi_exists(identifier, api_key):
                    print(f"DOI not found in Semantic Scholar: {identifier}")
                    failed_papers.append(
                        {
                            "identifier": identifier,
                            "reason": "DOI not found in Semantic Scholar",
                        }
                    )
                    continue

                print(f"Converting DOI {identifier} to Semantic Scholar paper ID...")
                paper_id = get_paper_id_from_doi(identifier, api_key)
                if not paper_id:
                    print(f"Could not get paper ID for DOI: {identifier}")
                    failed_papers.append(
                        {"identifier": identifier, "reason": "Could not get paper ID"}
                    )
                    continue

            elif ":" in identifier:  # Likely a corpus ID with type prefix
                parts = identifier.split(":", 1)
                if len(parts) != 2:
                    print(f"Invalid corpus ID format: {identifier}")
                    failed_papers.append(
                        {"identifier": identifier, "reason": "Invalid corpus ID format"}
                    )
                    continue

                corpus_type, corpus_id = parts
                print(
                    f"Checking if {corpus_type} ID {corpus_id} exists in Semantic Scholar..."
                )
                if not check_corpus_id_exists(corpus_type, corpus_id, api_key):
                    print(
                        f"{corpus_type} ID not found in Semantic Scholar: {corpus_id}"
                    )
                    failed_papers.append(
                        {
                            "identifier": identifier,
                            "reason": f"{corpus_type} ID not found in Semantic Scholar",
                        }
                    )
                    continue

                print(
                    f"Converting {corpus_type} ID {corpus_id} to Semantic Scholar paper ID..."
                )
                paper_id = get_paper_id_from_corpus_id(corpus_type, corpus_id, api_key)
                if not paper_id:
                    print(f"Could not get paper ID for {corpus_type} ID: {corpus_id}")
                    failed_papers.append(
                        {
                            "identifier": identifier,
                            "reason": f"Could not get paper ID for {corpus_type} ID",
                        }
                    )
                    continue

            # Assume it's already a Semantic Scholar paper ID if it doesn't match the above patterns

            print(f"Fetching citations for paper {paper_id}...")

            # Fetch citations for this paper
            citations = fetch_paper_citations(
                paper_id=paper_id,
                fields=fields,
                api_key=api_key,
                max_results=max_results,
            )

            if not citations:
                print(f"No citations found for paper {paper_id}.")
                failed_papers.append(
                    {"identifier": identifier, "reason": "No citations found"}
                )
                continue

            print(f"Found {len(citations)} citations for paper {paper_id}.")
            processed_papers.append(
                {
                    "identifier": identifier,
                    "paper_id": paper_id,
                    "dataset": dataset_name,
                    "citations_found": len(citations),
                }
            )
            all_citations.extend(citations)

            # Filter citations by AI terms
            paper_filtered_citations = filter_papers_by_keywords(citations, ai_terms)
            print(f"After filtering by keywords, {len(paper_filtered_citations)} citations remain out of {len(citations)} original citations.")

            # Store the dataset name for each paper ID
            for citation in paper_filtered_citations:
                citation_id = citation.get("paperId")
                if citation_id:
                    # Track which dataset this citation is linked to
                    if citation_id not in paper_id_to_dataset:
                        paper_id_to_dataset[citation_id] = [dataset_name]
                    else:
                        paper_id_to_dataset[citation_id].append(dataset_name)

            # Add these filtered citations
            filtered_citations.extend(paper_filtered_citations)

        end_time = time.time()

        if not filtered_citations:
            print("No citations match the filtering criteria.")
            return []

        # Remove duplicates by paper ID and combine dataset references
        unique_papers = {}
        for paper in filtered_citations:
            paper_id = paper.get("paperId")
            if paper_id and paper_id not in unique_papers:
                # Create a copy with added dataset field
                paper_copy = paper.copy()
                # Use the dataset mapping from our tracking dict
                if paper_id in paper_id_to_dataset:
                    datasets = paper_id_to_dataset[paper_id]
                    # Sort and deduplicate
                    datasets = sorted(list(set(datasets)))
                    paper_copy["cited_datasets"] = datasets
                else:
                    paper_copy["cited_datasets"] = ["Unknown"]

                unique_papers[paper_id] = paper_copy

        unique_filtered_citations = list(unique_papers.values())
        print(
            f"Total unique filtered citations across all papers: {len(unique_filtered_citations)}"
        )

        # Extract metadata with dataset information
        all_metadata = []
        for paper in unique_filtered_citations:
            # Extract basic metadata
            metadata = {
                "paper_id": paper.get("paperId", ""),
                "cited_dataset": json.dumps(paper.get("cited_datasets", ["Unknown"])),
                "title": paper.get("title", "") or "",
                "abstract": paper.get("abstract", "") or "",
                "year": paper.get("year"),
                "url": paper.get("url", ""),
                "citation_count": paper.get("citationCount", 0),
                "is_open_access": paper.get("isOpenAccess", False),
                "venue": paper.get("venue", ""),
                "fields_of_study": paper.get("fieldsOfStudy", []),
                "publication_types": paper.get("publicationTypes", []),
                "journal": paper.get("journal", ""),
                "authors": paper.get("authors", []),
                "external_ids": paper.get("externalIds", {}),
            }

            # Extract authors (as a comma-separated string)
            authors = paper.get("authors", [])
            author_names = [author.get("name", "") for author in authors if author]
            metadata["authors"] = ", ".join(author_names)

            # Extract publication venue
            venue = paper.get("venue", "")
            metadata["venue"] = venue

            # Extract fields of study
            fields_of_study = paper.get("fieldsOfStudy", [])
            metadata["fields_of_study"] = (
                ", ".join(fields_of_study) if fields_of_study else ""
            )

            # Extract external IDs
            external_ids = paper.get("externalIds", {})
            metadata["doi"] = external_ids.get("DOI", "")
            metadata["arxiv_id"] = external_ids.get("ArXiv", "")
            metadata["pubmed_id"] = external_ids.get("PubMed", "")

            # Extract publication types
            pub_types = paper.get("publicationTypes", [])
            metadata["publication_types"] = ", ".join(pub_types) if pub_types else ""

            # Add to results
            all_metadata.append(metadata)

        # Save results if output prefix is specified
        if output_prefix:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_prefix)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save results to CSV
            csv_file = f"{output_prefix}_citations.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                if all_metadata:
                    writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
                    writer.writeheader()
                    writer.writerows(all_metadata)

            # Save metadata about the query
            txt_file = f"{output_prefix}_citations_metadata.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"Semantic Scholar Citation Query Metadata\n")
                f.write(f"====================================\n\n")
                f.write(
                    f"Query executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                f.write(f"Papers queried for citations:\n")
                for paper in processed_papers:
                    f.write(
                        f"  - {paper['identifier']} (Dataset: {paper['dataset']}, Citations: {paper['citations_found']})\n"
                    )

                if failed_papers:
                    f.write(f"\nFailed papers:\n")
                    for paper in failed_papers:
                        f.write(
                            f"  - {paper['identifier']} (Reason: {paper['reason']})\n"
                        )

                f.write(f"\nFilter terms (AI keywords):\n")
                for term in ai_terms:
                    f.write(f"  - {term}\n")

                # Find papers that cite multiple datasets
                multi_dataset_papers = 0
                for paper_id, datasets in paper_id_to_dataset.items():
                    if len(datasets) > 1:
                        multi_dataset_papers += 1

                f.write(f"\nResults:\n")
                f.write(f"  Total citations found: {len(all_citations)}\n")
                f.write(
                    f"  Unique citations after filtering: {len(unique_filtered_citations)}\n"
                )
                f.write(f"  Papers citing multiple datasets: {multi_dataset_papers}\n")
                f.write(f"  Execution time: {end_time - start_time:.2f} seconds\n")

            print(f"Results saved to {csv_file}")
            print(f"Query metadata saved to {txt_file}")

            # Process the CSV file to clean up the citations format
            print("Post-processing the CSV file...")
            try:
                df = pd.read_csv(csv_file)
                # Convert JSON string to proper list format
                df["cited_dataset"] = df["cited_dataset"].apply(
                    lambda x: ", ".join(json.loads(x)) if isinstance(x, str) else x
                )
                # Save back to CSV
                df.to_csv(csv_file, index=False)
                print("CSV post-processing complete!")

                # Add this new line to call the enhancement function:
                filter_citations.enhance_dataset_references(csv_file)
                # After applying additional filters (date, citation count, duplicates)
                try:
                    # Load the final processed CSV to get the count of papers after all filtering
                    final_df = pd.read_csv(csv_file)
                    print("\nFinal filtering results:")
                    print(f"Original papers after keyword filtering: {len(df)}")
                    print(f"Papers remaining after date/citation/duplicate filtering: {len(final_df)}")
                    print(f"Removed papers: {len(df) - len(final_df)}")
                except Exception as e:
                    print(f"Error reading final filtered results: {str(e)}")
            except Exception as e:
                print(f"Error during CSV post-processing: {str(e)}")
        return all_metadata

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {str(e)}")
        return []
    except Exception as e:
        print(f"Error during citation processing: {str(e)}")
        return []


if __name__ == "__main__":
    # Process citation papers using YAML configuration
    results = process_citation_papers("./scripts/fetch/parameters.yaml")
    print(f"Retrieved {len(results)} filtered citations")
