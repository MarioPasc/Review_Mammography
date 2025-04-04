"""
Semantic Scholar article search module for retrieving article metadata
from academic journals, particularly focusing on engineering and computer science.
Uses the Semantic Scholar Bulk Search API for more efficient retrieval.
"""

import time
import csv
import yaml  # type: ignore
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
import requests  # type: ignore


# Constants for API
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_BULK_SEARCH_ENDPOINT = "/paper/search/bulk"
PAPER_DETAILS_ENDPOINT = "/paper"

# Sleep time between API calls to avoid rate limits
API_SLEEP_TIME = 1.0  # seconds


def configure_api_key(api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Configure API headers with an optional API key.

    Args:
        api_key: Optional Semantic Scholar API key

    Returns:
        Headers dictionary for API requests
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def build_semantic_query(
    keywords: Union[List[str], List[List[str]]],
    exclude_keywords: Optional[List[str]] = None,
) -> str:
    """
    Build a query string for the Semantic Scholar API using advanced query syntax.

    Args:
        keywords: Either a flat list of keywords, or a list of keyword sets
        exclude_keywords: Optional list of keywords to exclude

    Returns:
        A formatted query string for the Semantic Scholar API
    """
    # Check if we have keyword sets (list of lists) or a flat keyword list
    is_keyword_sets = any(isinstance(item, list) for item in keywords)

    if is_keyword_sets:
        # Process each keyword set separately
        keyword_set_queries = []
        for keyword_set in keywords:
            # Join keywords within a set with OR
            keyword_phrases = [f'"{kw}"' for kw in keyword_set]
            keyword_set_queries.append(f"({' | '.join(keyword_phrases)})")

        # Join different sets with AND
        query = f" + ".join(keyword_set_queries)
    else:
        # Simple list of keywords joined with OR
        keyword_phrases = [f'"{kw}"' for kw in keywords]  # type: ignore
        query = f"{' | '.join(keyword_phrases)}"

    # Add exclusions if provided
    if exclude_keywords and len(exclude_keywords) > 0:
        exclusion_phrases = [f'-"{kw}"' for kw in exclude_keywords]
        query = f"({query}) {' '.join(exclusion_phrases)}"

    return query


def search_semantic_scholar_bulk(
    query: str,
    fields: List[str],
    year_range: Optional[Tuple[int, int]] = None,
    publication_types: Optional[List[str]] = None,
    min_citation_count: Optional[int] = None,
    fields_of_study: Optional[List[str]] = None,
    open_access_only: bool = False,
    sort: str = "citationCount:desc",
    api_key: Optional[str] = None,
    max_results: int = 100,
    retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Search for articles using the Semantic Scholar Bulk API.

    Args:
        query: Query string using Semantic Scholar's query syntax
        fields: Fields to return
        year_range: Optional tuple of (start_year, end_year)
        publication_types: Optional list of publication types
        min_citation_count: Optional minimum citation count
        fields_of_study: Optional list of fields of study
        open_access_only: If True, only return papers with open access PDFs
        sort: Sort order (field:order)
        api_key: Optional API key
        max_results: Maximum number of results to return
        retries: Number of retries for failed requests

    Returns:
        List of article data dictionaries
    """
    headers = configure_api_key(api_key)
    all_papers: list = []

    # Build base parameters
    params = {
        "query": query,
        "fields": ",".join(fields),
        "sort": sort,
        "limit": min(1000, max_results),  # API allows up to 1000 per request
    }

    # Add optional filters
    if year_range:
        params["year"] = f"{year_range[0]}:{year_range[1]}"

    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)

    if min_citation_count is not None:
        params["minCitationCount"] = str(min_citation_count)

    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    if open_access_only:
        params["openAccessPdf"] = ""

    # Token for pagination
    next_token = None

    while len(all_papers) < max_results:
        # Add token if we're paginating
        if next_token:
            params["token"] = next_token

        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{SEMANTIC_SCHOLAR_API_URL}{PAPER_BULK_SEARCH_ENDPOINT}",
                    params=params,
                    headers=headers,
                )

                # Check for successful response
                response.raise_for_status()
                data = response.json()

                if "data" in data and isinstance(data["data"], list):
                    batch = data["data"]
                    # Only take what we need if this batch would exceed max_results
                    remaining = max_results - len(all_papers)
                    batch = batch[:remaining]
                    all_papers.extend(batch)

                    print(
                        f"Retrieved {len(batch)} papers, total: {len(all_papers)}/{max_results}"
                    )

                    # Check if there's more to fetch and we still need more results
                    if (
                        "next" in data
                        and data["next"]
                        and len(all_papers) < max_results
                    ):
                        next_token = data["next"]
                        # Sleep to avoid hitting rate limits
                        time.sleep(API_SLEEP_TIME)
                        break  # Success, continue to next page
                    else:
                        return all_papers  # No more results or we have enough
                else:
                    print("Unexpected response format")
                    if attempt < retries - 1:
                        time.sleep(API_SLEEP_TIME * 2)
                    else:
                        return all_papers  # Return what we have

            except requests.exceptions.RequestException as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(API_SLEEP_TIME * 2)
                else:
                    print(f"Failed after {retries} attempts")
                    return all_papers  # Return what we have

    return all_papers


def fetch_paper_details(
    paper_id: str, fields: List[str], api_key: Optional[str] = None, retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information for a specific paper.

    Args:
        paper_id: Semantic Scholar Paper ID
        fields: Fields to retrieve
        api_key: Optional API key
        retries: Number of retry attempts

    Returns:
        Paper details dictionary or None if not found
    """
    headers = configure_api_key(api_key)
    params = {"fields": ",".join(fields)}

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}{PAPER_DETAILS_ENDPOINT}/{paper_id}",
                params=params,
                headers=headers,
            )

            # Check for successful response
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(
                    f"Error fetching paper {paper_id}, attempt {attempt + 1}: {str(e)}"
                )
                time.sleep(API_SLEEP_TIME * 2)  # Longer sleep on failure
            else:
                print(f"Failed to fetch paper {paper_id} after {retries} attempts")
                return None

    # Explicit return for code clarity
    return None


def extract_article_metadata(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract and normalize metadata from Semantic Scholar article results.

    Args:
        articles: List of article dictionaries from Semantic Scholar

    Returns:
        List of dictionaries with normalized metadata
    """
    metadata_list = []

    for article in articles:
        # Extract basic fields
        paper_id = article.get("paperId", "")

        # Extract authors
        authors = []
        if "authors" in article and article["authors"]:
            for author in article["authors"]:
                if isinstance(author, dict) and "name" in author:
                    authors.append(author["name"])

        # Extract publication date and year
        year = article.get("year")
        publication_year = str(year) if year else ""

        # Try to get a more specific date if available
        publication_date = ""
        if "publicationDate" in article and article["publicationDate"]:
            publication_date = article["publicationDate"]
        else:
            # Default to January 1st of the year if we only have the year
            publication_date = f"{year}/01/01" if year else ""

        # Extract journal/venue
        journal = ""
        full_journal_name = ""
        if "venue" in article and article["venue"]:
            journal = article["venue"]
            full_journal_name = article["venue"]

        # Citation count
        citation_count = article.get("citationCount", 0)

        # External IDs
        external_ids = article.get("externalIds", {})
        doi = external_ids.get("DOI", "")

        # Fields of study
        fields_of_study = article.get("fieldsOfStudy", [])

        # Publication types
        publication_types = article.get("publicationTypes", [])
        article_type = publication_types[0] if publication_types else ""

        # Open access status
        is_open_access = article.get("isOpenAccess", False)

        # URL
        article_url = article.get(
            "url", f"https://www.semanticscholar.org/paper/{paper_id}"
        )

        # Fields to match NCBI and arXiv output format
        metadata = {
            "pmid": paper_id,  # Use Semantic Scholar paperId
            "title": article.get("title", ""),
            "journal": journal,
            "publication_date": publication_date,
            "publication_year": publication_year,
            "authors": authors,
            "abstract": article.get("abstract", ""),
            "doi": doi,
            "keywords": fields_of_study,
            "mesh_terms": [],  # Not available in Semantic Scholar
            "journal_issn": "",  # Not consistently available
            "article_language": "eng",  # Language info not consistently available
            "pubmed_status": "",  # Not applicable
            "publication_status": article_type,
            "pubmed_pubdate": "",  # Not applicable
            "article_type": article_type,
            "funding": [],  # Not consistently available
            "country": "",  # Not consistently available
            "citation_count": citation_count,
            "full_journal_name": full_journal_name,
            "pagination": "",  # Not consistently available
            "volume": "",  # Not consistently available
            "issue": "",  # Not consistently available
            "url": article_url,
            "source": "Semantic Scholar",
            "open_access": is_open_access,
            "fields_of_study": fields_of_study,
            "publication_types": publication_types,
            "s2_paperId": paper_id,
        }

        metadata_list.append(metadata)

    return metadata_list


def save_results_to_csv(metadata_list: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save article metadata to a CSV file.

    Args:
        metadata_list: List of article metadata dictionaries
        output_file: Path to the output CSV file
    """
    if not metadata_list:
        print("No data to save to CSV.")
        return

    # Get all unique keys from all dictionaries to use as fieldnames
    fieldnames: set = set()
    for item in metadata_list:
        fieldnames.update(item.keys())

    sorted_fieldnames: list = sorted(list(fieldnames))

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames)
        writer.writeheader()

        for item in metadata_list:
            # Convert list items to strings
            for key, value in item.items():
                if isinstance(value, list):
                    item[key] = "; ".join(str(v) for v in value)
            writer.writerow(item)

    print(f"Results saved to {output_file}")


def save_query_metadata(
    query: str,
    date_range: Tuple[str, str],
    max_results: int,
    result_count: int,
    filters: Dict[str, Any],
    start_time: float,
    end_time: float,
    output_file: str,
) -> None:
    """
    Save metadata about the query to a text file.

    Args:
        query: The search query string used
        date_range: Tuple containing start and end dates
        max_results: Maximum number of results requested
        result_count: Actual number of results retrieved
        filters: Dictionary of filters applied
        start_time: Timestamp when search started
        end_time: Timestamp when search completed
        output_file: Path to the output text file
    """
    elapsed_time = end_time - start_time

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Semantic Scholar Query Metadata\n")
        f.write(f"=============================\n\n")
        f.write(f"Query executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Search Parameters:\n")
        f.write(f"  Query: {query}\n")
        f.write(f"  Date range: {date_range[0]} to {date_range[1]}\n")
        f.write(f"  Max results requested: {max_results}\n\n")

        f.write(f"Filters Applied:\n")
        for key, value in filters.items():
            if value:
                f.write(f"  {key}: {value}\n")

        f.write(f"\nResults:\n")
        f.write(f"  Total articles found: {result_count}\n")
        f.write(f"  Execution time: {elapsed_time:.2f} seconds\n")

    print(f"Query metadata saved to {output_file}")


def search_from_yaml(config_file: str) -> List[Dict[str, Any]]:
    """
    Run a Semantic Scholar search using parameters from a YAML configuration file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        List of dictionaries containing article metadata
    """
    try:
        # Read and parse the YAML file
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract required parameters
        if "keyword_sets" in config:
            keywords = config["keyword_sets"]
        else:
            keywords = config["keywords"]

        date_range = (
            config["date_range"]["start_date"],
            config["date_range"]["end_date"],
        )

        # Convert date strings to years for Semantic Scholar
        start_year = int(date_range[0].split("/")[0])
        end_year = int(date_range[1].split("/")[0])
        year_range = (start_year, end_year)

        # Extract optional parameters with defaults
        max_results = config.get("max_results", 100)
        output_prefix = config.get("output_prefix", None)
        exclude_keywords = config.get("exclude_keywords", None)
        api_key = config.get("semantic_scholar_api_key", None)

        # Semantic Scholar specific parameters
        publication_types = config.get("semantic_scholar_publication_types", None)
        fields_of_study = config.get("semantic_scholar_fields_of_study", None)
        min_citation_count = config.get("semantic_scholar_min_citation_count", None)
        open_access_only = config.get("semantic_scholar_open_access_only", False)
        sort = config.get("semantic_scholar_sort", "citationCount:desc")

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

        # Build the query string
        query = build_semantic_query(
            keywords=keywords, exclude_keywords=exclude_keywords
        )

        print(f"Semantic Scholar query: {query}")
        print(f"Year range: {year_range[0]}-{year_range[1]}")

        start_time = time.time()

        # Search for articles
        articles = search_semantic_scholar_bulk(
            query=query,
            fields=fields,
            year_range=year_range,
            publication_types=publication_types,
            min_citation_count=min_citation_count,
            fields_of_study=fields_of_study,
            open_access_only=open_access_only,
            sort=sort,
            api_key=api_key,
            max_results=max_results,
        )

        if not articles:
            print("No articles found matching the search criteria.")
            return []

        print(f"Found {len(articles)} articles. Processing metadata...")

        # Extract metadata
        metadata_list = extract_article_metadata(articles)

        end_time = time.time()

        # Save results if output prefix is specified
        if output_prefix:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_prefix)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save results to CSV
            csv_file = f"{output_prefix}_semantic.csv"
            save_results_to_csv(metadata_list, csv_file)

            # Save query metadata to TXT
            txt_file = f"{output_prefix}_semantic_metadata.txt"

            # Collect filters for metadata file
            filters = {
                "Publication Types": publication_types,
                "Fields of Study": fields_of_study,
                "Min Citation Count": min_citation_count,
                "Open Access Only": open_access_only,
                "Sort": sort,
            }

            save_query_metadata(
                query=query,
                date_range=date_range,
                max_results=max_results,
                result_count=len(metadata_list),
                filters=filters,
                start_time=start_time,
                end_time=end_time,
                output_file=txt_file,
            )

        return metadata_list

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {str(e)}")
        return []
    except Exception as e:
        print(f"Error during Semantic Scholar search: {str(e)}")
        return []


if __name__ == "__main__":
    # Example using YAML configuration
    results = search_from_yaml("./scripts/fetch/parameters.yaml")
    print(f"Retrieved {len(results)} articles from Semantic Scholar")
