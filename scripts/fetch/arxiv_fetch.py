"""
arXiv article search module for retrieving article metadata
based on keywords and date ranges using the arXiv API.
"""

import arxiv  # type: ignore
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import csv
import yaml  # type: ignore
import os
import re


def parse_date_for_arxiv(date_str: str) -> str:
    """
    Convert date from YYYY/MM/DD format to YYYYMMDDHHMMSS format for arXiv API.

    Args:
        date_str: Date in YYYY/MM/DD format

    Returns:
        Date in YYYYMMDDHHMMSS format
    """
    # Parse the input date
    parts = date_str.split("/")
    if len(parts) != 3:
        raise ValueError("Date must be in YYYY/MM/DD format")

    year, month, day = parts
    # arXiv requires format YYYYMMDDhhmmss
    return f"{year}{month.zfill(2)}{day.zfill(2)}000000"


def build_arxiv_query(
    keywords: Union[List[str], List[List[str]]],
    date_range: Tuple[str, str],
    categories: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> str:
    """
    Build a query string for the arXiv API.

    Args:
        keywords: Either a flat list of keywords, or a list of keyword sets (list of lists)
        date_range: Tuple containing start and end dates in format 'YYYY/MM/DD'
        categories: Optional list of arXiv categories to search in
        exclude_keywords: Optional list of keywords to exclude

    Returns:
        A formatted query string for the arXiv API
    """
    # Check if we have keyword sets (list of lists) or a flat keyword list
    is_keyword_sets = any(isinstance(item, list) for item in keywords)

    if is_keyword_sets:
        # Process each keyword set separately
        keyword_set_queries = []
        for keyword_set in keywords:
            set_parts = []
            for kw in keyword_set:
                # Search in title and abstract
                set_parts.append(f'(ti:"{kw}" OR abs:"{kw}")')
            # Join keywords within a set with OR
            keyword_set_queries.append(f"({' OR '.join(set_parts)})")
        # Join different sets with AND
        keyword_query = " AND ".join(keyword_set_queries)
    else:
        # Original behavior for flat keyword list
        keyword_parts = []
        for kw in keywords:  # type: ignore
            keyword_parts.append(f'(ti:"{kw}" OR abs:"{kw}")')
        # Join with OR operator to find articles with at least one keyword match
        keyword_query = " OR ".join(keyword_parts)

    # Build the query parts
    query_parts = [f"({keyword_query})"]

    # Add date range
    # arXiv uses submittedDate for range searches
    start_date, end_date = date_range
    date_query = f"submittedDate:[{parse_date_for_arxiv(start_date)} TO {parse_date_for_arxiv(end_date)}]"
    query_parts.append(date_query)

    # Add categories if specified
    if categories and len(categories) > 0:
        category_parts = [f"cat:{cat}" for cat in categories]
        query_parts.append(f"({' OR '.join(category_parts)})")

    # Add exclude keywords if specified
    if exclude_keywords and len(exclude_keywords) > 0:
        for kw in exclude_keywords:
            query_parts.append(f'NOT (ti:"{kw}" OR abs:"{kw}")')

    # Combine all query parts
    full_query = " AND ".join(query_parts)
    print(f"arXiv Query: {full_query}")
    return full_query


def search_arxiv(
    query: str,
    max_results: int = 100,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> List[arxiv.Result]:
    """
    Search for articles on arXiv.

    Args:
        query: Query string for search
        max_results: Maximum number of results to return
        sort_by: Sort criterion (default: Relevance)
        sort_order: Sort order (default: Descending)

    Returns:
        List of arxiv.Result objects
    """
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    results = list(client.results(search))
    return results


def extract_article_metadata(articles: List[arxiv.Result]) -> List[Dict[str, Any]]:
    """
    Extract metadata from arXiv article results.

    Args:
        articles: List of arxiv.Result objects

    Returns:
        List of dictionaries containing article metadata
    """
    metadata_list = []

    for article in articles:
        # Default empty values for fields that may not be available
        authors = [author.name for author in article.authors]

        # Convert published/updated to YYYY/MM/DD format
        pub_date = article.published.strftime("%Y/%m/%d")
        update_date = article.updated.strftime("%Y/%m/%d") if article.updated else ""

        # Extract categories
        categories = [cat for cat in article.categories]

        # Build metadata with fields matching NCBI output where possible
        metadata = {
            "pmid": article.get_short_id(),  # Use arXiv ID as PMID equivalent
            "title": article.title,
            "journal": "arXiv",  # All are from arXiv
            "publication_date": pub_date,
            "publication_year": article.published.year,
            "authors": authors,
            "abstract": article.summary,
            "doi": article.doi if article.doi else "",
            "keywords": categories,  # Use categories as keywords
            "mesh_terms": [],  # Not available in arXiv
            "journal_issn": "",  # Not available in arXiv
            "article_language": "eng",  # Assume English for arXiv
            "pubmed_status": "",  # Not applicable
            "publication_status": "preprint",
            "pubmed_pubdate": "",  # Not applicable
            "article_type": "Preprint",
            "funding": [],  # Funding info not consistently available
            "country": "",  # Not typically available in arXiv
            "citation_count": 0,  # Not available via API
            "full_journal_name": "arXiv.org",
            "pagination": "",  # Not applicable to preprints
            "volume": "",  # Not applicable to preprints
            "issue": "",  # Not applicable to preprints
            "url": article.entry_id,  # Direct link to the article
            "arxiv_id": article.entry_id.split("/")[-1],  # The arXiv ID
            "primary_category": article.primary_category,
            "comment": article.comment if hasattr(article, "comment") else "",
            "journal_ref": (
                article.journal_ref if hasattr(article, "journal_ref") else ""
            ),
            "last_updated": update_date,
            "pdf_url": article.pdf_url,
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

    fieldnames = sorted(list(fieldnames))  # type: ignore

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
    keywords: List[str],
    date_range: Tuple[str, str],
    max_results: int,
    result_count: int,
    start_time: float,
    end_time: float,
    output_file: str,
) -> None:
    """
    Save metadata about the query to a text file.

    Args:
        query: The search query string used
        keywords: List of keywords searched for
        date_range: Tuple containing start and end dates
        max_results: Maximum number of results requested
        result_count: Actual number of results retrieved
        start_time: Timestamp when search started
        end_time: Timestamp when search completed
        output_file: Path to the output text file
    """
    elapsed_time = end_time - start_time

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"arXiv Query Metadata\n")
        f.write(f"==================\n\n")
        f.write(f"Query executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Search Parameters:\n")
        f.write(
            f"  Keywords: {', '.join(keywords if isinstance(keywords[0], str) else [k for sublist in keywords for k in sublist])}\n"
        )
        f.write(f"  Date range: {date_range[0]} to {date_range[1]}\n")
        f.write(f"  Max results requested: {max_results}\n\n")

        f.write(f"Results:\n")
        f.write(f"  Total articles found: {result_count}\n")
        f.write(f"  Execution time: {elapsed_time:.2f} seconds\n\n")

        f.write(f"Full Query:\n")
        f.write(f"  {query}\n")

    print(f"Query metadata saved to {output_file}")


def search_from_yaml(config_file: str) -> List[Dict[str, Any]]:
    """
    Run an arXiv search using parameters from a YAML configuration file.

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
            # Using keyword sets
            keywords = config["keyword_sets"]
        else:
            # Using flat keyword list
            keywords = config["keywords"]

        date_range = (
            config["date_range"]["start_date"],
            config["date_range"]["end_date"],
        )

        # Extract optional parameters with defaults
        max_results = config.get("max_results", 100)
        output_prefix = config.get("output_prefix", None)
        categories = config.get("arxiv_categories", None)
        exclude_keywords = config.get("exclude_keywords", None)

        # Build the query
        query = build_arxiv_query(
            keywords=keywords,
            date_range=date_range,
            categories=categories,
            exclude_keywords=exclude_keywords,
        )

        start_time = time.time()

        # Search for articles
        articles = search_arxiv(query, max_results=max_results)

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
            csv_file = f"{output_prefix}_arxiv.csv"
            save_results_to_csv(metadata_list, csv_file)

            # Save query metadata to TXT
            txt_file = f"{output_prefix}_arxiv_metadata.txt"
            save_query_metadata(
                query=query,
                keywords=(
                    keywords
                    if not isinstance(keywords[0], list)
                    else [k for sublist in keywords for k in sublist]
                ),
                date_range=date_range,
                max_results=max_results,
                result_count=len(metadata_list),
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
        print(f"Error during arXiv search: {str(e)}")
        return []


if __name__ == "__main__":
    # Example using YAML configuration
    results = search_from_yaml("./scripts/fetch/parameters.yaml")
    print(f"Retrieved {len(results)} articles from arXiv")
