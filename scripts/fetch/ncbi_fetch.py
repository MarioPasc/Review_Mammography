"""
NCBI article search module for retrieving peer-reviewed article metadata
based on keywords and date ranges using Bio.Entrez.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import time
import csv
import yaml  # type: ignore
import os
from Bio import Entrez  # type: ignore


def setup_entrez(email: str) -> None:
    """
    Configure the Entrez API with user's email.

    Args:
        email: A valid email address to identify the user to NCBI
    """
    Entrez.email = email


def search_articles(
    query: str,
    database: str = "pubmed",
    max_results: int = 100,
    retries: int = 3,
    sleep_seconds: int = 1,
) -> Dict[str, Any]:
    """
    Search for articles using the provided query.

    Args:
        query: The search query string
        database: The NCBI database to search (default: "pubmed")
        max_results: Maximum number of results to return
        retries: Number of times to retry on failure
        sleep_seconds: Number of seconds to sleep between API calls

    Returns:
        Dictionary containing search results
    """
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(
                db=database, term=query, retmax=max_results, sort="relevance"
            )
            results = Entrez.read(handle)
            handle.close()
            return results
        except Exception as e:
            if attempt < retries - 1:
                print(f"Search attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(sleep_seconds)
            else:
                raise e

    return {}


def fetch_article_details(
    id_list: List[str],
    database: str = "pubmed",
    batch_size: int = 20,
    retries: int = 3,
    sleep_seconds: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fetch detailed information for a list of article IDs.

    Args:
        id_list: List of article IDs (PMIDs) to fetch
        database: The NCBI database to fetch from (default: "pubmed")
        batch_size: Number of articles to fetch in each batch
        retries: Number of times to retry on failure
        sleep_seconds: Number of seconds to sleep between API calls

    Returns:
        List of dictionaries containing article details
    """
    articles = []

    # Process in batches to avoid overloading the API
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i : i + batch_size]

        for attempt in range(retries):
            try:
                handle = Entrez.efetch(
                    db=database, id=batch_ids, rettype="xml", retmode="text"
                )
                records = Entrez.read(handle)
                handle.close()

                # Add all articles from this batch
                if "PubmedArticle" in records:
                    articles.extend(records["PubmedArticle"])

                # Sleep to be nice to the NCBI server
                time.sleep(sleep_seconds)
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(
                        f"Fetch attempt {attempt + 1} for batch {i//batch_size + 1} failed: {str(e)}. Retrying..."
                    )
                    time.sleep(sleep_seconds * 2)  # Longer sleep on failure
                else:
                    print(
                        f"Failed to fetch batch {i//batch_size + 1} after {retries} attempts: {str(e)}"
                    )

    return articles


def extract_article_metadata(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all available metadata from article records.

    Args:
        articles: List of article records from Entrez.efetch

    Returns:
        List of dictionaries containing extracted metadata
    """
    metadata_list = []

    for article in articles:
        try:
            medline_citation = article["MedlineCitation"]
            article_data = medline_citation["Article"]
            pubmed_data = article.get("PubmedData", {})

            # Basic metadata fields
            metadata = {
                "pmid": medline_citation["PMID"],
                "title": article_data["ArticleTitle"],
                "journal": article_data["Journal"]["Title"],
                "publication_date": extract_publication_date(article_data),
                "publication_year": extract_publication_year(article_data),
                "authors": extract_authors(article_data),
                "abstract": extract_abstract(article_data),
                "doi": extract_doi(article),
                "keywords": extract_keywords(medline_citation),
                "mesh_terms": extract_mesh_terms(medline_citation),
                "journal_issn": extract_journal_issn(article_data),
                "article_language": extract_language(article_data),
                "pubmed_status": medline_citation.get("Status", ""),
                "publication_status": pubmed_data.get("PublicationStatus", ""),
                "pubmed_pubdate": extract_pubmed_pubdate(pubmed_data),
                "article_type": extract_article_type(article_data),
                "funding": extract_funding(article_data),
                "country": extract_country(medline_citation),
                "citation_count": extract_citation_count(pubmed_data),
                "full_journal_name": extract_full_journal_name(article_data),
                "pagination": extract_pagination(article_data),
                "volume": extract_volume(article_data),
                "issue": extract_issue(article_data),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{medline_citation['PMID']}/",
            }

            metadata_list.append(metadata)
        except KeyError as e:
            print(f"Error extracting metadata: {str(e)}")

    return metadata_list


def extract_publication_date(article_data: Dict[str, Any]) -> str:
    """Extract publication date from article data."""
    try:
        pub_date = article_data["Journal"]["JournalIssue"]["PubDate"]
        if "Year" in pub_date:
            year = pub_date["Year"]
            month = pub_date.get("Month", "01")
            day = pub_date.get("Day", "01")
            return f"{year}/{month}/{day}"
        return ""
    except KeyError:
        return ""


def extract_publication_year(article_data: Dict[str, Any]) -> str:
    """Extract publication year from article data."""
    try:
        pub_date = article_data["Journal"]["JournalIssue"]["PubDate"]
        if "Year" in pub_date:
            return pub_date["Year"]
        return ""
    except KeyError:
        return ""


def extract_authors(article_data: Dict[str, Any]) -> List[str]:
    """Extract author names from article data."""
    authors = []
    try:
        author_list = article_data.get("AuthorList", [])
        for author in author_list:
            if "LastName" in author and "ForeName" in author:
                authors.append(f"{author['LastName']} {author['ForeName']}")
            elif "LastName" in author:
                authors.append(f"{author['LastName']}")
            elif "CollectiveName" in author:
                authors.append(author["CollectiveName"])
    except Exception:
        pass
    return authors


def extract_abstract(article_data: Dict[str, Any]) -> str:
    """Extract abstract text from article data."""
    try:
        if "Abstract" in article_data:
            abstract_texts = article_data["Abstract"]["AbstractText"]
            if isinstance(abstract_texts, list):
                return " ".join([str(text) for text in abstract_texts])
            return str(abstract_texts)
    except Exception:
        pass
    return ""


def extract_doi(article: Dict[str, Any]) -> str:
    """Extract DOI from article data."""
    try:
        for id_obj in article["PubmedData"]["ArticleIdList"]:
            if id_obj.attributes["IdType"] == "doi":
                return str(id_obj)
    except Exception:
        pass
    return ""


def extract_keywords(medline_citation: Dict[str, Any]) -> List[str]:
    """Extract keywords from article data."""
    keywords = []
    try:
        if "KeywordList" in medline_citation:
            for keyword in medline_citation["KeywordList"][0]:
                keywords.append(str(keyword))
    except Exception:
        pass
    return keywords


def extract_mesh_terms(medline_citation: Dict[str, Any]) -> List[str]:
    """Extract MeSH terms from article data."""
    terms = []
    try:
        if "MeshHeadingList" in medline_citation:
            for heading in medline_citation["MeshHeadingList"]:
                if "DescriptorName" in heading:
                    terms.append(str(heading["DescriptorName"]))
    except Exception:
        pass
    return terms


def extract_journal_issn(article_data: Dict[str, Any]) -> str:
    """Extract journal ISSN from article data."""
    try:
        if "ISSN" in article_data["Journal"]:
            return article_data["Journal"]["ISSN"]
    except Exception:
        pass
    return ""


def extract_language(article_data: Dict[str, Any]) -> str:
    """Extract article language from article data."""
    try:
        if "Language" in article_data:
            return article_data["Language"][0]
    except Exception:
        pass
    return ""


def extract_pubmed_pubdate(pubmed_data: Dict[str, Any]) -> str:
    """Extract PubMed publication date from article data."""
    try:
        if "History" in pubmed_data:
            for date_item in pubmed_data["History"]:
                if date_item.attributes["PubStatus"] == "pubmed":
                    return f"{date_item['Year']}/{date_item.get('Month', '01')}/{date_item.get('Day', '01')}"
    except Exception:
        pass
    return ""


def extract_article_type(article_data: Dict[str, Any]) -> str:
    """Extract article type from article data."""
    try:
        if "PublicationTypeList" in article_data:
            return str(article_data["PublicationTypeList"][0])
    except Exception:
        pass
    return ""


def extract_funding(article_data: Dict[str, Any]) -> List[str]:
    """Extract funding information from article data."""
    funding = []
    try:
        if "GrantList" in article_data:
            for grant in article_data["GrantList"]:
                if "Agency" in grant:
                    funding.append(
                        f"{grant['Agency']} ({grant.get('GrantID', 'No ID')})"
                    )
    except Exception:
        pass
    return funding


def extract_country(medline_citation: Dict[str, Any]) -> str:
    """Extract country information from article data."""
    try:
        if "MedlineJournalInfo" in medline_citation:
            return medline_citation["MedlineJournalInfo"].get("Country", "")
    except Exception:
        pass
    return ""


def extract_citation_count(pubmed_data: Dict[str, Any]) -> int:
    """Extract citation count from article data."""
    try:
        if "ReferenceList" in pubmed_data:
            for ref_list in pubmed_data["ReferenceList"]:
                if "Reference" in ref_list:
                    return len(ref_list["Reference"])
    except Exception:
        pass
    return 0


def extract_full_journal_name(article_data: Dict[str, Any]) -> str:
    """Extract full journal name from article data."""
    try:
        return article_data["Journal"]["Title"]
    except Exception:
        pass
    return ""


def extract_pagination(article_data: Dict[str, Any]) -> str:
    """Extract pagination information from article data."""
    try:
        if "Pagination" in article_data:
            return article_data["Pagination"].get("MedlinePgn", "")
    except Exception:
        pass
    return ""


def extract_volume(article_data: Dict[str, Any]) -> str:
    """Extract volume information from article data."""
    try:
        return article_data["Journal"]["JournalIssue"].get("Volume", "")
    except Exception:
        pass
    return ""


def extract_issue(article_data: Dict[str, Any]) -> str:
    """Extract issue information from article data."""
    try:
        return article_data["Journal"]["JournalIssue"].get("Issue", "")
    except Exception:
        pass
    return ""


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
        f.write(f"NCBI PubMed Query Metadata\n")
        f.write(f"========================\n\n")
        f.write(f"Query executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Search Parameters:\n")
        f.write(f"  Keywords: {', '.join(keywords)}\n")
        f.write(f"  Date range: {date_range[0]} to {date_range[1]}\n")
        f.write(f"  Max results requested: {max_results}\n\n")

        f.write(f"Results:\n")
        f.write(f"  Total articles found: {result_count}\n")
        f.write(f"  Execution time: {elapsed_time:.2f} seconds\n\n")

        f.write(f"Full Query:\n")
        f.write(f"  {query}\n")

    print(f"Query metadata saved to {output_file}")


def search_ncbi(
    keywords: List[str],
    date_range: Tuple[str, str],
    email: str,
    database: str = "pubmed",
    max_results: int = 100,
    output_prefix: Optional[str] = None,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Main function to search for articles on NCBI.

    Args:
        keywords: List of keywords or phrases to search for
        date_range: Tuple containing start and end dates in format 'YYYY/MM/DD'
        email: Email address for NCBI Entrez API
        database: The NCBI database to search (default: "pubmed")
        max_results: Maximum number of results to return
        output_prefix: Optional prefix for output files (CSV and TXT)
        query: Optional custom query string (if provided, will override automatic query building)

    Returns:
        List of dictionaries containing article metadata
    """
    start_time = time.time()

    # Setup Entrez
    setup_entrez(email)

    # Build query if not provided
    if query is None:
        query = build_query(keywords, date_range)

    # Search for articles
    search_results = search_articles(query, database=database, max_results=max_results)

    # Get article IDs
    id_list = search_results.get("IdList", [])

    if not id_list:
        print("No articles found matching the search criteria.")
        return []

    print(f"Found {len(id_list)} articles. Fetching details...")

    # Fetch article details
    articles = fetch_article_details(id_list, database=database)

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
        csv_file = f"{output_prefix}.csv"
        save_results_to_csv(metadata_list, csv_file)

        # Save query metadata to TXT
        txt_file = f"{output_prefix}_metadata.txt"
        save_query_metadata(
            query=query,
            keywords=keywords,
            date_range=date_range,
            max_results=max_results,
            result_count=len(metadata_list),
            start_time=start_time,
            end_time=end_time,
            output_file=txt_file,
        )

    return metadata_list


def build_query(
    keywords: Union[List[str], List[List[str]]],
    date_range: Tuple[str, str],
    article_type: str = "journal article",
    journals: Optional[List[str]] = None,
    authors: Optional[List[str]] = None,
    mesh_terms: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    affiliations: Optional[List[str]] = None,
    free_full_text: bool = False,
    exclude_keywords: Optional[List[str]] = None,
) -> str:
    """
    Construct a PubMed query string based on keywords and multiple filters.
    Supports both flat keyword lists and keyword sets (lists of keyword lists).

    Args:
        keywords: Either a flat list of keywords, or a list of keyword sets (list of lists)
                 When using sets, keywords within a set are combined with OR,
                 and different sets are combined with AND.
        date_range: Tuple containing start and end dates in format 'YYYY/MM/DD'
        article_type: Type of article to filter for (default: "journal article")
        journals: Optional list of journal names to limit search to
        authors: Optional list of author names to filter by
        mesh_terms: Optional list of MeSH terms for more precise medical searches
        languages: Optional list of languages to limit results to (e.g., ["English"])
        affiliations: Optional list of institutional affiliations to search for
        free_full_text: If True, limit to articles with free full text
        exclude_keywords: Optional list of keywords to exclude from search

    Returns:
        A formatted query string for use with Entrez.esearch
    """
    # Check if we have keyword sets (list of lists) or a flat keyword list
    is_keyword_sets = any(isinstance(item, list) for item in keywords)

    if is_keyword_sets:
        # Process each keyword set separately
        keyword_set_queries = []
        for keyword_set in keywords:
            set_parts = []
            for kw in keyword_set:
                set_parts.append(f'(("{kw}"[Title]) OR ("{kw}"[Abstract]))')
            # Join keywords within a set with OR
            keyword_set_queries.append(f"({' OR '.join(set_parts)})")
        # Join different sets with AND
        keyword_query = " AND ".join(keyword_set_queries)
    else:
        # Original behavior for flat keyword list
        keyword_parts = []
        for kw in keywords:  # type: ignore
            keyword_parts.append(f'(("{kw}"[Title]) OR ("{kw}"[Abstract]))')
        # Join with OR operator to find articles with at least one keyword match
        keyword_query = " OR ".join(keyword_parts)

    # Add date range
    start_date, end_date = date_range
    date_query = f"{start_date}:{end_date}[Date - Publication]"

    # Add article type filter
    type_query = f"{article_type}[Publication Type]"

    # Start with base query
    queries = [f"({keyword_query})", date_query, type_query]

    # Rest of the function remains the same...
    # Add journal filter if specified
    if journals and len(journals) > 0:
        journal_parts = [f'"{journal}"[Journal]' for journal in journals]
        queries.append(f"({' OR '.join(journal_parts)})")

    # Add author filter if specified
    if authors and len(authors) > 0:
        author_parts = [f'"{author}"[Author]' for author in authors]
        queries.append(f"({' OR '.join(author_parts)})")

    # Add MeSH terms if specified
    if mesh_terms and len(mesh_terms) > 0:
        mesh_parts = [f'"{term}"[MeSH Terms]' for term in mesh_terms]
        queries.append(f"({' AND '.join(mesh_parts)})")

    # Add language filter if specified
    if languages and len(languages) > 0:
        lang_parts = [f'"{lang}"[Language]' for lang in languages]
        queries.append(f"({' OR '.join(lang_parts)})")

    # Add affiliation filter if specified
    if affiliations and len(affiliations) > 0:
        affil_parts = [f'"{affil}"[Affiliation]' for affil in affiliations]
        queries.append(f"({' OR '.join(affil_parts)})")

    # Add free full text filter if requested
    if free_full_text:
        queries.append("free full text[Filter]")

    # Add exclude keywords if specified
    if exclude_keywords and len(exclude_keywords) > 0:
        exclude_parts = [f'NOT "{kw}"[Title/Abstract]' for kw in exclude_keywords]
        queries.append(f"({' AND '.join(exclude_parts)})")

    # Combine all query parts with AND
    full_query = " AND ".join(queries)

    print(full_query)

    return full_query


def search_from_yaml(config_file: str) -> List[Dict[str, Any]]:
    """
    Run an NCBI search using parameters from a YAML configuration file.
    Supports both flat keyword lists and keyword sets.

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
        email = config["email"]

        # Extract optional parameters with defaults
        database = config.get("database", "pubmed")
        max_results = config.get("max_results", 100)
        output_prefix = config.get("output_prefix", None)

        # Extract new optional query parameters
        article_type = config.get("article_type", "journal article")
        journals = config.get("journals", None)
        authors = config.get("authors", None)
        mesh_terms = config.get("mesh_terms", None)
        languages = config.get("languages", None)
        affiliations = config.get("affiliations", None)
        free_full_text = config.get("free_full_text", False)
        exclude_keywords = config.get("exclude_keywords", None)

        # Build the query with all available parameters
        query = build_query(
            keywords=keywords,
            date_range=date_range,
            article_type=article_type,
            journals=journals,
            authors=authors,
            mesh_terms=mesh_terms,
            languages=languages,
            affiliations=affiliations,
            free_full_text=free_full_text,
            exclude_keywords=exclude_keywords,
        )

        # Call search_ncbi with extracted parameters
        results = search_ncbi(
            keywords=(
                keywords
                if not isinstance(keywords[0], list)
                else [k for sublist in keywords for k in sublist]
            ),  # Flatten if nested
            date_range=date_range,
            email=email,
            database=database,
            max_results=max_results,
            output_prefix=output_prefix,
            query=query,  # Pass the custom query
        )

        return results

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {str(e)}")
        return []
    except KeyError as e:
        print(f"Error: Missing required configuration parameter: {str(e)}")
        return []


if __name__ == "__main__":
    # Example using YAML configuration
    results = search_from_yaml("./scripts/fetch/parameters.yaml")
    print(f"Retrieved {len(results)} articles")
