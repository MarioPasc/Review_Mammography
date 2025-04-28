import pandas as pd
import requests
import time
import os
from typing import List, Dict, Any, Optional

def read_csv(csv_path: str) -> pd.DataFrame:
    """Read the CSV file and return a pandas DataFrame."""
    try:
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            print(f"CSV file {csv_path} not found. Creating a new one.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def fetch_paper_info(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch paper information from Semantic Scholar API using DOI."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {
        "fields": "title,authors,year,venue,url,citationCount,abstract,externalIds,journal,references"
    }
    headers = {"Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"Rate limit exceeded. Waiting 30 seconds before retrying...")
            time.sleep(30)
            return fetch_paper_info(doi)  # Retry after waiting
        elif response.status_code == 404:
            print(f"Paper with DOI {doi} not found")
            return None
        else:
            print(f"Error fetching paper with DOI {doi}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception while fetching paper with DOI {doi}: {e}")
        return None

def map_paper_to_csv_fields(paper_info: Dict[str, Any], csv_columns: List[str]) -> Dict[str, Any]:
    """Map paper information to CSV columns."""
    paper_data = {"inclusion_type": "manual"}
    
    # Map known fields from API to CSV
    field_mapping = {
        "title": ["title", "paper title"],
        "year": ["year", "publication year"],
        "venue": ["venue", "conference", "publication venue"],
        "url": ["url", "link"],
        "abstract": ["abstract"]
    }
    
    # Map direct fields
    for api_field, possible_csv_fields in field_mapping.items():
        if api_field in paper_info:
            for csv_field in possible_csv_fields:
                matching_cols = [col for col in csv_columns if col.lower() == csv_field.lower()]
                if matching_cols:
                    paper_data[matching_cols[0]] = paper_info[api_field]
                    break
    
    # Handle DOI
    if "externalIds" in paper_info and "DOI" in paper_info["externalIds"]:
        doi_col = next((col for col in csv_columns if col.lower() == "doi"), None)
        if doi_col:
            paper_data[doi_col] = paper_info["externalIds"]["DOI"]
    
    # Handle authors
    if "authors" in paper_info:
        author_names = [author.get("name", "") for author in paper_info["authors"]]
        authors_col = next((col for col in csv_columns if col.lower() in ["authors", "author"]), None)
        if authors_col:
            paper_data[authors_col] = ", ".join(author_names)
    
    # Handle citation count
    if "citationCount" in paper_info:
        citations_col = next((col for col in csv_columns if "citation" in col.lower()), None)
        if citations_col:
            paper_data[citations_col] = paper_info["citationCount"]
    
    return paper_data

def update_csv_with_papers(doi_dict: Dict[str, str], csv_path: str) -> None:
    """Update the CSV file with papers from the DOI dictionary.
    
    Args:
        doi_dict: Dictionary mapping DOIs to their cited dataset values
        csv_path: Path to the CSV file
    """
    # Read the CSV file
    df = read_csv(csv_path)
    
    # Add inclusion_type column if it doesn't exist
    if "inclusion_type" not in df.columns:
        df["inclusion_type"] = "automatic"
        
    # Add cited_dataset column if it doesn't exist
    if "cited_dataset" not in df.columns:
        df["cited_dataset"] = None
    
    # Get existing DOIs to avoid duplicates
    doi_col = next((col for col in df.columns if col.lower() == "doi"), None)
    existing_dois = set()
    
    if doi_col and not df.empty:
        existing_dois = set(df[doi_col].dropna().str.lower())
    
    # Process each DOI
    new_papers = []
    for doi, cited_dataset in doi_dict.items():
        doi_lower = doi.lower()
        if doi_lower not in existing_dois:
            print(f"Processing DOI: {doi}")
            paper_info = fetch_paper_info(doi)
            if paper_info:
                import hashlib
                paper_data = map_paper_to_csv_fields(paper_info, df.columns.tolist())
                paper_data["paper_id"] = hashlib.shake_256(str.encode("utf-8")).hexdigest(length=20)  # Unique identifier for the paper
                paper_data["cited_dataset"] = cited_dataset  # Add the cited dataset value
                new_papers.append(paper_data)
                print(f"Adding new paper: {paper_data.get('Title', doi)} with dataset: {cited_dataset}")
            time.sleep(1)  # Rate limiting
        else:
            print(f"DOI {doi} already exists in the CSV")
            # If you want to update the cited_dataset for existing entries, uncomment:
            # if doi_col:
            #     df.loc[df[doi_col].str.lower() == doi_lower, "cited_dataset"] = cited_dataset
            #     print(f"Updated cited_dataset for {doi} to {cited_dataset}")
    
    # Add new papers to DataFrame
    if new_papers:
        new_df = pd.DataFrame(new_papers)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"Added {len(new_papers)} new papers")
    
    # Write the updated DataFrame back to CSV
    df.to_csv(csv_path, index=False)
    print(f"CSV updated at {csv_path}")

def main():
    """Main function to handle script execution."""
    csv_path = "data/csvs/info_citations_included.csv"
    
    # Dictionary mapping DOIs to their cited dataset values
    dois = {
        "10.3934/MBE.2021256": "DDSM, BCDR, MIAS, INbreast",
        "10.1049/iet-cvi.2016.0425": "MIAS, BCDR"
    }
    
    update_csv_with_papers(dois, csv_path)

if __name__ == "__main__":
    main()