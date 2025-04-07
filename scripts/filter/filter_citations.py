import pandas as pd
import yaml  # type: ignore

def enhance_dataset_references(csv_file_path: str) -> None:
    """
    Check paper titles and abstracts for:
    1. Additional dataset references and update cited_dataset column
    2. Evaluation metrics and add them to a new column
    3. Remove rows containing any excluded keywords
    4. Filter out citations that don't mention any evaluation metric
    5. Remove rows with empty paper_id values

    Args:
        csv_file_path: Path to the CSV file containing citation data
    """
    try:
        print("Enhancing dataset references by scanning titles and abstracts...")

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        if "cited_dataset" not in df.columns:
            print("Error: 'cited_dataset' column not found in the CSV file.")
            return
            
        # Remove rows with empty paper_id values
        rows_before_id_filter = len(df)
        if "paper_id" in df.columns:
            # Filter out rows where paper_id is empty (NaN, None, or empty string)
            df = df[df["paper_id"].notna() & (df["paper_id"] != "")]
            id_filtered = rows_before_id_filter - len(df)
            if id_filtered > 0:
                print(f"Removed {id_filtered} rows with empty paper_id values")

        # Load configuration
        try:
            with open("./scripts/fetch/parameters.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Get evaluation metrics from config
            evaluation_metrics = {}
            for keyword_set in config.get("keyword_sets", []):
                # Find the evaluation metrics set
                if any(
                    metric in str(keyword_set)
                    for metric in ["ROC", "AUC", "sensitivity"]
                ):
                    evaluation_metrics = {
                        "metrics": [metric for metric in keyword_set if metric]
                    }
                    break

            # Get excluded keywords from config
            exclude_keywords = config.get("exclude_keywords", [])
        except Exception as e:
            print(f"Warning: Could not load evaluation metrics from config: {str(e)}")
            evaluation_metrics = {"metrics": []}
            exclude_keywords = []

        # Create a reverse mapping from dataset names to identifiers
        # Also include common variations and abbreviations
        dataset_names = {
            "DDSM": [
                "DDSM",
                "Digital Database for Screening Mammography",
                "digital database for screening",
            ],
            "MIAS": ["MIAS", "Mammographic Image Analysis Society"],
            "BancoWeb": ["BancoWeb", "LAPIMO", "Online Mammographic Images Database"],
            "UCSF/LLNL": [
                "UCSF/LLNL",
                "UCSF",
                "LLNL",
                "High Resolution Digital Mammogram Library",
            ],
            "CBIS-DDSM": [
                "CBIS-DDSM",
                "Curated Breast Imaging Subset",
                "curated breast imaging subset of ddsm",
            ],
            "INbreast": ["INbreast", "IN breast"],
            "VinDr-Mammo": ["VinDr-Mammo", "VinDr", "Vin-DR"],
            "CMMD": ["CMMD", "Chinese Mammography Database"],
            "OPTIMAM": ["OPTIMAM", "OPTIMAM Mammography Image Database"],
            "CSAW": ["CSAW", "Cohort of Screen-Aged Women"],
            "EMBED": ["EMBED", "EMory BrEast imaging Dataset", "Emory Breast imaging"],
            "ADMANI": [
                "ADMANI",
                "Annotated Digital Mammograms",
                "Associated Non-Image Datasets",
            ],
            "BCDR": ["BCDR", "Breast Cancer Digital Repository", "BCDR-FM", "BCDR-DM"],
        }

        # Count of modifications
        modified_rows = 0
        added_references = 0
        rows_before = len(df)

        # Add evaluation_metrics column if it doesn't exist
        if "evaluation_metrics" not in df.columns:
            df["evaluation_metrics"] = ""

        # Track papers that contain evaluation metrics
        contains_metrics = pd.Series([False] * len(df))

        # Process each row
        for index, row in df.iterrows():
            title = str(row["title"]).lower() if not pd.isna(row["title"]) else ""
            abstract = (
                str(row["abstract"]).lower() if not pd.isna(row["abstract"]) else ""
            )

            # Get current dataset references
            try:
                current_datasets = row["cited_dataset"].split(", ")
            except:
                # Handle case where cited_dataset is not in expected format
                if isinstance(row["cited_dataset"], str):
                    current_datasets = [row["cited_dataset"]]
                else:
                    current_datasets = []

            new_datasets = []

            # Check for each dataset name in title and abstract
            for dataset, variations in dataset_names.items():
                if dataset not in current_datasets:
                    for variation in variations:
                        if variation.lower() in title or variation.lower() in abstract:
                            new_datasets.append(dataset)
                            added_references += 1
                            break

            # If new datasets found, update the row
            if new_datasets:
                all_datasets = sorted(list(set(current_datasets + new_datasets)))
                df.at[index, "cited_dataset"] = ", ".join(all_datasets)
                modified_rows += 1

            # Check for evaluation metrics in title and abstract
            detected_metrics = []
            for metric in evaluation_metrics.get("metrics", []):
                if metric.lower() in title or metric.lower() in abstract:
                    detected_metrics.append(metric)

            # If metrics found, update the evaluation_metrics column and mark the paper
            if detected_metrics:
                df.at[index, "evaluation_metrics"] = ", ".join(
                    sorted(set(detected_metrics))
                )
                contains_metrics[index] = True

        # Now filter out rows containing excluded keywords
        if exclude_keywords:
            print(
                f"Filtering out rows with excluded keywords: {', '.join(exclude_keywords)}"
            )

            # Define text columns to check for excluded keywords
            text_columns = [
                "title",
                "abstract",
                "venue",
                "authors",
                "publication_types",
                "fields_of_study",
            ]

            # Create a mask for rows to keep (initially all True)
            mask = pd.Series([True] * len(df))

            # For each keyword, update mask to exclude matching rows
            for keyword in exclude_keywords:
                for col in text_columns:
                    if col in df.columns:
                        # Update mask to exclude rows where this keyword appears in this column
                        mask = mask & ~df[col].astype(str).str.lower().str.contains(
                            keyword.lower(), na=False
                        )

            # Apply the mask to filter the dataframe
            df = df[mask]

        # Filter out papers that don't mention any evaluation metric
        rows_before_metrics_filter = len(df)
        df = df[df["evaluation_metrics"] != ""]
        metrics_filtered = rows_before_metrics_filter - len(df)
        print(f"Filtered out {metrics_filtered} papers without evaluation metrics")

        # Save updated DataFrame
        df.to_csv(csv_file_path, index=False)

        rows_after = len(df)
        rows_removed = rows_before - rows_after

        print(f"Dataset reference enhancement complete:")
        print(f"  - Modified rows: {modified_rows}")
        print(f"  - Added references: {added_references}")
        print(f"  - Rows removed by exclusion filters: {rows_removed}")
        print(f"  - Rows removed for lacking evaluation metrics: {metrics_filtered}")
        if "paper_id" in df.columns:
            print(f"  - Rows removed for empty paper_id: {id_filtered}")
        print(f"  - Final number of rows: {rows_after}")

    except Exception as e:
        print(f"Error during dataset reference enhancement: {str(e)}")
        import traceback

        traceback.print_exc()


enhance_dataset_references(csv_file_path="data/csvs/info_citations.csv")
