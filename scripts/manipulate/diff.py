#!/usr/bin/env python3
# filepath: /home/mariopasc/Python/Projects/Review_Mammography/scripts/manipulate/diff.py

import pandas as pd
import argparse
from pathlib import Path
import sys


def find_unique_entries(
    csv1_path: str, csv2_path: str, output_path: str, id_column: str = "paper_id"
) -> None:
    """
    Create a new CSV with entries from csv2 that don't exist in csv1, based on the id_column.

    This performs the set operation: csv2 - (csv1 ∩ csv2)

    Args:
        csv1_path: Path to the first CSV file
        csv2_path: Path to the second CSV file
        output_path: Path to save the output CSV file
        id_column: Column name containing the unique identifier (default: 'paper_id')
    """
    try:
        # Read both CSV files
        print(f"Reading CSV files...")
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)

        # Verify that both DataFrames have the id_column
        if id_column not in df1.columns:
            sys.exit(f"Error: Column '{id_column}' not found in {csv1_path}")
        if id_column not in df2.columns:
            sys.exit(f"Error: Column '{id_column}' not found in {csv2_path}")

        # Get the counts before filtering
        total_entries_csv1 = len(df1)
        total_entries_csv2 = len(df2)

        print(f"CSV 1 '{Path(csv1_path).name}' has {total_entries_csv1:,} entries")
        print(f"CSV 2 '{Path(csv2_path).name}' has {total_entries_csv2:,} entries")

        # Extract the unique IDs from the first CSV
        ids_in_csv1 = set(df1[id_column].dropna().unique())

        # Filter the second CSV to keep only rows where id is not in the first CSV
        unique_entries = df2[~df2[id_column].isin(ids_in_csv1)]

        # Count of unique entries
        unique_count = len(unique_entries)

        # Save the unique entries to the output CSV
        unique_entries.to_csv(output_path, index=False)

        print(f"\nResults:")
        print(f"Found {unique_count:,} entries in CSV 2 that don't exist in CSV 1")
        print(f"Output saved to: {output_path}")

    except FileNotFoundError as e:
        sys.exit(f"Error: File not found - {e}")
    except pd.errors.EmptyDataError:
        sys.exit(f"Error: One of the CSV files is empty")
    except pd.errors.ParserError:
        sys.exit(
            f"Error: Unable to parse one of the CSV files. Please verify they are valid CSV files"
        )
    except Exception as e:
        sys.exit(f"An unexpected error occurred: {e}")


def parse_key_value_arg(arg):
    """Parse a key=value argument and return (key, value)"""
    if "=" not in arg:
        raise ValueError(f"Argument '{arg}' is not in the format 'key=value'")
    key, value = arg.split("=", 1)
    return key, value


def main():
    parser = argparse.ArgumentParser(
        description="Create a CSV with entries from the second CSV that don't exist in the first CSV"
    )

    # Accept all arguments and parse them manually
    parser.add_argument("args", nargs="*", help="Arguments in the format key=value")

    args = parser.parse_args()

    # Parse the arguments manually
    kwargs = {}
    for arg in args.args:
        try:
            key, value = parse_key_value_arg(arg)
            kwargs[key] = value
        except ValueError as e:
            sys.exit(str(e))

    # Required arguments
    if "csv1" not in kwargs:
        sys.exit("Missing required argument: csv1")
    if "csv2" not in kwargs:
        sys.exit("Missing required argument: csv2")
    if "output" not in kwargs:
        sys.exit("Missing required argument: output")

    # Optional arguments
    id_column = kwargs.get("id_column", "paper_id")

    find_unique_entries(kwargs["csv1"], kwargs["csv2"], kwargs["output"], id_column)


if __name__ == "__main__":
    main()
