import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def order_by_column(file_path: str = "data/csvs/info_citations.csv", 
                   column: str = "citation_count",
                   output_path: str = "data/csvs/info_citations.csv", 
                   ascending=False):
    """
    Orders the rows in a CSV file by a specified column.
    
    Parameters:
    file_path (str): Path to the input CSV file
    column (str): Column name to sort by
    output_path (str, optional): Path to save the sorted CSV file. If None, returns the sorted DataFrame.
    ascending (bool, optional): Whether to sort in ascending order. Default is False (descending order).
    
    Returns:
    pandas.DataFrame: Sorted DataFrame if output_path is None, otherwise None
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure the specified column exists
    if column not in df.columns:
        raise ValueError(f"The CSV file does not contain a '{column}' column.")
    
    # Try to convert column to numeric if possible, handling non-numeric values
    try:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    except:
        # If conversion fails, use the column as-is (e.g., for string columns)
        pass
    
    # Sort by the specified column
    df_sorted = df.sort_values(by=column, ascending=ascending, na_position='last')
    
    # Save or return the sorted DataFrame
    if output_path:
        # Make sure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df_sorted.to_csv(output_path, index=False)
        return None
    else:
        return df_sorted

def plot_citation_histogram(file_path: str = "data/csvs/info_citations.csv",
                           output_path: str = "data/figures/citation_histogram.png",
                           bins: int = 20,
                           log_scale: bool = False):
    """
    Creates a histogram of citation counts from the CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing citation counts
    output_path (str): Path to save the histogram plot
    bins (int): Number of bins for the histogram
    log_scale (bool): Whether to use log scale for citation counts
    
    Returns:
    None
    """
    # Get the data - use the existing function but don't sort or save to file
    df = pd.read_csv(file_path)
    
    # Convert citation_count to numeric, handling non-numeric values
    df['citation_count'] = pd.to_numeric(df['citation_count'], errors='coerce')
    
    # Remove NA values
    citation_counts = df['citation_count'].dropna()
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Calculate appropriate bins
    if log_scale:
        # For log scale, we need to handle zeros and use log bins
        citation_counts = citation_counts[citation_counts > 0]
        if not citation_counts.empty:
            bins = np.logspace(np.log10(citation_counts.min()), 
                              np.log10(citation_counts.max()), 
                              bins)
    
    # Create the histogram
    plt.hist(citation_counts, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Set labels and title
    plt.xlabel('Citation Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Citation Counts')
    
    # Set log scale if requested
    if log_scale:
        plt.xscale('log')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Make sure the directory exists
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {output_path}")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process mammography review data.')
    parser.add_argument('--input', '-i', type=str, default="data/csvs/info_citations.csv",
                        help='Path to the input CSV file')
    parser.add_argument('--output', '-o', type=str, default="data/csvs/info_citations.csv",
                        help='Path to save the sorted CSV file')
    parser.add_argument('--column', '-c', type=str, default="citation_count",
                        help='Column name to sort by (default: citation_count)')
    parser.add_argument('--ascending', '-a', action='store_true',
                        help='Sort in ascending order (default is descending)')
    parser.add_argument('--histogram', '-hist', action='store_true',
                        help='Plot histogram of citation counts')
    parser.add_argument('--hist-output', type=str, default="data/figures/citation_histogram.png",
                        help='Path to save the histogram')
    parser.add_argument('--bins', type=int, default=20,
                        help='Number of bins for the histogram')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use log scale for the histogram')
    
    args = parser.parse_args()
    
    # If histogram option is selected, plot histogram and exit
    if args.histogram:
        plot_citation_histogram(args.input, args.hist_output, args.bins, args.log_scale)
        return
    
    # Otherwise, perform the sorting operation
    result = order_by_column(args.input, args.column, args.output, args.ascending)
    
    if result is not None:
        print(f"Data sorted by {args.column}:")
        print(result)
    else:
        print(f"Data sorted by {args.column} saved to {args.output}")

if __name__ == "__main__":
    main()