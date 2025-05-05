import pandas as pd
import numpy as np # Import numpy for NaN handling if needed

# Define file paths (adjust if necessary)
reason_file = '/home/mario/Python/Projects/Review_Mammography/data/csvs/info_citations_exclusion_reason.csv'
included_file = '/home/mario/Python/Projects/Review_Mammography/data/csvs/info_citations_included.csv'
excluded_file = '/home/mario/Python/Projects/Review_Mammography/data/csvs/info_citations_excluded.csv'

try:
    # 1. Read the reason file and identify papers to move
    reason_df = pd.read_csv(reason_file)
    papers_to_move_info = reason_df[reason_df['reject_reason'].notna() & (reason_df['reject_reason'] != '')]

    if papers_to_move_info.empty:
        print("No papers found with a 'reject_reason' in", reason_file)
        # Exit or continue depending on desired behavior
    else:
        ids_to_move = papers_to_move_info['paper_id'].tolist()
        # Create a map for paper_id to reject_reason
        reason_map = papers_to_move_info.set_index('paper_id')['reject_reason'].to_dict()
        print(f"Found {len(ids_to_move)} paper(s) to move based on 'reject_reason'.")

        # 2. Read the included file
        included_df = pd.read_csv(included_file)
        original_included_count = len(included_df)

        # 3. Identify and extract rows to move from included_df
        rows_to_move = included_df[included_df['paper_id'].isin(ids_to_move)].copy()

        if not rows_to_move.empty:
            # 4. Remove rows from included_df
            included_df = included_df[~included_df['paper_id'].isin(ids_to_move)]
            print(f"Removed {len(rows_to_move)} row(s) from {included_file}.")

            # 5. Read the excluded file
            excluded_df = pd.read_csv(excluded_file)
            original_excluded_count = len(excluded_df)

            # 6. Prepare the rows to be added to excluded_df
            # Add the new columns
            rows_to_move['exclusion_filter'] = 'manual'
            rows_to_move['exclusion_reason'] = rows_to_move['paper_id'].map(reason_map)

            # Ensure columns match the target excluded_df
            # Drop columns from rows_to_move that are not in excluded_df
            cols_to_drop = [col for col in rows_to_move.columns if col not in excluded_df.columns]
            if cols_to_drop:
                rows_to_move = rows_to_move.drop(columns=cols_to_drop)
                print(f"Dropped columns not in target: {cols_to_drop}")

            # Add columns present in excluded_df but not in rows_to_move (if any)
            # This step might not be strictly necessary if included_df has a superset
            # of columns (excluding the specific exclusion/inclusion ones), but it's safer.
            missing_cols = [col for col in excluded_df.columns if col not in rows_to_move.columns]
            for col in missing_cols:
                rows_to_move[col] = np.nan # Or pd.NA, or '', depending on desired fill value

            # Reorder columns to match excluded_df
            rows_to_move = rows_to_move[excluded_df.columns]

            # 7. Append rows to excluded_df
            excluded_df = pd.concat([excluded_df, rows_to_move], ignore_index=True)
            print(f"Added {len(rows_to_move)} row(s) to {excluded_file}.")

            # 8. Write updated DataFrames back to CSV
            included_df.to_csv(included_file, index=False)
            excluded_df.to_csv(excluded_file, index=False)

            print("\n--- Summary ---")
            print(f"Original included count: {original_included_count}")
            print(f"Final included count:    {len(included_df)}")
            print(f"Original excluded count: {original_excluded_count}")
            print(f"Final excluded count:    {len(excluded_df)}")
            print("CSV files updated successfully.")

        else:
            print(f"No matching paper_ids found in {included_file} to move.")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except KeyError as e:
    print(f"Error: Column not found - {e}. Please check CSV headers.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
