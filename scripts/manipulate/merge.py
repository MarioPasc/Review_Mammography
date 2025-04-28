import pandas as pd

csv1_path = "data/csvs/old.csv"
csv2_path = "data/csvs/new_entries.csv"
output_path = "data/csvs/combined.csv"

# Read both CSVs
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Concatenate and drop duplicates by 'paper_id'
df_union = pd.concat([df1, df2], ignore_index=True)
df_union = df_union.drop_duplicates(subset="paper_id", keep="first")

# Save to output
df_union.to_csv(output_path, index=False)

print(f"Union CSV saved to {output_path}")
