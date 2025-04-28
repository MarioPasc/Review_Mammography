import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def get_dataset_counts(csv_path):
    df = pd.read_csv(csv_path)
    all_datasets = []
    if 'cited_dataset' in df.columns:
        for entry in df['cited_dataset'].dropna():
            all_datasets.extend([d.strip() for d in entry.split(',')])
    return Counter(all_datasets)

# Paths to the two CSV files
main_csv = 'data/csvs/info_citations_included.csv'
other_csv = 'data/csvs/info_citations_excluded.csv'

# Get counts
main_counts = get_dataset_counts(main_csv)
other_counts = get_dataset_counts(other_csv)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Main CSV pie chart
axes[0].pie(main_counts.values(), labels=main_counts.keys(), autopct='%d', startangle=140)
axes[0].set_title('Number of Included Datasets (info_citations_included.csv)')
axes[0].axis('equal')

# Other CSV pie chart
axes[1].pie(other_counts.values(), labels=other_counts.keys(), autopct='%d', startangle=140)
axes[1].set_title(f'Number of Excluded Datasets ({other_csv})')
axes[1].axis('equal')

plt.tight_layout()
plt.show()