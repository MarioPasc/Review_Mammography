import plotly.graph_objects as go  # type: ignore
from collections import Counter, defaultdict
import itertools
import pandas as pd
import os
import numpy as np
from plotly.offline import plot  # type: ignore


def plot_dataset_author_venue_sankey(
    data_source, output_dir="data/plots", top_n_authors=20
):
    """
    Generates a Sankey diagram (datasets → authors → venues) from citation data.

    Parameters
    ----------
    data_source : str or pandas.DataFrame
        Either a path to a CSV file or a pandas DataFrame with citation data
    output_dir : str
        Directory to save the visualization
    top_n_authors : int
        Number of top authors to include in the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data if path is provided
    if isinstance(data_source, str):
        print(f"Loading citation data from {data_source}")
        df = pd.read_csv(data_source)
    else:
        df = data_source

    # Verify and prepare columns
    required_columns = [
        "cited_dataset",
        "authors",
        "venue",
        "fields_of_study",
        "citation_count",
        "year",
    ]
    for col in required_columns:
        if col not in df.columns:
            if col == "venue" and "venue_name" in df.columns:
                df["venue"] = df["venue_name"]
            else:
                df[col] = None

    # Rename venue column if needed
    if "venue_name" not in df.columns:
        df["venue_name"] = df["venue"]

    # Fill NaN values
    df["venue_name"] = df["venue_name"].fillna("Unknown Venue")
    df["fields_of_study"] = df["fields_of_study"].fillna("Unknown")
    df["year"] = df["year"].fillna(0).astype(int)
    df["citation_count"] = df["citation_count"].fillna(0).astype(int)

    # Process dataset and author lists
    df["authors_list"] = (
        df["authors"]
        .fillna("")
        .apply(lambda x: [a.strip() for a in x.split(",") if a.strip()])
    )
    df["datasets_list"] = (
        df["cited_dataset"]
        .fillna("")
        .apply(lambda x: [d.strip() for d in x.split(",") if d.strip()])
    )

    # Count top authors by paper frequency
    all_authors = list(itertools.chain.from_iterable(df["authors_list"].dropna()))
    author_counts = Counter(all_authors)
    top_authors = [a for a, c in author_counts.most_common(top_n_authors) if c > 1]

    # Get citation counts per author
    author_citations = defaultdict(int)
    for _, row in df.iterrows():
        for author in row["authors_list"]:
            if author in top_authors:
                author_citations[author] += row["citation_count"]

    # Count datasets
    dataset_counter = Counter(
        itertools.chain.from_iterable(df["datasets_list"].dropna())
    )
    all_datasets = list(dataset_counter.keys())
    print(f"Including all {len(all_datasets)} datasets in visualization")
    top_datasets = all_datasets  # Use all datasets instead of just top 20

    # Simplify field of study to main categories
    def simplify_field(field):
        if pd.isnull(field):
            return "Unknown"

        fields = str(field).split(",")
        for primary in fields:
            primary = primary.strip()
            if primary in [
                "Medicine",
                "Computer Science",
                "Engineering",
                "Biology",
                "Physics",
            ]:
                return primary

        # Try to detect field from first term
        primary = fields[0].strip()
        if "computer" in primary.lower():
            return "Computer Science"
        elif "medic" in primary.lower() or "health" in primary.lower():
            return "Medicine"
        elif "engineer" in primary.lower():
            return "Engineering"

        return "Other"

    df["primary_field"] = df["fields_of_study"].apply(simplify_field)

    # Assign each venue its field of study
    venue_fields = {}
    venue_citations = defaultdict(int)
    venue_years = defaultdict(list)

    for _, row in df.iterrows():
        venue = row["venue_name"]
        field = row["primary_field"]
        venue_citations[venue] += row["citation_count"]
        venue_years[venue].append(row["year"])

        if venue not in venue_fields:
            venue_fields[venue] = field

    # Build links between nodes
    dataset_to_author_links = []
    author_to_venue_links = []

    for _, row in df.iterrows():
        if not isinstance(row.get("authors_list"), list) or not isinstance(
            row.get("datasets_list"), list
        ):
            continue

        venue = row["venue_name"]
        authors = [a for a in row["authors_list"] if a in top_authors]
        datasets = [d for d in row["datasets_list"] if d in top_datasets]

        for author in authors:
            for dataset in datasets:
                dataset_to_author_links.append((dataset, author, row["citation_count"]))
            author_to_venue_links.append((author, venue, row["citation_count"]))

    # Aggregate link values
    da_links = defaultdict(int)
    for source, target, value in dataset_to_author_links:
        da_links[(source, target)] += value

    av_links = defaultdict(int)
    for source, target, value in author_to_venue_links:
        av_links[(source, target)] += value

    # Count connections for each node for better sorting
    dataset_connections = defaultdict(int)
    author_connections = defaultdict(int)
    venue_connections = defaultdict(int)

    # Count dataset connections (how many authors a dataset connects to)
    for (dataset, author), _ in da_links.items():
        dataset_connections[dataset] += 1

    # Count venue connections (how many authors connect to a venue)
    for (author, venue), _ in av_links.items():
        venue_connections[venue] += 1

    # Count author connections (connections to both datasets and venues)
    for (dataset, author), _ in da_links.items():
        author_connections[author] += 1
    for (author, venue), _ in av_links.items():
        author_connections[author] += 1

    # Prepare Sankey data
    # Create lists of unique nodes
    datasets_used = set(source for (source, _), _ in da_links.items())
    authors_used = set(target for (_, target), _ in da_links.items())
    venues_used = set(target for (_, target), _ in av_links.items())
    # Sort nodes by number of connections (most connected first)
    datasets_sorted = sorted(
        datasets_used, key=lambda x: -dataset_connections.get(x, 0)
    )
    authors_sorted = sorted(authors_used, key=lambda x: -author_connections.get(x, 0))
    venues_sorted = sorted(venues_used, key=lambda x: -venue_connections.get(x, 0))

    # Create a single list of all nodes
    all_nodes = datasets_sorted + authors_sorted + venues_sorted

    # Create a mapping from node names to indices
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    # Create source, target, and value lists for Sankey
    sources = []
    targets = []
    values = []

    # Add dataset-author links
    for (source, target), value in da_links.items():
        if source in node_indices and target in node_indices:
            sources.append(node_indices[source])
            targets.append(node_indices[target])
            values.append(value)

    # Add author-venue links
    for (source, target), value in av_links.items():
        if source in node_indices and target in node_indices:
            sources.append(node_indices[source])
            targets.append(node_indices[target])
            values.append(value)

    # Define node colors
    node_colors = []
    node_labels = []

    # Color palette for fields
    field_palette = {
        "Medicine": "#2171b5",  # Blue
        "Engineering": "#31a354",  # Green
        "Computer Science": "#f16913",  # Orange
        "Biology": "#6a51a3",  # Purple
        "Physics": "#cc4c02",  # Brown
        "Other": "#cc4c02",  # Brown
        "Unknown": "#969696",  # Gray
    }

    # Create node labels and colors
    for node in all_nodes:
        # Format labels
        if node in datasets_sorted:
            # Truncate dataset names if too long
            node_labels.append(node[:30] + "..." if len(node) > 30 else node)
            node_colors.append("#DDAA33")  # Yellow for datasets
        elif node in authors_sorted:
            # Format author names
            name_parts = node.split()
            if len(name_parts) > 1:
                label = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"
            else:
                label = node
            node_labels.append(label)
            node_colors.append("#BB5566")  # Pink for authors
        else:
            # Venue with year range
            venue = node
            years = venue_years.get(venue, [])
            if years:
                year_range = (
                    f"{min(years)}-{max(years)}"
                    if min(years) != max(years)
                    else f"{min(years)}"
                )
                venue_short = venue[:20] + "..." if len(venue) > 20 else venue
                label = f"{venue_short} ({year_range})"
            else:
                venue_short = venue[:20] + "..." if len(venue) > 20 else venue
                label = venue_short
            node_labels.append(label)

            # Color venues by field
            field = venue_fields.get(node, "Unknown")
            color = field_palette.get(field, "#969696")
            node_colors.append(color)

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    hovertemplate="%{source.label} → %{target.label}<br>Value: %{value}<extra></extra>",
                ),
            )
        ]
    )

    # Update layout - removed the title as requested
    fig.update_layout(
        font=dict(size=12), width=1200, height=800, margin=dict(l=25, r=25, b=25, t=50)
    )

    # Add annotations for column labels with adjusted positions
    fig.add_annotation(
        x=-0.02,
        y=1.05,  # Moved left
        xref="paper",
        yref="paper",
        text="DATASETS",
        showarrow=False,
        font=dict(size=16, color="black", family="Arial Black"),
    )

    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text="AUTHORS",
        showarrow=False,
        font=dict(size=16, color="black", family="Arial Black"),
    )

    fig.add_annotation(
        x=1.02,
        y=1.05,  # Moved right
        xref="paper",
        yref="paper",
        text="VENUES",
        showarrow=False,
        font=dict(size=16, color="black", family="Arial Black"),
    )

    # Add legend for node types
    legend_items = [
        {"name": "Dataset", "color": "#DDAA33"},
        {"name": "Author", "color": "#BB5566"},
    ]

    # Add field colors to legend
    for field, color in field_palette.items():
        if field in venue_fields.values():
            legend_items.append({"name": field, "color": color})

    # Position the legend items horizontally
    legend_x = 0.5
    legend_y = -0.15
    spacing = 0.1

    for i, item in enumerate(legend_items):
        x_pos = legend_x + (i - len(legend_items) / 2) * spacing

        # Add colored rectangle
        fig.add_shape(
            type="rect",
            x0=x_pos - 0.02,
            y0=legend_y - 0.01,
            x1=x_pos + 0.02,
            y1=legend_y + 0.01,
            fillcolor=item["color"],
            line=dict(color="black", width=1),
            xref="paper",
            yref="paper",
        )

        # Add text label
        fig.add_annotation(
            x=x_pos,
            y=legend_y - 0.03,
            text=item["name"],
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=10),
        )

    # Save the figure
    output_file = os.path.join(output_dir, "dataset_author_venue_sankey.html")
    plot(fig, filename=output_file, auto_open=False)
    print(f"Sankey diagram saved to {output_file}")

    # Also save as static image
    image_file = os.path.join(output_dir, "dataset_author_venue_sankey.svg")
    fig.write_image(image_file, width=1200, height=800, scale=2)
    print(f"Sankey image saved to {image_file}")

    # Return stats for reporting
    return {
        "datasets": len(datasets_sorted),
        "authors": len(authors_sorted),
        "venues": len(venues_sorted),
        "total_links": len(sources),
    }


if __name__ == "__main__":
    # Add a function call to plot with our citation CSV
    csv_file = "data/csvs/info_citations.csv"
    stats = plot_dataset_author_venue_sankey(
        csv_file,
        output_dir="data/plots",
        top_n_authors=1000,  # Adjust number of authors as needed
    )
    print(f"Sankey diagram statistics: {stats}")
