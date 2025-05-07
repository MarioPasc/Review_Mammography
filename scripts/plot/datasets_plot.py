#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset analytics figure (2 × 2 grid).

a. Stacked paper counts      b. Citation distribution (log)
c. Included papers / year    d. Excluded papers / year
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, LogLocator, ScalarFormatter

# ───────────────────────────────────────────── configuration ──
FONTSIZE = 14
PALETTE = {"included": "#0072B2",        # Okabe–Ito blue
           "excluded": "#D55E00"}        # Okabe–Ito vermillion
plt.rcParams.update({"font.size": FONTSIZE})
sns.set_style("whitegrid")

# ───────────────────────────────────────────── data helpers ──
def load_and_tag(exc_csv: Path, inc_csv: Path) -> pd.DataFrame:
    exc = pd.read_csv(exc_csv).assign(status="excluded")
    inc = pd.read_csv(inc_csv).assign(status="included")
    return pd.concat([exc, inc], ignore_index=True)


def explode_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper × dataset)."""
    records = []
    for _, row in df.iterrows():
        if pd.isna(row["cited_dataset"]):
            continue
        for ds in str(row["cited_dataset"]).strip('"').split(","):
            records.append(
                dict(
                    dataset=ds.strip(),
                    paper_id=row["paper_id"],
                    status=row["status"],
                    citation_count=pd.to_numeric(row["citation_count"], errors="coerce"),
                    year=pd.to_numeric(row["year"], errors="coerce"),
                )
            )
    return pd.DataFrame(records).dropna(subset=["dataset"])

# ───────────────────────────────────────────── plotting ──
def stacked_counts(ax: plt.Axes, dset: pd.DataFrame, ordered: list[str]) -> None:
    counts = (
        dset.groupby(["dataset", "status"])
        .size()
        .unstack(fill_value=0)
        .loc[ordered]
    )
    y = np.arange(len(ordered))
    ax.barh(y, counts["included"], color=PALETTE["included"], label="Included")
    ax.barh(y, counts["excluded"], left=counts["included"],
            color=PALETTE["excluded"], label="Excluded")
    ax.set_yticks(y, ordered)
    ax.invert_yaxis()
    ax.set_xlabel("Number of papers")
    ax.set_title("a. Dataset counts", loc="left", fontweight="bold")


def citation_box(ax: plt.Axes, dset: pd.DataFrame, ordered: list[str]) -> None:
    sns.boxenplot(
        data=dset,
        x="citation_count",
        y="dataset",
        hue="status",
        order=ordered,
        palette=PALETTE,
        ax=ax,
        linewidth=0.6,
        saturation=0.8,
        dodge=True,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="all"))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # add small left margin so boxes are not flush with y-axis
    ax.set_xlim(left=10**-0.1)           # ≈ 0.79 on log scale
    ax.set_xlabel("Number of citations (log scale)")
    ax.set_ylabel(None)
    ax.set_title("b. Citation counts", loc="left", fontweight="bold")
    ax.get_legend().remove()


def yearly_scatter(ax: plt.Axes, dset: pd.DataFrame,
                   ordered: list[str], status: str, letter: str) -> None:
    grp = (
        dset.query("status == @status")
        .groupby(["year", "dataset"])
        .size()
        .reset_index(name="n")
        .dropna(subset=["year"])
    )
    grp["s"] = grp["n"].pow(0.5) * 30
    sns.scatterplot(
        data=grp,
        x="year",
        y="dataset",
        size="s",
        sizes=(grp["s"].min(), grp["s"].max()),
        legend=False,
        color=PALETTE[status],
        ax=ax,
    )
    ax.set_ylabel(None)
    ax.set_xlabel("Year")
    ax.set_title(f"{letter}. {status.capitalize()} papers per year",
                 loc="left", fontweight="bold")

# ───────────────────────────────────────────── orchestrator ──
def make_figure(exc_csv: Path, inc_csv: Path) -> plt.Figure:
    raw = load_and_tag(exc_csv, inc_csv)
    dset = explode_datasets(raw)

    # common dataset order for ALL panels
    ordered = (
        dset.groupby("dataset")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    dset["dataset"] = pd.Categorical(dset["dataset"],
                                     categories=ordered,
                                     ordered=True)

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    gs_top = outer[0].subgridspec(1, 2, wspace=0.05)
    gs_bot = outer[1].subgridspec(1, 2, wspace=0.05)

    # ── create axes
    ax_a = fig.add_subplot(gs_top[0])
    ax_b = fig.add_subplot(gs_top[1], sharey=ax_a)
    ax_c = fig.add_subplot(gs_bot[0])
    ax_d = fig.add_subplot(gs_bot[1], sharey=ax_c)

    # ── plotting
    stacked_counts(ax_a, dset, ordered)
    citation_box(ax_b, dset, ordered)
    yearly_scatter(ax_c, dset, ordered, "included", "c")
    yearly_scatter(ax_d, dset, ordered, "excluded", "d")

    # ── hide duplicate y-tick labels on panels b & d
    ax_b.tick_params(labelleft=False)
    ax_d.tick_params(labelleft=False)

    # ── identical x-ticks for the scatter panels
    min_year = int(dset["year"].min(skipna=True))
    max_year = int(dset["year"].max(skipna=True))
    for ax in (ax_c, ax_d):
        ax.set_xlim(min_year - 1, max_year + 1)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xticklabels(ax.get_xticks().astype(int))

    # ── global legend below everything
    handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[s], markersize=15,
               label=s.capitalize())
        for s in ("included", "excluded")
    ]
    fig.legend(handles=handles,
               loc="lower center",
               ncol=2,
               bbox_to_anchor=(0.5, -0.04),
               frameon=False)

    return fig


# ───────────────────────────────────────────── CLI entry point ──
def main() -> None:
    src = Path("data/csvs")
    excl_csv = src / "info_citations_excluded.csv"
    incl_csv = src / "info_citations_included.csv"

    outdir = Path("data/plots")
    outdir.mkdir(parents=True, exist_ok=True)

    fig = make_figure(excl_csv, incl_csv)
    fig.savefig(outdir / "dataset_analysis.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
