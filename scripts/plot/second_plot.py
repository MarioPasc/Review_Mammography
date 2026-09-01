#!/usr/bin/env python3
"""Visual analytics for the citation-screening pipeline.

Creates a 2 × 2 figure:
a. OA × decision heat-map        b. automatic filter counts
c. top keyword exclusions        d. manual-screening codes (1–7)
"""

from pathlib import Path
from typing import Mapping

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FONTSIZE = 16  # single point of control

#: Mapping between manual numeric codes and their semantics (Neurocomputing style).
MANUAL_CODES: dict[int, str] = {
    1: "No-Access",
    2: "Book",
    3: "Not mammography-specific task",
    4: "Unspecified #images",
    5: "Out of scope",
    6: "ML but not DL",
    7: "Private dataset",
}


# -----------------------------------------------------------------------------#
#                               helper functions                               #
# -----------------------------------------------------------------------------#
def load_csv(path: str | Path) -> pd.DataFrame:
    """Return *path* as a pandas table."""
    return pd.read_csv(Path(path))


def strip_location_modifiers(txt: str) -> str:
    """Delete trailing “(in title)”, “(in abstract)”, … substrings."""
    return txt.split("(")[0].strip()


def explode_keywords(reason: pd.Series) -> pd.Series:
    """
    ➔ One keyword per row for every record whose `exclusion_reason`
    contains a semicolon-separated list.
    """
    return (
        reason.dropna()
        .str.split(";")
        .explode()
        .map(strip_location_modifiers)
        .str.title()
    )

def _coerce_open_access(s: pd.Series) -> pd.Series:
    """Return a clean Boolean Series; treat missing / 0 / False as closed."""
    return (
        s.fillna(False)          # NaN  →  False
         .astype(int)            # 0/1/True/False → 0/1
         .astype(bool)           # 0 → False, 1 → True
    )

# -----------------------------------------------------------------------------#
#                                main function                                 #
# -----------------------------------------------------------------------------#
def make_figure(
    excluded: pd.DataFrame,
    included: pd.DataFrame,
    manual_extra: Mapping[int, int] | None = None,
) -> plt.Figure:
    """Build and return the requested 2 × 2 matplotlib Figure."""
    manual_extra = manual_extra or {}
    included = included.copy()
    excluded = excluded.copy()

    included["is_open_access"] = _coerce_open_access(included["is_open_access"])
    excluded["is_open_access"] = _coerce_open_access(excluded["is_open_access"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax_oa, ax_filter), (ax_kw, ax_manual) = axes

    # a) OA vs decision --------------------------------------------------------
    oa_tab = pd.DataFrame(
        {
            "Open Access": [
                included["is_open_access"].sum(),
                excluded["is_open_access"].sum(),
            ],
            "Closed": [
                (~included["is_open_access"]).sum(),
                (~excluded["is_open_access"]).sum(),
            ],
        },
        index=["Included", "Excluded"],
    )
    sns.heatmap(
        oa_tab,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        ax=ax_oa,
        annot_kws={"fontsize": FONTSIZE},
    )
    ax_oa.set_title("a. Open-access status", loc="left", fontweight="bold")

    # b) automatic filter counts ----------------------------------------------
    filt_counts = (
        excluded["exclusion_filter"]
        .value_counts()
        .sort_values(ascending=True)  # horizontal bar: smallest at bottom
    )
    ax_filter.barh(
        filt_counts.index.str.replace("_", " ").str.title(), filt_counts.values
    )
    ax_filter.set_xlabel("Number of records")
    ax_filter.set_title("b. Automatic filter", loc="left", fontweight="bold")

    # c) top keyword exclusions ------------------------------------------------
    kw_tokens = explode_keywords(
        excluded.loc[excluded["exclusion_filter"] == "keyword", "exclusion_reason"]
    )
    kw_counts = kw_tokens.value_counts().head(20)  # keep figure readable
    ax_kw.barh(kw_counts.index, kw_counts.values, color="steelblue")
    ax_kw.set_xlabel("Number of records")
    ax_kw.set_title("c. Frequent keywords", loc="left", fontweight="bold")
    ax_kw.invert_yaxis()

    # d) manual codes ----------------------------------------------------------
    manual_counts = (
        excluded.loc[excluded["exclusion_filter"] == "manual", "exclusion_reason"]
        .astype(float).astype(int)   # codes are stored as "3.0", not "3"
        .value_counts()
        .add(pd.Series(manual_extra), fill_value=0)
        .reindex(range(1, 8), fill_value=0)
        .astype(int)
    )
    ax_manual.bar(manual_counts.index, manual_counts.values, color="firebrick")
    ax_manual.set_xticks(manual_counts.index, labels=manual_counts.index)
    ax_manual.set_xlabel("Manual code")
    ax_manual.set_ylabel("Number of records")
    ax_manual.set_title("d. Manual screening", loc="left", fontweight="bold")

    # global aesthetics --------------------------------------------------------
    for ax in axes.flat:
        ax.tick_params(labelsize=FONTSIZE - 2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------#
#                        (optional) entry point for CLI                         #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    EXCL = load_csv("data/csvs/info_citations_excluded.csv")
    INCL = load_csv("data/csvs/info_citations_included.csv")
    # any hand-count corrections can be injected here, e.g. {1: 3, 7: 1}
    FIG = make_figure(EXCL, INCL)
    FIG.savefig("data/plots/citation_screening.pdf", dpi=300, bbox_inches="tight")
