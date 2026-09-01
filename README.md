# Deep Learning for Mammography-based Breast Cancer Analysis

Retrieval, screening and figure code for the review *Deep Learning for
Mammography-based Breast Cancer Analysis: A Dataset-centred Review of Public
Datasets and Recent Studies* (Neurocomputing, under revision).

The review is **dataset-centred**: instead of querying a bibliographic database by
topic, it starts from the descriptor publication of each public mammography dataset
and walks the citation graph forward to find the studies that use it.

```
16 seed publications  ->  1,831 citing records  ->  keyword filter  ->  1,211
                      ->  date + de-duplication ->  119 assessed at full text
                      ->  61 included studies over 14 datasets
```

**The corpus is in this repository.** You do not need an API key, and you do not
need to re-run the search to reproduce the figures.

---

## Quick start

```bash
git clone https://github.com/MarioPasc/Review_Mammography.git
cd Review_Mammography
pip install -r requirements.txt

# Reproduce both figures from the shipped corpus (~10 s, no network)
python scripts/plot/second_plot.py       # -> data/plots/citation_screening.pdf
python scripts/plot/datasets_plot.py     # -> data/plots/dataset_analysis.pdf
```

Run every script from the **repository root**; paths are relative to it.

---

## The data

`data/csvs/` holds the corpus exactly as it was screened for the paper.

| File | Rows | What it is |
|---|---|---|
| `info_citations_included.csv` | 61 | The final corpus. One row per included study. |
| `info_citations_excluded.csv` | 968 | Every excluded record, with the reason. |
| `info_citations_automatic.csv` | 51 | Candidates from the automatic phase. |
| `info_citations_manual_additions.csv` | 68 | Candidates added in the manual/snowball phase. |

The two candidate files together are the 119 records assessed at full text.

Useful columns: `key` (BibTeX key used in the manuscript), `title`, `abstract`,
`year`, `venue`, `doi`, `cited_dataset` (which seed dataset the record was found
through), `evaluation_metrics`, and, in the included file, `inclusion_type`
(`automatic` for 58 studies, `manual` for 3).

Exclusions carry `exclusion_filter` (`keyword`, `no_metrics`, `year_range`,
`manual`) and `exclusion_reason`. For manual exclusions the reason is one of seven
full-text screening codes:

| | | | |
|---|---|---|---|
| 1 no access | 2 book chapter | 3 task not mammography-specific | 4 image count unspecified |
| 5 out of scope | 6 machine learning but not deep learning | 7 private dataset | |

A quick look:

```python
import pandas as pd
inc = pd.read_csv("data/csvs/info_citations_included.csv")
len(inc)                                     # 61

# cited_dataset holds a comma-separated list, so split before counting
inc.cited_dataset.str.split(",").explode().str.strip().value_counts()
# CBIS-DDSM 38, INbreast 23, MIAS 14, DDSM 13, VinDr-Mammo 7, CMMD 5, BCDR 2, OPTIMAM 2, RSNA 1
```

The counts sum to more than 61 because 35 of the 61 studies evaluate on two or
more datasets.

---

## Re-running the search

This queries Semantic Scholar live, so it returns **today's** citation graph, not
the April 2025 snapshot the paper reports. Expect different totals; the corpus
grows.

```bash
python scripts/fetch/papers_semantic_scholar.py   # ~10 min, rate-limited
```

Everything it does is controlled by `scripts/fetch/parameters.yaml`:

- `citation_papers` — the 16 seed identifiers
- `keyword_sets` — three inclusion lists (medical, deep-learning, evaluation
  metric). A record is kept if it matches **at least one term from each** list.
- `exclude_keywords` — 23 terms that reject a record outright
- `date_range` — the 2020–2025 publication window

Two notes on faithfulness to the published run:

- The corpus was built through the `/citations` endpoint, which sends only
  `fields`, `limit` and `offset`. The `min_citation_count` and `open_access_only`
  keys in the YAML belong to a different, unused bulk-search path and **did not
  filter this corpus**.
- The BCDR seed was corrected after the paper's search. At the time of the April
  2025 run, both identifiers labelled BCDR resolved to a different, Canadian
  dataset; the Breast Cancer Digital Repository descriptor was therefore not among
  the seeds. See the commit history for `parameters.yaml`.

---

## Layout

```
scripts/
  fetch/       papers_semantic_scholar.py   corpus construction (main entry point)
               parameters.yaml              seeds, keywords, date window
               citation_analysis.py         per-dataset citation counts over time
  manipulate/  filter_citations.py          keyword / date / duplicate filtering
               exclude_citations.py         applies manual screening decisions
               merge.py  diff.py  order_by.py  parse_to_csv.py
  plot/        second_plot.py               Fig. 5, citation screening (4 panels)
               datasets_plot.py             Fig. 6, per-dataset view (4 panels)
               network_terms.py             keyword co-occurrence network
               cites_year_datasets.py       citations per dataset per year
data/
  csvs/        the corpus (tracked)
  plots/       generated figures (not tracked)
```

`arxiv_fetch.py` and `ncbi_fetch.py` are alternative retrieval backends that were
explored but not used for the published corpus.

---

## Requirements

Python 3.10+ and the packages in `requirements.txt` (requests, pandas, numpy,
matplotlib, seaborn, networkx, PyYAML). No API key is needed: Semantic Scholar's
public endpoints are used unauthenticated, which is rate-limited but sufficient.

## Citing

If you use this corpus, please cite the review. Dataset licences remain with their
original providers; this repository redistributes bibliographic records only, not
mammography images.

## Licence

Code released under the licence in [`LICENSE`](LICENSE).
