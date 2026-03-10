"""
Correlation diagnostics for macro panels:
- Compute correlation matrix
- Plot clustered heatmap (with dendrogram)
- List highly correlated pairs
- Cluster variables by |correlation|

Notes
-----
* Designed for stationary, seasonally-adjusted inputs.
* Pearson correlation is scale-invariant, so standardization is not required.
* Clustering operates on a distance matrix D = 1 - |corr| to group variables
  that move together (positively or negatively).

Dependencies
------------
- numpy
- pandas
- matplotlib
- seaborn (for clustermap)
- scipy (hierarchical clustering utilities)

Recommended usage
-----------------
from correlation_clustering import (
    correlation_heatmap,
    list_high_corr_pairs,
    cluster_variables,
    cluster_table
)

# Compute & plot heatmap
fig = correlation_heatmap(df, method="pearson", min_non_null=20, figsize=(12, 10))
fig.savefig("reports/diagnostics/corr_clustermap.png", dpi=150, bbox_inches="tight")

# Report pairs above threshold
pairs = list_high_corr_pairs(df, threshold=0.95)
print(pairs.head())

# Build clusters and a tidy table
clusters = cluster_variables(df, corr=None, threshold=0.85)
ctbl = cluster_table(clusters)
print(ctbl)
ctbl.to_csv("reports/diagnostics/correlation_clusters.csv", index=False)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


# -----------------------------
# Utilities
# -----------------------------

def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns and drop columns that are entirely NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Numeric-only dataframe with all-NaN columns removed.
    """
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.dropna(axis=1, how="all")
    return num


def _mask_low_support(corr: pd.DataFrame, counts: pd.DataFrame, min_non_null: int) -> pd.DataFrame:
    """Mask correlations whose pairwise non-null count is below threshold.

    Any pair (i, j) with counts[i, j] < min_non_null will be set to NaN in `corr`.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.
    counts : pd.DataFrame
        Pairwise non-missing observation counts for each (i, j).
    min_non_null : int
        Minimum number of overlapping observations required.

    Returns
    -------
    pd.DataFrame
        Masked correlation matrix.
    """
    masked = corr.copy()
    masked[counts < min_non_null] = np.nan
    return masked


def _pairwise_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise non-missing counts for each pair of columns.

    Returns a symmetric DataFrame with the same index/columns as `df`.
    """
    notna = (~df.isna()).astype(int)
    counts = notna.T @ notna
    counts.index = df.columns
    counts.columns = df.columns
    return counts


# -----------------------------
# Public API
# -----------------------------

def correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    min_non_null: int = 10,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "vlag",
    cluster: bool = True,
    absolute: bool = False,
    zorder_columns: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Plot a (clustered) correlation heatmap with optional dendrogram.

    Parameters
    ----------
    df : pd.DataFrame
        Panel of variables (columns) over time (index). Non-numeric columns are ignored.
    method : {"pearson", "spearman"}
        Correlation method.
    min_non_null : int
        Minimum overlapping observations required per pair; otherwise correlation is masked.
    figsize : tuple
        Figure size (width, height).
    cmap : str
        Matplotlib/Seaborn colormap.
    cluster : bool
        If True, use seaborn.clustermap (with dendrogram). If False, plain heatmap.
    absolute : bool
        If True, use |corr| for visualization (helpful to emphasize co-movement regardless of sign).
    zorder_columns : Optional sequence of str
        If given, ensures these columns appear first (useful for pinning important variables).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure. For clustermap, returns the underlying figure from the Grid.
    """
    data = _select_numeric(df)

    # Optional column ordering (e.g., pin target variables first)
    if zorder_columns:
        zorder = [c for c in zorder_columns if c in data.columns]
        rest = [c for c in data.columns if c not in zorder]
        data = data[zorder + rest]

    # Compute correlation and counts
    corr = data.corr(method=method, min_periods=min_non_null)
    counts = _pairwise_counts(data)
    corr = _mask_low_support(corr, counts, min_non_null)

    corr_for_plot = corr.abs() if absolute else corr

    if cluster:
        # Seaborn clustermap handles missing by masking; build a mask
        mask = corr_for_plot.isna()
        # Replace NaNs with 0 for distance transform; they will be masked in the heatmap
        corr_filled = corr_for_plot.fillna(0.0)

        # Use 1 - |corr| distance for clustering so strong negative and positive are both "close"
        dist = 1 - corr_filled.abs().values
        # Convert to condensed distance for linkage
        dist_condensed = squareform(dist, checks=False)
        Z = linkage(dist_condensed, method="average")

        # Create clustermap with our precomputed row/col linkage
        g = sns.clustermap(
            corr_for_plot,
            row_linkage=Z,
            col_linkage=Z,
            cmap=cmap,
            figsize=figsize,
            vmin=-1.0 if not absolute else 0.0,
            vmax=1.0,
            center=0.0 if not absolute else 0.5,
            mask=mask,
            linewidths=0.0,
        )
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.ax_heatmap.set_title(
            f"Correlation heatmap ({'|ρ|' if absolute else 'ρ'}, method={method})\n"
            f"min overlap per pair: {min_non_null}"
        )
        return g.fig

    # Plain heatmap (no dendrogram)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_for_plot,
        cmap=cmap,
        vmin=-1.0 if not absolute else 0.0,
        vmax=1.0,
        center=0.0 if not absolute else 0.5,
        square=True,
        cbar_kws={"label": "|ρ|" if absolute else "ρ"},
    )
    ax.set_title(
        f"Correlation heatmap ({'|ρ|' if absolute else 'ρ'}, method={method})\n"
        f"min overlap per pair: {min_non_null}"
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return fig


def list_high_corr_pairs(
    df: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "pearson",
    min_non_null: int = 10,
    take_abs: bool = True,
) -> pd.DataFrame:
    """Return a tidy table of highly correlated pairs above a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Panel of variables.
    threshold : float
        Correlation threshold for reporting (e.g., 0.95).
    method : str
        Correlation method (pearson/spearman).
    min_non_null : int
        Minimum overlapping observations.
    take_abs : bool
        If True, filter by |ρ| >= threshold; otherwise by ρ >= threshold.

    Returns
    -------
    pd.DataFrame
        Columns: [var_i, var_j, rho, n_overlap]. Each pair appears once (i < j).
    """
    data = _select_numeric(df)
    corr = data.corr(method=method, min_periods=min_non_null)
    counts = _pairwise_counts(data)
    corr = _mask_low_support(corr, counts, min_non_null)

    rows = []
    cols = corr.columns
    for i, ci in enumerate(cols):
        for j in range(i + 1, len(cols)):
            cj = cols[j]
            rho = corr.loc[ci, cj]
            if np.isnan(rho):
                continue
            val = abs(rho) if take_abs else rho
            if val >= threshold:
                rows.append({
                    "var_i": ci,
                    "var_j": cj,
                    "rho": float(rho),
                    "n_overlap": int(counts.loc[ci, cj]),
                })

    out = pd.DataFrame(rows).sort_values(by=["rho"], ascending=False, ignore_index=True)
    return out


def cluster_variables(
    df: pd.DataFrame,
    corr: Optional[pd.DataFrame] = None,
    threshold: float = 0.85,
    method: str = "average",
    min_non_null: int = 10,
) -> List[List[str]]:
    """Cluster variables by absolute correlation using hierarchical clustering.

    Distance is defined as D = 1 - |corr|. We then cut the dendrogram at
    distance t = 1 - threshold so that variables with |corr| >= threshold
    tend to fall into the same cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Panel of variables.
    corr : Optional[pd.DataFrame]
        Precomputed correlation matrix (if available). If None, computed internally.
    threshold : float
        Correlation threshold that defines cluster compactness (e.g., 0.85).
    method : str
        Linkage method for hierarchical clustering (e.g., 'average', 'complete', 'single', 'ward').
        Note: 'ward' is not appropriate with a precomputed distance matrix not in Euclidean form.
    min_non_null : int
        Minimum overlapping observations per pair when computing correlations.

    Returns
    -------
    List[List[str]]
        A list of clusters, each a list of variable names.
    """
    data = _select_numeric(df)

    if corr is None:
        corr = data.corr(method="pearson", min_periods=min_non_null)
        counts = _pairwise_counts(data)
        corr = _mask_low_support(corr, counts, min_non_null)
        corr = corr.fillna(0.0)

    # Build distance from absolute correlation
    dist = 1 - corr.abs().values
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method=method)

    # Cut the tree at distance t = 1 - threshold
    t = 1.0 - float(threshold)
    labels = fcluster(Z, t, criterion="distance")

    # Group columns by their cluster label
    clusters: Dict[int, List[str]] = {}
    cols = list(corr.columns)
    for col, lab in zip(cols, labels):
        clusters.setdefault(lab, []).append(col)

    # Sort clusters by size descending for readability
    cluster_list = sorted(clusters.values(), key=len, reverse=True)
    return cluster_list


def cluster_table(clusters: List[List[str]]) -> pd.DataFrame:
    """Build a tidy table describing clusters (one row per cluster).

    Parameters
    ----------
    clusters : List[List[str]]
        Output of `cluster_variables`.

    Returns
    -------
    pd.DataFrame
        Columns: [cluster_id, size, members].
    """
    rows = []
    for idx, members in enumerate(clusters, start=1):
        rows.append({
            "cluster_id": idx,
            "size": len(members),
            "members": ", ".join(members),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Optional helper: choose representatives per cluster
# -----------------------------

def choose_representatives(
    df: pd.DataFrame,
    clusters: List[List[str]],
    prefer_long_series: bool = True,
) -> List[str]:
    """Select one representative variable per cluster.

    Heuristic: choose the column with the fewest NaNs (longest effective sample)
    when `prefer_long_series=True`; otherwise choose the first column.

    Parameters
    ----------
    df : pd.DataFrame
        Original data (to inspect NaN counts per column).
    clusters : List[List[str]]
        Grouping of columns.
    prefer_long_series : bool
        If True, pick member with minimal NaN count in its column.

    Returns
    -------
    List[str]
        Representative column names (one per cluster).
    """
    data = _select_numeric(df)
    reps: List[str] = []
    for members in clusters:
        if not members:
            continue
        if not prefer_long_series:
            reps.append(members[0])
            continue
        sub = data[members]
        nan_counts = sub.isna().sum(axis=0)
        rep = nan_counts.sort_values().index[0]
        reps.append(rep)
    return reps
