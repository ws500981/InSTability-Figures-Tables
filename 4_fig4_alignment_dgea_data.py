#!/usr/bin/env python3
"""
Repeated-clustering DGEA consistency with example-to-run cluster matching.

For each dataset/sample/method:
  1. Choose the first run having the minimum number of clusters.
  2. Build instability bins and the fixed within-bin DGEA pair plan from that run.
  3. For every example cluster and every run, select exactly one run cluster
     by highest Jaccard; break any Jaccard tie by largest intersection.
     Run clusters are never merged.
  4. Apply the fixed example-run pair plan to every run. If both sides select
     the same run cluster, record a matching collision and skip that run/contrast.
  5. Save all candidate Jaccards, selected matches, Jaccard distributions and
     instability correlations, DGEA results, and cross-run DGEA consistency.
"""

from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from itertools import combinations

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.stats import spearmanr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_H5AD_DIR = "./data/ST_datasets/AAA_with_uncert"
OUT_DIR = ("./data/fig4/")

DATASETS = [
    "dlpfc",
    "mouse_brain",
    "mouse_brain_cerebellum",
    "human_breast_cancer",
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "ov_ffpe",
    "coad_ffpe",
]

DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

METHODS = [
    "leiden", "louvain", "sedr", "graphst",
    "bayesspace", "stagate", "spicemix", "sedr_mclust",
]

SKIP_JOBS = set() # skip jobs that are already processed
SKIP_EXISTING = True
USE_VALID_ANNOTATION_MASK = False

N_JOBS_DGEA = int(os.environ.get("N_JOBS_DGEA", "60"))
JOBLIB_BACKEND = "loky"

MIN_CELLS_PER_GROUP = 100
TOP_GENES = 200
TOP_K_LIST = [5, 10, 20, 50, 100, 200]
ADJ_PVAL_MAX = 0.05
MIN_ABS_LOGFC = 0.5
MIN_PCT_EXPR = 0.1

JACCARD_TIE_RTOL = 1e-12
JACCARD_TIE_ATOL = 1e-12

FINE_BINS = [
    "[0.0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)",
    "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)",
    "[0.8,0.9)", "[0.9,1.0]",
]

COARSE_BIN = {
    "[0.0,0.1)": "Stable",
    "[0.1,0.2)": "Stable",
    "[0.2,0.3)": "Modestly Stable",
    "[0.3,0.4)": "Modestly Stable",
    "[0.4,0.5)": "Unstable",
    "[0.5,0.6)": "Unstable",
    "[0.6,0.7)": "Highly Unstable",
    "[0.7,0.8)": "Highly Unstable",
    "[0.8,0.9)": "Highly Unstable",
    "[0.9,1.0]": "Highly Unstable",
}

ID_COLS = [
    "dataset", "sample", "method", "example_run_idx",
    "comparison", "group_a", "group_b",
    "group_a_fine_bin", "group_b_fine_bin",
    "group_a_coarse_bin", "group_b_coarse_bin",
    "contrast_fine_bin", "contrast_coarse_bin",
    "u_a", "u_b", "contrast_u_max", "contrast_u_weighted",
    "n_a_example", "n_b_example",
]

RUN_ALIGNMENT_COLS = [
    "matched_a_run_cluster", "matched_b_run_cluster",
    "group_a_match_jaccard", "group_b_match_jaccard",
    "group_a_match_overlap", "group_b_match_overlap",
    "group_a_match_run_size", "group_b_match_run_size",
    "group_a_match_tie_count", "group_b_match_tie_count",
    "contrast_match_jaccard_min", "contrast_match_jaccard_mean",
]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def write_csv(df, path, index=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    log(f"Saved: {path}")


def sample_id(dataset, sample):
    return dataset if sample is None else f"{dataset}_{sample}"


def file_stem(dataset, sample, method):
    return f"{sample_id(dataset, sample)}_{method}"


def cluster_sort_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def valid_annotation_mask(annotation):
    cleaned = annotation.astype(str).str.strip().str.lower()
    return annotation.notna().to_numpy() & (~cleaned.isin(["", "none", "nan"]).to_numpy())


def build_jobs():
    jobs = []
    for dataset in DATASETS:
        samples = DLPFC_SAMPLES if dataset == "dlpfc" else [None]
        for sample in samples:
            for method in METHODS:
                job = (dataset, sample, method)
                if job in SKIP_JOBS:
                    continue
                if method == "sedr_mclust" and dataset != "dlpfc":
                    continue
                jobs.append(job)
    return jobs


def fine_bin_label(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite instability value: {value}")
    if value < 0:
        return FINE_BINS[0]
    if value >= 1:
        return FINE_BINS[-1]
    return FINE_BINS[min(int(np.floor(value * 10)), 9)]


def identify_example_run(method_results):
    """First run with the minimum number of clusters."""
    return min(
        range(len(method_results)),
        key=lambda i: (len(set(np.asarray(method_results[i]).astype(str))), i),
    )


# -----------------------------------------------------------------------------
# Example-cluster -> run-cluster Jaccard matching
# -----------------------------------------------------------------------------
def match_example_clusters_to_run(example_labels, run_labels):
    """
    Independently select one original run cluster for every example cluster.

    Selection order:
      1. highest Jaccard;
      2. largest intersection among all highest-Jaccard candidates;
      3. deterministic cluster-label order only if both remain tied.

    Returns
    -------
    selected : one row per example cluster
    all_scores : one row per example-cluster/run-cluster candidate pair
    """
    example = np.asarray(example_labels).astype(str)
    run = np.asarray(run_labels).astype(str)
    if len(example) != len(run):
        raise ValueError("Example and run label arrays have different lengths")

    example_clusters = sorted(pd.unique(example), key=cluster_sort_key)
    run_clusters = sorted(pd.unique(run), key=cluster_sort_key)

    tab = pd.crosstab(
        pd.Series(example, name="example_cluster"),
        pd.Series(run, name="run_cluster"),
    ).reindex(index=example_clusters, columns=run_clusters, fill_value=0)

    example_sizes = tab.sum(axis=1)
    run_sizes = tab.sum(axis=0)
    selected_rows = []
    all_rows = []

    for example_cluster in example_clusters:
        candidates = []
        for run_cluster in run_clusters:
            overlap = int(tab.loc[example_cluster, run_cluster])
            example_size = int(example_sizes.loc[example_cluster])
            run_size = int(run_sizes.loc[run_cluster])
            union = example_size + run_size - overlap
            jaccard = overlap / union if union else 0.0

            candidates.append({
                "example_cluster": str(example_cluster),
                "run_cluster": str(run_cluster),
                "jaccard": float(jaccard),
                "overlap": overlap,
                "example_cluster_size": example_size,
                "run_cluster_size": run_size,
                "reference_recall": overlap / example_size if example_size else np.nan,
                "matched_precision": overlap / run_size if run_size else np.nan,
            })

        best_jaccard = max(row["jaccard"] for row in candidates)
        best_jaccard_ties = [
            row for row in candidates
            if np.isclose(
                row["jaccard"], best_jaccard,
                rtol=JACCARD_TIE_RTOL,
                atol=JACCARD_TIE_ATOL,
            )
        ]
        largest_tied_overlap = max(row["overlap"] for row in best_jaccard_ties)
        finalists = [
            row for row in best_jaccard_ties
            if row["overlap"] == largest_tied_overlap
        ]
        chosen = sorted(finalists, key=lambda row: cluster_sort_key(row["run_cluster"]))[0]

        ordered = sorted(
            candidates,
            key=lambda row: (
                -row["jaccard"],
                -row["overlap"],
                cluster_sort_key(row["run_cluster"]),
            ),
        )
        second_best_jaccard = ordered[1]["jaccard"] if len(ordered) > 1 else np.nan
        tied_ids = "|".join(
            row["run_cluster"]
            for row in sorted(best_jaccard_ties, key=lambda row: cluster_sort_key(row["run_cluster"]))
        )

        for rank, row in enumerate(ordered, start=1):
            out = row.copy()
            out.update({
                "jaccard_rank": rank,
                "is_best_jaccard_tie": bool(row in best_jaccard_ties),
                "is_selected": bool(row["run_cluster"] == chosen["run_cluster"]),
            })
            all_rows.append(out)

        selected = chosen.copy()
        selected.update({
            "matched_run_cluster": chosen["run_cluster"],
            "best_jaccard": chosen["jaccard"],
            "second_best_jaccard": float(second_best_jaccard),
            "jaccard_margin": (
                float(chosen["jaccard"] - second_best_jaccard)
                if np.isfinite(second_best_jaccard) else np.nan
            ),
            "n_best_jaccard_ties": len(best_jaccard_ties),
            "best_jaccard_tied_clusters": tied_ids,
            "n_final_overlap_ties": len(finalists),
        })
        selected_rows.append(selected)

    return pd.DataFrame(selected_rows), pd.DataFrame(all_rows)


def build_cluster_instability(example_labels, instability):
    rows = []
    example_labels = np.asarray(example_labels).astype(str)
    for cluster in sorted(pd.unique(example_labels), key=cluster_sort_key):
        mask = example_labels == cluster
        mean_instability = float(np.mean(instability[mask]))
        fine_bin = fine_bin_label(mean_instability)
        rows.append({
            "cluster": str(cluster),
            "mean_instability": mean_instability,
            "n_cells": int(mask.sum()),
            "fine_bin": fine_bin,
            "coarse_bin": COARSE_BIN[fine_bin],
        })
    return pd.DataFrame(rows)


def summarize_cluster_jaccards(best_matches, all_scores, cluster_df, example_idx):
    """Summarize selected-match and all-candidate Jaccard distributions."""
    best_nonexample = best_matches[best_matches["run"] != example_idx].copy()
    all_nonexample = all_scores[all_scores["run"] != example_idx].copy()

    best_summary = (
        best_nonexample.groupby("example_cluster", as_index=False)
        .agg(
            n_nonexample_runs=("run", "nunique"),
            best_jaccard_mean=("best_jaccard", "mean"),
            best_jaccard_median=("best_jaccard", "median"),
            best_jaccard_sd=("best_jaccard", "std"),
            best_jaccard_min=("best_jaccard", "min"),
            best_jaccard_q25=("best_jaccard", lambda x: x.quantile(0.25)),
            best_jaccard_q75=("best_jaccard", lambda x: x.quantile(0.75)),
            best_jaccard_max=("best_jaccard", "max"),
            best_overlap_mean=("overlap", "mean"),
            best_overlap_min=("overlap", "min"),
            matched_run_size_mean=("run_cluster_size", "mean"),
            reference_recall_mean=("reference_recall", "mean"),
            matched_precision_mean=("matched_precision", "mean"),
            tied_best_fraction=("n_best_jaccard_ties", lambda x: np.mean(x > 1)),
            best_jaccard_margin_mean=("jaccard_margin", "mean"),
        )
    )

    candidate_summary = (
        all_nonexample.groupby("example_cluster", as_index=False)
        .agg(
            n_candidate_scores=("jaccard", "size"),
            all_candidate_jaccard_mean=("jaccard", "mean"),
            all_candidate_jaccard_median=("jaccard", "median"),
            all_candidate_jaccard_sd=("jaccard", "std"),
            all_candidate_jaccard_min=("jaccard", "min"),
            all_candidate_jaccard_q25=("jaccard", lambda x: x.quantile(0.25)),
            all_candidate_jaccard_q75=("jaccard", lambda x: x.quantile(0.75)),
            all_candidate_jaccard_max=("jaccard", "max"),
        )
    )

    cluster_summary = (
        cluster_df.rename(columns={"cluster": "example_cluster"})
        .merge(best_summary, on="example_cluster", how="left")
        .merge(candidate_summary, on="example_cluster", how="left")
    )

    best_fine_bin_summary = (
        best_nonexample.groupby("example_fine_bin", as_index=False)
        .agg(
            n_example_clusters=("example_cluster", "nunique"),
            n_best_matches=("best_jaccard", "size"),
            best_jaccard_mean=("best_jaccard", "mean"),
            best_jaccard_median=("best_jaccard", "median"),
            best_jaccard_sd=("best_jaccard", "std"),
            best_jaccard_min=("best_jaccard", "min"),
            best_jaccard_q25=("best_jaccard", lambda x: x.quantile(0.25)),
            best_jaccard_q75=("best_jaccard", lambda x: x.quantile(0.75)),
            best_jaccard_max=("best_jaccard", "max"),
            overlap_mean=("overlap", "mean"),
            reference_recall_mean=("reference_recall", "mean"),
            matched_precision_mean=("matched_precision", "mean"),
            tied_best_fraction=("n_best_jaccard_ties", lambda x: np.mean(x > 1)),
        )
    )

    candidate_fine_bin_summary = (
        all_nonexample.groupby("example_fine_bin", as_index=False)
        .agg(
            n_all_candidate_scores=("jaccard", "size"),
            all_candidate_jaccard_mean=("jaccard", "mean"),
            all_candidate_jaccard_median=("jaccard", "median"),
            all_candidate_jaccard_sd=("jaccard", "std"),
            all_candidate_jaccard_min=("jaccard", "min"),
            all_candidate_jaccard_q25=("jaccard", lambda x: x.quantile(0.25)),
            all_candidate_jaccard_q75=("jaccard", lambda x: x.quantile(0.75)),
            all_candidate_jaccard_max=("jaccard", "max"),
            all_candidate_overlap_mean=("overlap", "mean"),
        )
    )
    fine_bin_summary = best_fine_bin_summary.merge(
        candidate_fine_bin_summary,
        on="example_fine_bin",
        how="outer",
    )
    fine_bin_summary["example_fine_bin"] = pd.Categorical(
        fine_bin_summary["example_fine_bin"], categories=FINE_BINS, ordered=True
    )
    fine_bin_summary = fine_bin_summary.sort_values("example_fine_bin").reset_index(drop=True)

    correlation_rows = []
    for jaccard_col in [
        "best_jaccard_mean", "best_jaccard_median", "best_jaccard_min"
    ]:
        valid = cluster_summary[["mean_instability", jaccard_col]].dropna()
        if len(valid) >= 3:
            result = spearmanr(valid["mean_instability"], valid[jaccard_col])
            rho = float(result.statistic) if np.isfinite(result.statistic) else np.nan
            pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
        else:
            rho = pvalue = np.nan
        correlation_rows.append({
            "jaccard_summary": jaccard_col,
            "n_example_clusters": len(valid),
            "spearman_rho": rho,
            "spearman_pvalue": pvalue,
        })

    return cluster_summary, fine_bin_summary, pd.DataFrame(correlation_rows)


# -----------------------------------------------------------------------------
# Fixed example-run contrast plan
# -----------------------------------------------------------------------------
def build_contrast_plan(cluster_df):
    meta = cluster_df.set_index("cluster").to_dict("index")
    clusters = sorted(cluster_df["cluster"].astype(str), key=cluster_sort_key)
    rows = []

    for group_a, group_b in combinations(clusters, 2):
        a = meta[group_a]
        b = meta[group_b]
        if a["n_cells"] < MIN_CELLS_PER_GROUP or b["n_cells"] < MIN_CELLS_PER_GROUP:
            continue
        if a["fine_bin"] != b["fine_bin"]:
            continue

        n_a, n_b = int(a["n_cells"]), int(b["n_cells"])
        u_a, u_b = float(a["mean_instability"]), float(b["mean_instability"])
        fine_bin = a["fine_bin"]
        rows.append({
            "comparison": f"{group_a}vs{group_b}",
            "group_a": group_a,
            "group_b": group_b,
            "group_a_fine_bin": fine_bin,
            "group_b_fine_bin": fine_bin,
            "group_a_coarse_bin": COARSE_BIN[fine_bin],
            "group_b_coarse_bin": COARSE_BIN[fine_bin],
            "contrast_fine_bin": fine_bin,
            "contrast_coarse_bin": COARSE_BIN[fine_bin],
            "u_a": u_a,
            "u_b": u_b,
            "contrast_u_max": max(u_a, u_b),
            "contrast_u_weighted": (u_a * n_a + u_b * n_b) / (n_a + n_b),
            "n_a_example": n_a,
            "n_b_example": n_b,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# DGEA
# -----------------------------------------------------------------------------
def preprocess_for_dgea(adata_in, counts_layer="counts"):
    adata = adata_in.copy()
    if counts_layer in adata.layers:
        adata.X = adata.layers[counts_layer].copy()
    else:
        log(f"[WARN] layer '{counts_layer}' not found; using current adata.X")
    adata.uns.pop("log1p", None)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.raw = None
    return adata


def pct_expr_for_genes(adata, group_col, group_label, genes):
    genes = pd.Index(pd.Series(genes).astype(str))
    indices = adata.var_names.get_indexer(genes)
    valid = indices >= 0
    out = np.full(len(genes), np.nan)
    if valid.any():
        mask = adata.obs[group_col].astype(str).eq(str(group_label)).to_numpy()
        matrix = adata[mask, indices[valid]].X
        out[valid] = np.asarray((matrix > 0).mean(axis=0)).ravel()
    return pd.Series(out, index=genes)


def one_vs_one_dgea(adata, original_labels, run_group_a, run_group_b):
    labels = np.asarray(original_labels).astype(str)
    keep = np.isin(labels, [str(run_group_a), str(run_group_b)])
    adata_pair = adata[keep].copy()
    pair_labels = labels[keep]
    adata_pair.obs["_pair"] = pd.Categorical(
        pair_labels,
        categories=[str(run_group_a), str(run_group_b)],
        ordered=True,
    )

    sc.tl.rank_genes_groups(
        adata_pair,
        groupby="_pair",
        groups=[str(run_group_a)],
        reference=str(run_group_b),
        method="wilcoxon",
        rankby_abs=True,
        tie_correct=True,
        use_raw=False,
        key_added="_dgea_pair",
    )
    result = adata_pair.uns["_dgea_pair"]
    group = str(run_group_a)
    genes = np.asarray(result["names"][group]).astype(str)

    raw = pd.DataFrame({
        "gene": genes,
        "raw_rank": np.arange(1, len(genes) + 1),
        "score": np.asarray(result["scores"][group], dtype=float),
        "logfc": np.asarray(result["logfoldchanges"][group], dtype=float),
        "pval": np.asarray(result["pvals"][group], dtype=float),
        "pval_adj": np.asarray(result["pvals_adj"][group], dtype=float),
    })
    raw["pct_expr_a"] = pct_expr_for_genes(
        adata_pair, "_pair", run_group_a, raw["gene"]
    ).to_numpy()
    raw["pct_expr_b"] = pct_expr_for_genes(
        adata_pair, "_pair", run_group_b, raw["gene"]
    ).to_numpy()
    raw["pct_expr_max"] = raw[["pct_expr_a", "pct_expr_b"]].max(axis=1)
    raw["abs_score"] = raw["score"].abs()
    raw["abs_logfc"] = raw["logfc"].abs()

    filtered = raw.dropna(
        subset=["score", "logfc", "pval_adj", "pct_expr_max", "abs_logfc"]
    )
    filtered = filtered[
        (filtered["pval_adj"] < ADJ_PVAL_MAX)
        & (filtered["pct_expr_max"] > MIN_PCT_EXPR)
        & (filtered["abs_logfc"] >= MIN_ABS_LOGFC)
    ]
    filtered = (
        filtered.sort_values("abs_score", ascending=False)
        .head(TOP_GENES)
        .reset_index(drop=True)
    )
    filtered["rank"] = np.arange(1, len(filtered) + 1)
    return filtered, len(raw)


def run_one_run_dgea(
    adata_de,
    original_labels,
    run_matches,
    pair_plan,
    dataset,
    sample,
    method,
    example_idx,
    run_idx,
):
    match_lookup = run_matches.set_index("example_cluster")
    label_counts = pd.Series(np.asarray(original_labels).astype(str)).value_counts()
    dgea_rows = []
    status_rows = []

    for _, pair in pair_plan.iterrows():
        group_a = str(pair["group_a"])
        group_b = str(pair["group_b"])
        match_a = match_lookup.loc[group_a]
        match_b = match_lookup.loc[group_b]
        run_group_a = str(match_a["matched_run_cluster"])
        run_group_b = str(match_b["matched_run_cluster"])
        collision = run_group_a == run_group_b
        n_a = int(label_counts.get(run_group_a, 0))
        n_b = int(label_counts.get(run_group_b, 0))

        alignment = {
            "matched_a_run_cluster": run_group_a,
            "matched_b_run_cluster": run_group_b,
            "group_a_match_jaccard": float(match_a["best_jaccard"]),
            "group_b_match_jaccard": float(match_b["best_jaccard"]),
            "group_a_match_overlap": int(match_a["overlap"]),
            "group_b_match_overlap": int(match_b["overlap"]),
            "group_a_match_run_size": int(match_a["run_cluster_size"]),
            "group_b_match_run_size": int(match_b["run_cluster_size"]),
            "group_a_match_tie_count": int(match_a["n_best_jaccard_ties"]),
            "group_b_match_tie_count": int(match_b["n_best_jaccard_ties"]),
            "contrast_match_jaccard_min": float(
                min(match_a["best_jaccard"], match_b["best_jaccard"])
            ),
            "contrast_match_jaccard_mean": float(
                np.mean([match_a["best_jaccard"], match_b["best_jaccard"]])
            ),
        }
        has_min_cells = (
            not collision
            and n_a >= MIN_CELLS_PER_GROUP
            and n_b >= MIN_CELLS_PER_GROUP
        )

        status = {column: pair[column] for column in pair_plan.columns}
        status.update({
            "dataset": dataset,
            "sample": sample,
            "method": method,
            "example_run_idx": int(example_idx),
            "run": int(run_idx),
            "is_example_run": bool(run_idx == example_idx),
            **alignment,
            "matching_collision": bool(collision),
            "n_a_run": n_a,
            "n_b_run": n_b,
            "has_min_cells": bool(has_min_cells),
            "n_raw_genes": 0,
            "has_raw_dgea": False,
            "n_de_genes": 0,
            "has_dgea": False,
            "status_reason": (
                "matching_collision" if collision
                else "too_few_cells" if not has_min_cells
                else "pending"
            ),
        })

        if not has_min_cells:
            status_rows.append(status)
            continue

        filtered, n_raw_genes = one_vs_one_dgea(
            adata_de, original_labels, run_group_a, run_group_b
        )
        status["n_raw_genes"] = n_raw_genes
        status["has_raw_dgea"] = n_raw_genes > 0
        status["n_de_genes"] = len(filtered)
        status["has_dgea"] = len(filtered) > 0
        status["status_reason"] = "ok" if len(filtered) else "no_filtered_dgea"
        status_rows.append(status)

        if len(filtered) == 0:
            continue

        filtered = filtered.copy()
        for column in pair_plan.columns:
            filtered[column] = pair[column]
        filtered["dataset"] = dataset
        filtered["sample"] = sample
        filtered["method"] = method
        filtered["example_run_idx"] = int(example_idx)
        filtered["run"] = int(run_idx)
        filtered["is_example_run"] = bool(run_idx == example_idx)
        for column, value in alignment.items():
            filtered[column] = value
        dgea_rows.append(filtered)

    return (
        pd.concat(dgea_rows, ignore_index=True) if dgea_rows else pd.DataFrame(),
        pd.DataFrame(status_rows),
    )


def run_dgea_for_all_runs(
    adata_de,
    method_results,
    matches_by_run,
    pair_plan,
    dataset,
    sample,
    method,
    example_idx,
):
    n_jobs = min(N_JOBS_DGEA, len(method_results))
    results = Parallel(n_jobs=n_jobs, backend=JOBLIB_BACKEND, verbose=10)(
        delayed(run_one_run_dgea)(
            adata_de,
            labels,
            matches_by_run[run_idx],
            pair_plan,
            dataset,
            sample,
            method,
            example_idx,
            run_idx,
        )
        for run_idx, labels in enumerate(method_results)
    )
    dgea_parts = [result[0] for result in results if len(result[0])]
    status_parts = [result[1] for result in results if len(result[1])]
    return (
        pd.concat(dgea_parts, ignore_index=True) if dgea_parts else pd.DataFrame(),
        pd.concat(status_parts, ignore_index=True) if status_parts else pd.DataFrame(),
    )


def summarize_dgea_status(status):
    if status.empty:
        return pd.DataFrame()

    summary = (
        status.groupby(ID_COLS, dropna=False, as_index=False)
        .agg(
            n_runs_total=("run", "nunique"),
            n_runs_min_cells=("has_min_cells", "sum"),
            n_runs_with_raw_dgea=("has_raw_dgea", "sum"),
            n_runs_with_dgea=("has_dgea", "sum"),
            valid_min_cell_fraction=("has_min_cells", "mean"),
            valid_raw_dgea_fraction=("has_raw_dgea", "mean"),
            valid_dgea_fraction=("has_dgea", "mean"),
            n_a_run_mean=("n_a_run", "mean"),
            n_b_run_mean=("n_b_run", "mean"),
            n_raw_genes_mean=("n_raw_genes", "mean"),
            n_de_genes_mean=("n_de_genes", "mean"),
        )
    )

    nonexample = status[~status["is_example_run"]]
    nonexample_summary = (
        nonexample.groupby(ID_COLS, dropna=False, as_index=False)
        .agg(
            n_nonexample_runs=("run", "nunique"),
            n_matching_collisions=("matching_collision", "sum"),
            matching_collision_fraction=("matching_collision", "mean"),
            group_a_match_jaccard_mean=("group_a_match_jaccard", "mean"),
            group_b_match_jaccard_mean=("group_b_match_jaccard", "mean"),
            contrast_match_jaccard_min_mean=("contrast_match_jaccard_min", "mean"),
            contrast_match_jaccard_mean_mean=("contrast_match_jaccard_mean", "mean"),
        )
    )
    return summary.merge(nonexample_summary, on=ID_COLS, how="left")


# -----------------------------------------------------------------------------
# Cross-run DGEA consistency relative to the example run
# -----------------------------------------------------------------------------
def topk(frame, k):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["gene", "rank", "logfc"])
    return frame.sort_values("rank").head(k).drop_duplicates("gene").copy()


def gene_agreement(example_df, run_df):
    base = {
        "jaccard": np.nan,
        "logfc_spearman": np.nan,
        "sign_consistency": np.nan,
        "sign_flip_fraction": np.nan,
        "n_sign_flips": 0,
        "mean_abs_logfc_diff": np.nan,
        "n_shared": 0,
        "n_genes_1": len(example_df),
        "n_genes_2": len(run_df),
    }
    if example_df.empty or run_df.empty:
        return base

    genes_example = set(example_df["gene"])
    genes_run = set(run_df["gene"])
    shared = genes_example & genes_run
    union = genes_example | genes_run
    out = base.copy()
    out["jaccard"] = len(shared) / len(union) if union else np.nan
    out["n_shared"] = len(shared)

    if shared:
        merged = example_df[["gene", "logfc"]].merge(
            run_df[["gene", "logfc"]], on="gene", suffixes=("_1", "_2")
        )
        out["mean_abs_logfc_diff"] = float(
            np.mean(np.abs(merged["logfc_1"] - merged["logfc_2"]))
        )
        sign_1 = np.sign(merged["logfc_1"].to_numpy(float))
        sign_2 = np.sign(merged["logfc_2"].to_numpy(float))
        nonzero = (sign_1 != 0) & (sign_2 != 0)
        if nonzero.any():
            flips = sign_1[nonzero] != sign_2[nonzero]
            out["n_sign_flips"] = int(flips.sum())
            out["sign_consistency"] = float(np.mean(~flips))
            out["sign_flip_fraction"] = float(np.mean(flips))
        if len(shared) >= 3:
            rho = spearmanr(merged["logfc_1"], merged["logfc_2"]).statistic
            out["logfc_spearman"] = float(rho) if np.isfinite(rho) else np.nan
    return out


def penalized_rank_difference(example_df, run_df, k):
    if example_df.empty and run_df.empty:
        return {
            "rank_abs_diff_vs_example_mean": np.nan,
            "rank_abs_diff_vs_example_norm_mean": np.nan,
            "n_rank_union": 0,
            "n_example_genes_missing_in_run": 0,
            "n_run_genes_missing_in_example": 0,
        }

    example_ranks = example_df.drop_duplicates("gene").set_index("gene")["rank"].astype(float)
    run_ranks = run_df.drop_duplicates("gene").set_index("gene")["rank"].astype(float)
    genes = example_ranks.index.union(run_ranks.index)
    example_aligned = example_ranks.reindex(genes)
    run_aligned = run_ranks.reindex(genes)
    missing_rank = float(k + 1)
    differences = np.abs(
        example_aligned.fillna(missing_rank).to_numpy()
        - run_aligned.fillna(missing_rank).to_numpy()
    )
    return {
        "rank_abs_diff_vs_example_mean": float(np.mean(differences)),
        "rank_abs_diff_vs_example_norm_mean": float(np.mean(differences) / k),
        "n_rank_union": len(genes),
        "n_example_genes_missing_in_run": int(
            run_aligned.loc[example_ranks.index].isna().sum()
        ),
        "n_run_genes_missing_in_example": int(
            example_aligned.loc[run_ranks.index].isna().sum()
        ),
    }


def cross_run_metrics(dgea_long, status_summary):
    if dgea_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_rows = []
    for _, contrast in dgea_long.groupby("comparison", sort=False):
        example_idx = int(contrast["example_run_idx"].iloc[0])
        runs = sorted(contrast["run"].unique())
        if example_idx not in runs:
            continue
        by_run = {run: contrast[contrast["run"] == run] for run in runs}
        topk_cache = {
            (run, k): topk(by_run[run], k)
            for run in runs
            for k in TOP_K_LIST
        }
        example_meta = contrast.iloc[0][ID_COLS].to_dict()

        for run in runs:
            if run == example_idx:
                continue
            run_meta = by_run[run].iloc[0]
            for k in TOP_K_LIST:
                example_top = topk_cache[(example_idx, k)]
                run_top = topk_cache[(run, k)]
                gene_metrics = gene_agreement(example_top, run_top)
                rank_metrics = penalized_rank_difference(example_top, run_top, k)
                detail_rows.append({
                    **example_meta,
                    "top_k": k,
                    "example_run": example_idx,
                    "run": int(run),
                    **{column: run_meta[column] for column in RUN_ALIGNMENT_COLS},
                    **gene_metrics,
                    **rank_metrics,
                })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summary = (
        detail.groupby(ID_COLS + ["top_k"], dropna=False, as_index=False)
        .agg(
            n_run_comparisons=("run", "nunique"),
            jaccard_mean=("jaccard", "mean"),
            logfc_spearman_mean=("logfc_spearman", "mean"),
            sign_consistency_mean=("sign_consistency", "mean"),
            sign_flip_fraction_mean=("sign_flip_fraction", "mean"),
            n_sign_flips_mean=("n_sign_flips", "mean"),
            mean_abs_logfc_diff=("mean_abs_logfc_diff", "mean"),
            n_shared_mean=("n_shared", "mean"),
            n_genes_1_mean=("n_genes_1", "mean"),
            n_genes_2_mean=("n_genes_2", "mean"),
            rank_abs_diff_vs_example_mean=("rank_abs_diff_vs_example_mean", "mean"),
            rank_abs_diff_vs_example_norm_mean=("rank_abs_diff_vs_example_norm_mean", "mean"),
            n_rank_union_mean=("n_rank_union", "mean"),
            n_example_genes_missing_in_run_mean=(
                "n_example_genes_missing_in_run", "mean"
            ),
            n_run_genes_missing_in_example_mean=(
                "n_run_genes_missing_in_example", "mean"
            ),
            group_a_match_jaccard_valid_dgea_mean=(
                "group_a_match_jaccard", "mean"
            ),
            group_b_match_jaccard_valid_dgea_mean=(
                "group_b_match_jaccard", "mean"
            ),
            contrast_match_jaccard_min_valid_dgea_mean=(
                "contrast_match_jaccard_min", "mean"
            ),
            contrast_match_jaccard_valid_dgea_mean=(
                "contrast_match_jaccard_mean", "mean"
            ),
        )
    )
    if status_summary is not None and not status_summary.empty:
        summary = summary.merge(status_summary, on=ID_COLS, how="left")
    return detail, summary


# -----------------------------------------------------------------------------
# One dataset/sample/method job
# -----------------------------------------------------------------------------
def process_job(dataset, sample, method):
    started = time.time()
    stem = file_stem(dataset, sample, method)
    final_path = os.path.join(OUT_DIR, f"{stem}_contrast_dgea_consistency.csv")
    if SKIP_EXISTING and os.path.exists(final_path):
        log(f"[SKIP EXISTING] {final_path}")
        return

    input_path = os.path.join(INPUT_H5AD_DIR, f"{sample_id(dataset, sample)}.h5ad")
    log(f"[START] {stem}")
    adata0 = sc.read_h5ad(input_path)
    result_key = f"{method}_results"
    uncertainty_key = f"{method}_uncertainty"
    if result_key not in adata0.uns or uncertainty_key not in adata0.obs:
        log(f"[SKIP] missing {result_key} or {uncertainty_key}")
        return

    if USE_VALID_ANNOTATION_MASK and "annotation" in adata0.obs:
        keep = valid_annotation_mask(adata0.obs["annotation"])
    else:
        keep = np.ones(adata0.n_obs, dtype=bool)

    method_results = []
    for run_idx, labels in enumerate(adata0.uns[result_key]):
        labels = np.asarray(labels)
        if len(labels) != adata0.n_obs:
            raise ValueError(f"{stem}: run {run_idx} length mismatch")
        method_results.append(labels[keep].astype(str))

    adata = adata0[keep].copy()
    instability = adata0.obs[uncertainty_key].to_numpy(float)[keep]
    example_idx = identify_example_run(method_results)
    example_labels = method_results[example_idx]
    log(
        f"Example run={example_idx}; "
        f"n_clusters={len(pd.unique(example_labels))}; n_runs={len(method_results)}"
    )

    cluster_df = build_cluster_instability(example_labels, instability)
    pair_plan = build_contrast_plan(cluster_df)
    cluster_meta = cluster_df.rename(columns={
        "cluster": "example_cluster",
        "mean_instability": "example_mean_instability",
        "n_cells": "example_n_cells",
        "fine_bin": "example_fine_bin",
        "coarse_bin": "example_coarse_bin",
    })

    matches_by_run = []
    selected_parts = []
    all_score_parts = []
    for run_idx, labels in enumerate(method_results):
        selected, all_scores = match_example_clusters_to_run(example_labels, labels)
        for frame in (selected, all_scores):
            frame["run"] = run_idx
            frame["example_run_idx"] = example_idx
            frame["is_example_run"] = run_idx == example_idx
        selected = selected.merge(cluster_meta, on="example_cluster", how="left")
        all_scores = all_scores.merge(cluster_meta, on="example_cluster", how="left")
        matches_by_run.append(selected)
        selected_parts.append(selected)
        all_score_parts.append(all_scores)

    best_matches = pd.concat(selected_parts, ignore_index=True)
    all_scores = pd.concat(all_score_parts, ignore_index=True)
    cluster_jaccard_summary, fine_bin_jaccard_summary, correlation_df = (
        summarize_cluster_jaccards(
            best_matches, all_scores, cluster_df, example_idx
        )
    )

    # Keep the computational pair plan free of job-level metadata; add metadata
    # only to the copy written to disk. This avoids assigning the same fields
    # again inside the DGEA loop while preserving the contrast-plan CSV schema.
    pair_plan_out = pair_plan.copy()
    for frame in [cluster_df, pair_plan_out, best_matches, all_scores,
                  cluster_jaccard_summary, fine_bin_jaccard_summary, correlation_df]:
        frame.insert(0, "method", method)
        frame.insert(0, "sample", sample)
        frame.insert(0, "dataset", dataset)

    write_csv(cluster_df, os.path.join(OUT_DIR, f"{stem}_cluster_instability.csv"))
    write_csv(pair_plan_out, os.path.join(OUT_DIR, f"{stem}_contrast_plan.csv"))
    write_csv(best_matches, os.path.join(OUT_DIR, f"{stem}_best_cluster_matches.csv"))
    write_csv(
        all_scores,
        os.path.join(OUT_DIR, f"{stem}_all_cluster_jaccard_scores.csv.gz"),
    )
    write_csv(
        cluster_jaccard_summary,
        os.path.join(OUT_DIR, f"{stem}_cluster_jaccard_summary.csv"),
    )
    write_csv(
        fine_bin_jaccard_summary,
        os.path.join(OUT_DIR, f"{stem}_jaccard_by_fine_bin_summary.csv"),
    )
    write_csv(
        correlation_df,
        os.path.join(OUT_DIR, f"{stem}_instability_jaccard_correlation.csv"),
    )

    if pair_plan.empty:
        log(f"[SKIP DGEA] no valid within-bin cluster pairs for {stem}")
        return

    adata_de = preprocess_for_dgea(adata)
    dgea_long, dgea_status = run_dgea_for_all_runs(
        adata_de,
        method_results,
        matches_by_run,
        pair_plan,
        dataset,
        sample,
        method,
        example_idx,
    )
    status_summary = summarize_dgea_status(dgea_status)

    write_csv(dgea_long, os.path.join(OUT_DIR, f"{stem}_dgea_long_top200.csv"))
    write_csv(dgea_status, os.path.join(OUT_DIR, f"{stem}_dgea_status.csv"))
    write_csv(
        status_summary,
        os.path.join(OUT_DIR, f"{stem}_dgea_status_summary.csv"),
    )

    pair_detail, contrast_summary = cross_run_metrics(dgea_long, status_summary)
    write_csv(
        pair_detail,
        os.path.join(OUT_DIR, f"{stem}_pairwise_dgea_consistency.csv"),
    )
    write_csv(contrast_summary, final_path)

    del adata0, adata, adata_de, dgea_long, dgea_status
    gc.collect()
    log(f"[DONE] {stem} in {(time.time() - started) / 60:.2f} min")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = build_jobs()
    log(f"Running {len(jobs)} jobs; output={OUT_DIR}")
    for index, job in enumerate(jobs, start=1):
        log(f"Job {index}/{len(jobs)}")
        process_job(*job)


if __name__ == "__main__":
    main()
