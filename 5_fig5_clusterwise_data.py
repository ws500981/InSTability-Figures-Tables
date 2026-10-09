#!/usr/bin/env python3

import gc
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path(
    "./data/ST_datasets/AAA_with_uncert"
)

RESULTS_ROOT = Path(
    "./data/clustering_results"
)

INSTABILITY_ROOT = Path(
    "./data/1-nested-non-sampling"
)

OUT_CSV = Path(
    "./data/fig5/clusterwise_metrics_by_number_of_runs_new.csv"
)

RUN_COUNTS = list(range(5, 51, 5))
MAX_RUNS = 50

# Options:
#   "example_50" = example run selected using all 50 runs
#   "example_k"  = example run selected using the first k runs
#   "run_0"      = always use run 0
REFERENCE_MODE = "example_50"

ENTROPY_NORMALIZE = False

JACCARD_TIE_RTOL = 1e-12
JACCARD_TIE_ATOL = 1e-12

DATASETS = [
    "dlpfc",
    "mouse_brain",
    "mouse_brain_cerebellum",
    "human_breast_cancer",
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "ov_ffpe",
    "coad_ffpe",
    "Visium_HD_Human_Colon_Cancer",
    "coad_ffpe_full",
    "ov_ffpe_full",
]

DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

METHODS = [
    "leiden",
    "louvain",
    "bayesspace",
    "graphst",
    "stagate",
    "spicemix",
    "sedr",
    "sedr_mclust",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def cluster_sort_key(value):
    """Sort numeric labels numerically and all other labels lexically."""
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def match_cluster_to_annotation(
    cluster_mask,
    valid_annotation,
    annotation,
    annotation_sizes,
):
    """
    Match one predicted cluster to one manual annotation.

    Selection order:
      1. highest Jaccard;
      2. largest overlap among tied highest-Jaccard candidates;
      3. deterministic annotation-label order if still tied.
    """
    eval_mask = cluster_mask & valid_annotation

    if not eval_mask.any():
        return None, np.nan, np.nan, np.nan

    predicted_size = int(eval_mask.sum())
    labels, overlaps = np.unique(
        annotation[eval_mask],
        return_counts=True,
    )

    candidates = []
    for label, overlap in zip(labels, overlaps):
        label = str(label)
        overlap = int(overlap)
        reference_size = int(annotation_sizes[label])
        union = predicted_size + reference_size - overlap
        jaccard = overlap / union if union else 0.0

        candidates.append({
            "label": label,
            "overlap": overlap,
            "reference_size": reference_size,
            "jaccard": float(jaccard),
        })

    best_jaccard = max(row["jaccard"] for row in candidates)
    best_jaccard_ties = [
        row
        for row in candidates
        if np.isclose(
            row["jaccard"],
            best_jaccard,
            rtol=JACCARD_TIE_RTOL,
            atol=JACCARD_TIE_ATOL,
        )
    ]
    largest_tied_overlap = max(
        row["overlap"] for row in best_jaccard_ties
    )
    finalists = [
        row
        for row in best_jaccard_ties
        if row["overlap"] == largest_tied_overlap
    ]
    chosen = sorted(
        finalists,
        key=lambda row: cluster_sort_key(row["label"]),
    )[0]

    precision = chosen["overlap"] / predicted_size
    recall = chosen["overlap"] / chosen["reference_size"]

    return chosen["label"], chosen["jaccard"], precision, recall

def example_run_index(results):
    """First run with the minimum number of clusters."""
    return min(
        range(len(results)),
        key=lambda i: (len(np.unique(results[i])), i),
    )


def reference_run_index(results, k):
    if REFERENCE_MODE == "run_0":
        return 0

    if REFERENCE_MODE == "example_50":
        return example_run_index(results[:MAX_RUNS])

    if REFERENCE_MODE == "example_k":
        return example_run_index(results[:k])

    raise ValueError(
        "REFERENCE_MODE must be 'example_50', 'example_k', or 'run_0'."
    )


def load_clustering_results(dataset, sample, method):
    directory = RESULTS_ROOT / dataset / method

    if dataset == "dlpfc":
        filename = (
            f"{sample}_{method}_results_{MAX_RUNS}iterations.csv"
            if method == "bayesspace"
            else f"{sample}_{method}_results_{MAX_RUNS}iterations.pkl"
        )
    else:
        filename = (
            f"{method}_results_{MAX_RUNS}iterations.csv"
            if method == "bayesspace"
            else f"{method}_results_{MAX_RUNS}iterations.pkl"
        )

    path = directory / filename

    if method == "bayesspace":
        frame = pd.read_csv(path)
        results = [frame[column].to_numpy() for column in frame.columns]
    else:
        results = joblib.load(path)

    results = [np.asarray(labels) for labels in results[:MAX_RUNS]]

    if len(results) < MAX_RUNS:
        raise ValueError(
            f"{path} contains {len(results)} runs; "
            f"{MAX_RUNS} are required."
        )

    if dataset == "dlpfc" and method == "sedr_mclust":
        idx = joblib.load(
            RESULTS_ROOT
            / "dlpfc"
            / "sedr_mclust"
            / f"{sample}_idx.pkl"
        )
        results = [labels[idx] for labels in results]

    return results


def load_instability(dataset, sample, method, k):
    stem = f"{dataset}{'_' + sample if sample else ''}"

    path = (
        INSTABILITY_ROOT
        / dataset
        / method
        / f"{stem}_{method}_uncertainty_{k}runs.pkl"
    )

    result = joblib.load(path)
    return np.asarray(result["spot_uncertainty"], dtype=float)


def expression_embedding(adata):
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)

    n_pcs = min(50, tmp.n_obs - 1, tmp.n_vars - 1)
    if n_pcs < 20:
        return None

    X = tmp.X

    try:
        X_pca = PCA(
            n_components=n_pcs,
            svd_solver="arpack",
            random_state=0,
        ).fit_transform(X)
    except (TypeError, ValueError):
        if sparse.issparse(X):
            X = X.toarray()

        X_pca = PCA(
            n_components=n_pcs,
            svd_solver="arpack",
            random_state=0,
        ).fit_transform(np.asarray(X, dtype=np.float64))

    X20 = np.asarray(X_pca[:, :20], dtype=np.float64)
    norms = np.linalg.norm(X20, axis=1)
    norms[norms == 0] = 1.0

    del tmp
    return X20 / norms[:, None]


def entropy_from_probabilities(probabilities, eps=1e-12):
    probabilities = np.asarray(probabilities, dtype=np.float64)

    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        row_sums,
        out=np.zeros_like(probabilities),
        where=row_sums > 0,
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities + eps),
        axis=1,
    )

    if ENTROPY_NORMALIZE and probabilities.shape[1] > 1:
        entropy /= np.log(probabilities.shape[1])

    return entropy


def load_bayesspace_entropy(dataset, sample, run_index, n_spots):
    filename = (
        f"{sample}_bayesspace_softlabels_50iterations.csv"
        if sample else
        "bayesspace_softlabels_50iterations.csv"
    )

    path = RESULTS_ROOT / dataset / "bayesspace" / filename

    if not path.exists():
        return np.full(n_spots, np.nan)

    columns = pd.read_csv(path, nrows=0).columns
    selected = []

    # Supports either zero-based or one-based iteration names.
    for iteration in (run_index, run_index + 1):
        selected = [
            column
            for column in columns
            if re.match(
                rf"^iter_{iteration}_cluster_\d+$",
                str(column),
            )
        ]

        if selected:
            selected.sort(
                key=lambda column: int(
                    re.search(r"cluster_(\d+)$", str(column)).group(1)
                )
            )
            break

    if not selected:
        return np.full(n_spots, np.nan)

    probabilities = pd.read_csv(
        path,
        usecols=selected,
    ).to_numpy(dtype=np.float64)

    if probabilities.shape[0] != n_spots:
        return np.full(n_spots, np.nan)

    return entropy_from_probabilities(probabilities)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    rows = []

    samples = [
        (dataset, sample)
        for dataset in DATASETS
        for sample in (
            DLPFC_SAMPLES if dataset == "dlpfc" else [None]
        )
    ]

    for dataset, sample in samples:
        dataset_sample = (
            f"{dataset}_{sample}" if sample else dataset
        )
        h5ad_path = DATA_DIR / f"{dataset_sample}.h5ad"

        if not h5ad_path.exists():
            print(f"[skip] Missing AnnData: {h5ad_path}")
            continue

        print(f"[start] {dataset_sample}")
        adata = sc.read_h5ad(h5ad_path)

        try:
            X_pca = expression_embedding(adata)
        except Exception as error:
            print(f"[expression metrics NA] {error}")
            X_pca = None

        if "annotation" in adata.obs:
            annotation_series = adata.obs["annotation"]
            valid_annotation = annotation_series.notna().to_numpy()
            annotation = annotation_series.astype(str).to_numpy()
            labels, counts = np.unique(
                annotation[valid_annotation],
                return_counts=True,
            )
            annotation_sizes = {
                str(label): int(count)
                for label, count in zip(labels, counts)
            }
        else:
            valid_annotation = np.zeros(adata.n_obs, dtype=bool)
            annotation = np.full(adata.n_obs, "", dtype=object)
            annotation_sizes = {}

        for method in METHODS:
            try:
                results = load_clustering_results(
                    dataset,
                    sample,
                    method,
                )
            except FileNotFoundError:
                continue
            except Exception as error:
                print(f"[skip] {dataset_sample}, {method}: {error}")
                continue

            for k in RUN_COUNTS:
                try:
                    instability = load_instability(
                        dataset,
                        sample,
                        method,
                        k,
                    )
                except FileNotFoundError:
                    print(
                        f"[skip] Missing instability: "
                        f"{dataset_sample}, {method}, k={k}"
                    )
                    continue

                run_index = reference_run_index(results, k)
                predicted = np.asarray(
                    results[run_index]
                ).astype(str)

                if (
                    len(predicted) != adata.n_obs
                    or len(instability) != adata.n_obs
                ):
                    print(
                        f"[skip] Length mismatch: "
                        f"{dataset_sample}, {method}, k={k}"
                    )
                    continue

                if method == "bayesspace":
                    spot_entropy = load_bayesspace_entropy(
                        dataset,
                        sample,
                        run_index,
                        adata.n_obs,
                    )
                else:
                    spot_entropy = np.full(adata.n_obs, np.nan)

                for cluster in np.unique(predicted):
                    cluster_mask = predicted == cluster
                    cluster_idx = np.flatnonzero(cluster_mask)
                    n_cluster = len(cluster_idx)

                    (
                        matched_annotation,
                        jaccard,
                        precision,
                        recall,
                    ) = match_cluster_to_annotation(
                        cluster_mask,
                        valid_annotation,
                        annotation,
                        annotation_sizes,
                    )

                    coherence = np.nan
                    expression_variance = np.nan

                    if X_pca is not None:
                        X_cluster = X_pca[cluster_idx]

                        if n_cluster >= 2:
                            vector_sum = X_cluster.sum(axis=0)
                            pairwise_sum = 0.5 * (
                                float(vector_sum @ vector_sum)
                                - n_cluster
                            )
                            coherence = pairwise_sum / (
                                n_cluster * (n_cluster - 1) / 2
                            )

                        squared_sum = float(
                            np.sum(X_cluster * X_cluster)
                        )
                        mean_vector = X_cluster.mean(axis=0)

                        expression_variance = (
                            squared_sum / n_cluster
                            - float(mean_vector @ mean_vector)
                        ) / 20.0

                        if (
                            expression_variance < 0
                            and abs(expression_variance) < 1e-12
                        ):
                            expression_variance = 0.0

                    cluster_entropy = np.nan

                    if np.isfinite(
                        spot_entropy[cluster_mask]
                    ).any():
                        cluster_entropy = float(
                            np.nanmean(
                                spot_entropy[cluster_mask]
                            )
                        )

                    rows.append({
                        "dataset-sample": dataset_sample,
                        "method": method,
                        "iterations": k,
                        "reference_mode": REFERENCE_MODE,
                        "reference_run": run_index,
                        "cluster": cluster,
                        "n_spots": n_cluster,
                        "matched annotation": matched_annotation,
                        "clusterwise mean instability": float(
                            np.nanmean(instability[cluster_mask])
                        ),
                        "clusterwise jaccard": jaccard,
                        "clusterwise purity (precision)": precision,
                        "clusterwise recall": recall,
                        "expression coherence": coherence,
                        "expression variance": expression_variance,
                        "bayesspace clusterwise entropy": cluster_entropy,
                    })

        del adata, X_pca
        gc.collect()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(df)} rows to:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()
