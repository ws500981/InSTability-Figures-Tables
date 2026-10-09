#----------------------------------------------
# save csv to contain all info: all methods ari (ex run, across runs), instability, bayesspace entropy (ex run, across runs)
# extending the all_sample_method_metrics.csv code to include k-run instability
#----------------------------------------------
#!/usr/bin/env python3

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

H5AD_DIR = Path("./data/ST_datasets/AAA_with_uncert")

RESULTS_ROOT = Path("./data/clustering_results")

INSTABILITY_ROOT = Path("./data/1-nested-non-sampling")

OUT_DIR = Path("./data/fig3")

CSV_OUT = OUT_DIR / "all_sample_method_metrics_by_number_of_runs.csv"

RUN_COUNTS = list(range(5, 51, 5))
MAX_RUNS = 50

# Options:
#   "example_50": example run selected from all 50 runs
#   "example_k":  example run selected from the first k runs
#   "example_0":  always use run 0
EXAMPLE_MODE = "example_50"

METHODS = [
    "sedr",
    "graphst",
    "stagate",
    "bayesspace",
    "louvain",
    "leiden",
    "spicemix",
    "sedr_mclust",
]

DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

SAMPLES = [
    ("mouse_brain_cerebellum", None, "MBC"),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None, "CRC"),
    ("human_breast_cancer", None, "HBC"),
    ("mouse_brain", None, "MB"),
    ("coad_ffpe", None, "COAD"),
    ("ov_ffpe", None, "OV"),
] + [
    ("dlpfc", sample, sample)
    for sample in DLPFC_SAMPLES
]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def dataset_tag(dataset, sample):
    return f"{dataset}_{sample}" if sample else dataset


def h5ad_path(dataset, sample):
    return H5AD_DIR / f"{dataset_tag(dataset, sample)}.h5ad"


def instability_path(dataset, sample, method, k):
    tag = dataset_tag(dataset, sample)

    return (
        INSTABILITY_ROOT
        / dataset
        / method
        / f"{tag}_{method}_uncertainty_{k}runs.pkl"
    )


def bayesspace_softlabel_path(dataset, sample):
    prefix = f"{sample}_" if sample else ""

    return (
        RESULTS_ROOT
        / dataset
        / "bayesspace"
        / f"{prefix}bayesspace_softlabels_50iterations.csv"
    )


# ---------------------------------------------------------------------
# General metrics
# ---------------------------------------------------------------------

def identify_example_run(results):
    return min(
        range(len(results)),
        key=lambda i: (
            len(np.unique(np.asarray(results[i]).astype(str))),
            i,
        ),
    )


def select_example_run(results, k):
    if EXAMPLE_MODE == "example_50":
        return identify_example_run(results[:MAX_RUNS])

    if EXAMPLE_MODE == "example_k":
        return identify_example_run(results[:k])

    if EXAMPLE_MODE == "example_0":
        return 0

    raise ValueError(
        "EXAMPLE_MODE must be 'example_50', "
        "'example_k', or 'example_0'."
    )


def prepare_reference_labels(adata):
    if "annotation" not in adata.obs:
        return None, None

    valid_mask = adata.obs["annotation"].notna().to_numpy()

    if valid_mask.sum() < 2:
        return None, None

    reference = (
        adata.obs.loc[valid_mask, "annotation"]
        .astype(str)
        .to_numpy()
    )

    return valid_mask, reference


def calculate_ari(reference, predicted):
    if reference is None:
        return np.nan

    return float(
        adjusted_rand_score(
            reference,
            np.asarray(predicted).astype(str),
        )
    )


def load_mean_instability(dataset, sample, method, k):
    path = instability_path(dataset, sample, method, k)

    if not path.exists():
        return np.nan

    result = joblib.load(path)

    return float(
        np.nanmean(
            np.asarray(result["spot_uncertainty"], dtype=float)
        )
    )


def calculate_method_metrics(
    adata,
    dataset,
    sample,
    method,
    k,
    valid_mask,
    reference,
):
    output = {
        "mean_ari_all_runs": np.nan,
        "example_run_ari": np.nan,
        "mean_instability": load_mean_instability(
            dataset, sample, method, k
        ),
        "example_run_index": None,
    }

    results_key = f"{method}_results"

    if results_key not in adata.uns:
        return output

    results = adata.uns[results_key]

    if results is None or len(results) < k:
        return output

    results = [
        np.asarray(labels)
        for labels in results[:MAX_RUNS]
    ]

    example_index = select_example_run(results, k)
    output["example_run_index"] = example_index

    if valid_mask is None:
        return output

    run_aris = [
        calculate_ari(reference, labels[valid_mask])
        for labels in results[:k]
    ]

    output["mean_ari_all_runs"] = float(
        np.mean(run_aris)
    )

    output["example_run_ari"] = calculate_ari(
        reference,
        results[example_index][valid_mask],
    )

    return output


# ---------------------------------------------------------------------
# BayesSpace entropy
# ---------------------------------------------------------------------

def entropy_from_softlabels(probabilities, eps=1e-12):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    row_sums = probabilities.sum(axis=1, keepdims=True)

    probabilities = np.divide(
        probabilities,
        row_sums,
        out=np.zeros_like(probabilities),
        where=row_sums > 0,
    )

    return -np.sum(
        probabilities * np.log(probabilities + eps),
        axis=1,
    )


def group_softlabel_columns(softlabels):
    pattern = re.compile(r"^iter_(\d+)_cluster_(\d+)$")
    grouped = {}

    for column in softlabels.columns:
        match = pattern.match(str(column))

        if match:
            iteration = int(match.group(1))
            cluster = int(match.group(2))

            grouped.setdefault(iteration, []).append(
                (cluster, column)
            )

    return {
        iteration: [
            column
            for _, column in sorted(columns)
        ]
        for iteration, columns in grouped.items()
    }


def calculate_bayesspace_entropy(
    dataset,
    sample,
    n_spots,
    k,
    example_index,
):
    output = {
        "mean_entropy_all_runs": np.nan,
        "example_run_mean_entropy": np.nan,
    }

    if example_index is None:
        return output

    path = bayesspace_softlabel_path(dataset, sample)

    if not path.exists():
        return output

    softlabels = pd.read_csv(path)
    grouped = group_softlabel_columns(softlabels)

    if not grouped:
        return output

    offset = 0 if 0 in grouped else 1
    run_entropies = []

    for run_index in range(k):
        columns = grouped.get(run_index + offset)

        if not columns:
            continue

        probabilities = softlabels[
            columns
        ].to_numpy(dtype=np.float64)

        if probabilities.shape[0] != n_spots:
            raise ValueError(
                f"{dataset_tag(dataset, sample)}: "
                "BayesSpace soft-label row mismatch."
            )

        run_entropies.append(
            float(
                np.nanmean(
                    entropy_from_softlabels(probabilities)
                )
            )
        )

    if run_entropies:
        output["mean_entropy_all_runs"] = float(
            np.mean(run_entropies)
        )

    example_columns = grouped.get(
        example_index + offset
    )

    if example_columns:
        probabilities = softlabels[
            example_columns
        ].to_numpy(dtype=np.float64)

        output["example_run_mean_entropy"] = float(
            np.nanmean(
                entropy_from_softlabels(probabilities)
            )
        )

    return output


# ---------------------------------------------------------------------
# One sample and run count
# ---------------------------------------------------------------------

def calculate_sample(dataset, sample, sample_label, k):
    adata = sc.read_h5ad(h5ad_path(dataset, sample))

    valid_mask, reference = prepare_reference_labels(adata)

    row = {
        "dataset": dataset,
        "sample": sample if sample else "",
        "sample_label": sample_label,
        "iterations": k,
        "example_mode": EXAMPLE_MODE,
    }

    method_outputs = {}

    for method in METHODS:
        metrics = calculate_method_metrics(
            adata=adata,
            dataset=dataset,
            sample=sample,
            method=method,
            k=k,
            valid_mask=valid_mask,
            reference=reference,
        )

        method_outputs[method] = metrics

        row[f"{method}_mean_ari_all_runs"] = (
            metrics["mean_ari_all_runs"]
        )
        row[f"{method}_example_run_ari"] = (
            metrics["example_run_ari"]
        )
        row[f"{method}_mean_instability"] = (
            metrics["mean_instability"]
        )
        row[f"{method}_example_run_index"] = (
            metrics["example_run_index"]
        )

    bayesspace_entropy = calculate_bayesspace_entropy(
        dataset=dataset,
        sample=sample,
        n_spots=adata.n_obs,
        k=k,
        example_index=method_outputs[
            "bayesspace"
        ]["example_run_index"],
    )

    row["bayesspace_mean_entropy_all_runs"] = (
        bayesspace_entropy["mean_entropy_all_runs"]
    )
    row["bayesspace_example_run_mean_entropy"] = (
        bayesspace_entropy["example_run_mean_entropy"]
    )

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset, sample, sample_label in SAMPLES:
        tag = dataset_tag(dataset, sample)

        for k in RUN_COUNTS:
            print(f"[processing] {tag}, k={k}")

            try:
                rows.append(
                    calculate_sample(
                        dataset,
                        sample,
                        sample_label,
                        k,
                    )
                )

            except Exception as error:
                print(
                    f"[error] {tag}, k={k}: {repr(error)}"
                )

    results = pd.DataFrame(rows)

    ordered_columns = [
        "dataset",
        "sample",
        "sample_label",
        "iterations",
        "example_mode",
    ]

    for method in METHODS:
        ordered_columns.extend([
            f"{method}_mean_ari_all_runs",
            f"{method}_example_run_ari",
            f"{method}_mean_instability",
            f"{method}_example_run_index",
        ])

    ordered_columns.extend([
        "bayesspace_mean_entropy_all_runs",
        "bayesspace_example_run_mean_entropy",
    ])

    results = results[ordered_columns]

    results.to_csv(
        CSV_OUT,
        index=False,
        float_format="%.8f",
        na_rep="",
    )

    print(f"\nSaved CSV:\n{CSV_OUT}")


if __name__ == "__main__":
    main()
