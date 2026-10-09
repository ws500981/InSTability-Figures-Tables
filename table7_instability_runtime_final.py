#!/usr/bin/env python3

"""Recalculate Table 7 instability runtimes and write CSV + LaTeX only.

This script is intentionally benchmark-focused:
  * nested run selection only
  * exactly 50 clustering runs per sample/method
  * recomputes the full instability score calculation
  * writes a temporary instability PKL for benchmarking and deletes it immediately
  * runs each sample/method calculation independently in parallel with joblib
  * saves one raw runtime CSV plus the final Table 7 CSV and LaTeX

The Table 7 runtime is the end-to-end time required to compute
and save instability from saved clustering results. It includes:

    clustering-result loading
    label encoding
    temporary-array setup
    direct count computation
    instability-score calculation
    instability-result serialization

Temporary benchmark PKL files are deleted after timing.
"""

import os

# Force common numerical backends to one thread. These must be set before
# importing NumPy/Pandas/joblib so the benchmark is genuinely single-threaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_RESULTS_ROOT = Path(
    "./data/clustering_results"
)

BASE = Path(
    "./data"
)

# Keep all timing/table outputs separate from the existing instability outputs.
OUT_DIR = BASE / "supp/t7"

RAW_RUNTIME_CSV = OUT_DIR / "table7_runtime_raw_50runs_correct_node.csv"
OUT_CSV = OUT_DIR / "table7_instability_runtime_50runs.csv"
OUT_TEX = OUT_DIR / "table7_instability_runtime_50runs.tex"

MAX_ITERATIONS = 50
# RUNTIME_COL = "effective_compute_seconds_for_k"
RUNTIME_COL = "end_to_end_seconds"

# One independent process per sample/method calculation.
# None => use all CPUs allocated by Slurm, capped by the number of tasks.
N_JOBS = None

METHODS = [
    ("Leiden", "leiden"),
    ("Louvain", "louvain"),
    ("Bayes-Space", "bayesspace"),
    ("GraphST", "graphst"),
    ("STAGATE", "stagate"),
    ("SpiceMix", "spicemix"),
    ("SEDR", "sedr"),
    ("SEDR-mclust", "sedr_mclust"),
]

# Only samples that appear in Table 7.
# tuple = (section, display_label, dataset_directory, sample_name)
TABLE_ROWS = [
    ("DLPFC", "151507", "dlpfc", "151507"),
    ("DLPFC", "151508", "dlpfc", "151508"),
    ("DLPFC", "151509", "dlpfc", "151509"),
    ("DLPFC", "151510", "dlpfc", "151510"),
    ("DLPFC", "151669", "dlpfc", "151669"),
    ("DLPFC", "151670", "dlpfc", "151670"),
    ("DLPFC", "151671", "dlpfc", "151671"),
    ("DLPFC", "151672", "dlpfc", "151672"),
    ("DLPFC", "151673", "dlpfc", "151673"),
    ("DLPFC", "151674", "dlpfc", "151674"),
    ("DLPFC", "151675", "dlpfc", "151675"),
    ("DLPFC", "151676", "dlpfc", "151676"),
    ("Other", "MB", "mouse_brain", None),
    ("Other", "HBC", "human_breast_cancer", None),
    ("Other", "MBC", "mouse_brain_cerebellum", None),
    (
        "Cropped Visium HD Samples",
        "CRC",
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        None,
    ),
    ("Cropped Visium HD Samples", "COAD", "coad_ffpe", None),
    ("Cropped Visium HD Samples", "OV", "ov_ffpe", None),
    (
        "Full Visium HD Samples",
        "CRC",
        "Visium_HD_Human_Colon_Cancer",
        None,
    ),
    ("Full Visium HD Samples", "OV", "ov_ffpe_full", None),
    ("Full Visium HD Samples", "COAD", "coad_ffpe_full", None),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def elapsed_seconds(start):
    return time.perf_counter() - start


def method_was_run(dataset, method):
    """Match the method/sample combinations used in the original analysis."""
    if (
        "full" in dataset
        or dataset == "Visium_HD_Human_Colon_Cancer"
    ) and method not in ("leiden", "louvain"):
        return False

    if method == "stagate" and dataset == "mouse_brain_cerebellum":
        return False

    if method == "sedr_mclust" and dataset != "dlpfc":
        return False

    return True


def encode_labels(label_arrays):
    """Replace arbitrary cluster labels by compact int32 codes."""
    codes = []
    n = len(label_arrays[0])

    for run, labels in enumerate(label_arrays):
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError(
                f"Run {run} has {len(labels)} spots, expected {n}."
            )

        _, c = np.unique(labels, return_inverse=True)
        codes.append(c.astype(np.int32, copy=False))

    return codes


# -----------------------------------------------------------------------------
# Exact direct instability calculation
# -----------------------------------------------------------------------------


def add_runs_to_counts(
    codes,
    row_sums_counts,
    intra_sum_counts,
    start_run,
    end_run,
):
    """Compute compact direct co-association counts for the selected runs."""
    start = time.perf_counter()

    for q in range(start_run, end_run):
        cq = codes[q]

        sizes_q = np.bincount(cq).astype(np.uint32, copy=False)
        row_sums_counts += sizes_q[cq]

        n_q_clusters = len(sizes_q)

        for r in range(q + 1):
            cr = codes[r]

            n_r_clusters = int(cr.max()) + 1
            if n_r_clusters * n_q_clusters <= np.iinfo(np.int32).max:
                pair_key = cr * np.int32(n_q_clusters)
                pair_key += cq
            else:
                pair_key = cr.astype(np.int64) * n_q_clusters
                pair_key += cq

            pair_counts = np.bincount(pair_key).astype(
                np.uint32,
                copy=False,
            )
            contribution = pair_counts[pair_key]
            contribution -= np.uint32(1)

            intra_sum_counts[r] += contribution
            if r != q:
                intra_sum_counts[q] += contribution

    return elapsed_seconds(start)


def calculate_uncertainty(codes, row_sums_counts, intra_sum_counts, k):
    """Calculate LOO-normalized spot and sample instability."""
    start = time.perf_counter()

    if k < 2:
        raise ValueError("At least two clustering runs are required.")

    n = codes[0].size

    row_sums_f = row_sums_counts.astype(np.float32)
    spot_score_sum = np.zeros(n, dtype=np.float32)
    loo_factor = np.float32(k / (k - 1))

    for run in range(k):
        labels = codes[run]
        intra_sum_f = intra_sum_counts[run].astype(np.float32)

        intra_sum = intra_sum_f / k
        inter_sum = (
            row_sums_f - intra_sum_f - k
        ) / k

        cluster_sizes = np.bincount(labels).astype(
            np.uint32,
            copy=False,
        )
        sizes_per_spot = cluster_sizes[labels]

        intra_count = sizes_per_spot - np.uint32(1)
        inter_count = np.uint32(n) - sizes_per_spot

        mean_intra = np.divide(
            intra_sum,
            intra_count,
            out=np.ones_like(intra_sum),
            where=intra_count != 0,
        )

        mean_inter = np.divide(
            inter_sum,
            inter_count,
            out=np.zeros_like(inter_sum),
            where=inter_count != 0,
        )

        run_score = 1.0 - (mean_intra - mean_inter)
        run_score *= loo_factor
        spot_score_sum += run_score

    spot_instability = spot_score_sum / np.float32(k)
    sample_instability = float(
        np.mean(spot_instability, dtype=np.float64)
    )

    return (
        spot_instability,
        sample_instability,
        elapsed_seconds(start),
    )


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def load_clustering_results(dataset, method, sample_name, iterations):
    load_dir = INPUT_RESULTS_ROOT / dataset / method

    if dataset != "dlpfc":
        result_path = (
            load_dir / f"{method}_results_{iterations}iterations.csv"
            if method == "bayesspace"
            else load_dir / f"{method}_results_{iterations}iterations.pkl"
        )
    else:
        result_path = (
            load_dir
            / f"{sample_name}_{method}_results_{iterations}iterations.csv"
            if method == "bayesspace"
            else load_dir
            / f"{sample_name}_{method}_results_{iterations}iterations.pkl"
        )

    if not result_path.exists():
        raise FileNotFoundError(f"Missing clustering result: {result_path}")

    if method == "bayesspace":
        frame = pd.read_csv(result_path)
        clustering_results = [
            frame[col].to_numpy()
            for col in frame.columns
        ]
    else:
        clustering_results = joblib.load(result_path)

    if len(clustering_results) < iterations:
        raise ValueError(
            f"{result_path} contains only {len(clustering_results)} runs; "
            f"{iterations} are required."
        )

    return clustering_results[:iterations], result_path


# -----------------------------------------------------------------------------
# One benchmark task
# -----------------------------------------------------------------------------


def benchmark_one(dataset, method, sample_name=None):
    sample_label = sample_name if sample_name else "single_sample"

    print("\n" + "=" * 80)
    print(
        f"STARTING: dataset={dataset}, sample={sample_label}, "
        f"method={method}"
    )
    print("=" * 80)

    end_to_end_start = time.perf_counter()
    load_start = time.perf_counter()
    clustering_results, clustering_path = load_clustering_results(
        dataset=dataset,
        method=method,
        sample_name=sample_name,
        iterations=MAX_ITERATIONS,
    )

    if dataset == "dlpfc" and method == "sedr_mclust":
        idx_path = (
            INPUT_RESULTS_ROOT
            / "dlpfc"
            / "sedr_mclust"
            / f"{sample_name}_idx.pkl"
        )
        idx = joblib.load(idx_path)
        clustering_results = [
            np.asarray(labels)[idx]
            for labels in clustering_results
        ]

    codes = encode_labels(clustering_results)
    del clustering_results

    clustering_load_seconds = elapsed_seconds(load_start)
    n = codes[0].size

    if MAX_ITERATIONS * n >= np.iinfo(np.uint32).max:
        raise ValueError(
            "MAX_ITERATIONS * n_spots exceeds uint32 capacity."
        )

    # Nested K=50 means runs 0..49, exactly as in the original nested mode.
    selected_codes = codes[:MAX_ITERATIONS]

    row_sums_counts = np.zeros(n, dtype=np.uint32)
    intra_sum_counts = np.zeros(
        (MAX_ITERATIONS, n),
        dtype=np.uint32,
    )

    direct_count_seconds = add_runs_to_counts(
        codes=selected_codes,
        row_sums_counts=row_sums_counts,
        intra_sum_counts=intra_sum_counts,
        start_run=0,
        end_run=MAX_ITERATIONS,
    )

    (
        spot_instability,
        sample_instability,
        uncertainty_compute_seconds,
    ) = calculate_uncertainty(
        codes=selected_codes,
        row_sums_counts=row_sums_counts,
        intra_sum_counts=intra_sum_counts,
        k=MAX_ITERATIONS,
    )

    # ---------------------------------------------------------
    # Save instability result to a temporary PKL.
    # This serialization time IS included in end-to-end runtime.
    # ---------------------------------------------------------

    sample_tag = sample_name if sample_name else "single_sample"

    temp_instability_path = (
        OUT_DIR
        / f".tmp_table7_{dataset}_{sample_tag}_{method}_{os.getpid()}.pkl"
    )

    save_start = time.perf_counter()

    joblib.dump(
        {
            "spot_uncertainty": spot_instability,
            "uncertainty": sample_instability,
            "iterations": MAX_ITERATIONS,
            "run_selection_mode": "nested",
            "selected_run_indices": list(range(MAX_ITERATIONS)),
        },
        temp_instability_path,
    )

    instability_save_seconds = elapsed_seconds(save_start)

    # End-to-end timing stops AFTER the result has been serialized.
    end_to_end_seconds = elapsed_seconds(end_to_end_start)

    print(f"Loaded clustering runs from: {clustering_path}")
    print(f"Spots: {n:,}")
    print(
        "Clustering-result loading/encoding runtime: "
        f"{clustering_load_seconds:.4f} seconds"
    )

    # Delete temporary benchmark file.
    # Deletion is deliberately outside the reported end-to-end runtime.
    temp_instability_path.unlink()

    effective_compute_seconds = (
        direct_count_seconds + uncertainty_compute_seconds
    )

    print(
        f"Direct-count runtime: {direct_count_seconds:.6f} seconds"
    )
    print(
        "Instability calculation runtime: "
        f"{uncertainty_compute_seconds:.6f} seconds"
    )
    # print(
    #     "Effective compute runtime for Table 7: "
    #     f"{effective_compute_seconds:.6f} seconds"
    # )
    print(
        "Compute-only runtime: "
        f"{effective_compute_seconds:.6f} seconds"
    )

    print(
        "Instability-result save runtime: "
        f"{instability_save_seconds:.6f} seconds"
    )

    print(
        "End-to-end runtime for Table 7: "
        f"{end_to_end_seconds:.6f} seconds"
    )

    # Discard in-memory instability results after temporary serialization.
    del (
        spot_instability,
        sample_instability,
        selected_codes,
        codes,
        row_sums_counts,
        intra_sum_counts,
    )

    return {
        "worker_pid": os.getpid(),
        "dataset": dataset,
        "sample": sample_name if sample_name else "",
        "method": method,
        "iterations": MAX_ITERATIONS,
        "n_spots": n,
        "run_selection_mode": "nested",
        "clustering_load_seconds_shared": clustering_load_seconds,
        "direct_count_compute_seconds": direct_count_seconds,
        "uncertainty_compute_seconds": uncertainty_compute_seconds,
        "instability_save_seconds": instability_save_seconds,
        "effective_compute_seconds_for_k": effective_compute_seconds,
        "end_to_end_seconds": end_to_end_seconds,
    }


# -----------------------------------------------------------------------------
# Table construction
# -----------------------------------------------------------------------------


def latex_value(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.2f}"


def build_table(runtime_df):
    rows = []

    for section, label, dataset, sample_name in TABLE_ROWS:
        row = {
            "Section": section,
            "Dataset": label,
            "# Spots": None,
        }

        for display_method, method in METHODS:
            if not method_was_run(dataset, method):
                row[display_method] = pd.NA
                continue

            sample_key = sample_name if sample_name else ""
            hit = runtime_df[
                (runtime_df["dataset"] == dataset)
                & (runtime_df["sample"] == sample_key)
                & (runtime_df["method"] == method)
                & (runtime_df["iterations"] == MAX_ITERATIONS)
                & (runtime_df["run_selection_mode"] == "nested")
            ]

            if len(hit) != 1:
                raise ValueError(
                    "Expected exactly one runtime row for "
                    f"dataset={dataset}, sample={sample_name}, "
                    f"method={method}; found {len(hit)}."
                )

            source = hit.iloc[0]
            n_spots = int(source["n_spots"])

            if row["# Spots"] is None:
                row["# Spots"] = n_spots
            elif row["# Spots"] != n_spots:
                raise ValueError(
                    f"Inconsistent n_spots for {dataset}/{sample_name}: "
                    f"{row['# Spots']} vs {n_spots}."
                )

            row[display_method] = float(source[RUNTIME_COL])

        if row["# Spots"] is None:
            raise ValueError(
                f"Could not determine n_spots for {dataset}/{sample_name}."
            )

        rows.append(row)

    return pd.DataFrame(rows)


def latex_table(table):
    lines = []
    previous_section = None

    for _, row in table.iterrows():
        section = row["Section"]

        if section != previous_section:
            if previous_section is not None:
                lines.append(r"\midrule")

            if section != "Other":
                lines.append(
                    rf"\multicolumn{{10}}{{l}}{{\textbf{{{section}}}}} \\"
                )

            previous_section = section

        values = [
            str(row["Dataset"]),
            f"{int(row['# Spots']):,}",
        ]
        values.extend(
            latex_value(row[display_method])
            for display_method, _ in METHODS
        )
        lines.append(" & ".join(values) + r" \\")

    body = "\n".join(lines)

    return rf"""\begin{{table}}[t]
\centering
\scriptsize
\caption[Runtime of instability score calculation from 50 runs]{{\textbf{{Runtime of instability score calculation from 50 runs.}}
Runtimes are given in seconds (single thread).
NA indicates the method was not run on the sample.
All instability score calculations were performed on the NIH Biowolf cluster on a compute node with AMD EPYC 9454 processor and 3TB memory (note this is a CPU node not a GPU node).
Jobs were submitted via Slurm with flags \texttt{{--exclusive}}, and \texttt{{--mem=0}} to ensure there were not competing processes on the node.
}}
\label{{tab:instability-runtime}}
\begin{{tabular}}{{
>{{\raggedright\arraybackslash}}p{{1.4cm}}
>{{\raggedleft\arraybackslash}}p{{1.4cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
>{{\raggedleft\arraybackslash}}p{{1.1cm}}
}}
\toprule
Dataset & \# Spots & Leiden & Louvain & Bayes-Space & GraphST & STAGATE & SpiceMix & SEDR & SEDR-mclust \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _allocated_cpus():
    """Best-effort count of CPUs available to this Slurm job."""
    if N_JOBS is not None:
        return int(N_JOBS)

    value = os.environ.get("SLURM_CPUS_ON_NODE")
    if value:
        # Normally this is a plain integer. Keep parsing robust to values
        # with extra Slurm notation by taking the leading integer.
        digits = ""
        for char in value:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return int(digits)

    return os.cpu_count() or 1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, _, dataset, sample_name in TABLE_ROWS:
        for _, method in METHODS:
            if method_was_run(dataset, method):
                tasks.append((dataset, method, sample_name))

    allocated_cpus = _allocated_cpus()
    n_jobs = min(len(tasks), allocated_cpus)

    print(f"Benchmarking {len(tasks)} sample-method combinations")
    print("Run selection: nested")
    print("Runs per calculation: 50")
    print(f"Slurm/visible CPUs: {allocated_cpus}")
    print(f"Independent joblib workers: {n_jobs}")
    print("Each worker: 1 CPU thread")
    print("Instability PKL output: temporary benchmark file, deleted after timing")
    print(f"Runtime/table output directory: {OUT_DIR}")

    full_script_start = time.perf_counter()

    # Each tuple is a completely independent sample x method calculation.
    # loky uses separate processes. The environment variables at the top of
    # this file keep NumPy/BLAS/OpenMP work inside each process single-threaded.
    runtime_records = joblib.Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
        batch_size=1,
        pre_dispatch="all",
    )(
        joblib.delayed(benchmark_one)(
            dataset=dataset,
            method=method,
            sample_name=sample_name,
        )
        for dataset, method, sample_name in tasks
    )

    full_script_seconds = elapsed_seconds(full_script_start)

    runtime_df = pd.DataFrame(runtime_records)
    runtime_df["full_script_wall_clock_seconds"] = full_script_seconds

    # Only runtime/table outputs are retained; temporary instability PKLs
    # have already been deleted.
    runtime_df.to_csv(
        RAW_RUNTIME_CSV,
        index=False,
        float_format="%.9f",
    )

    table = build_table(runtime_df)

    # Numeric final Table 7 source.
    table.drop(columns="Section").to_csv(
        OUT_CSV,
        index=False,
        float_format="%.6f",
        na_rep="NA",
    )

    # Manuscript-ready LaTeX.
    OUT_TEX.write_text(
        latex_table(table),
        encoding="utf-8",
    )

    display = table.drop(columns="Section").copy()
    display["# Spots"] = display["# Spots"].map(
        lambda x: f"{int(x):,}"
    )
    for display_method, _ in METHODS:
        display[display_method] = display[display_method].map(latex_value)

    print("\n" + "#" * 80)
    print("ALL TABLE 7 RUNTIME CALCULATIONS FINISHED")
    print(f"Independent joblib workers used: {n_jobs}")
    print(
        "Total wall-clock time: "
        f"{full_script_seconds:.2f} seconds "
        f"({full_script_seconds / 3600.0:.2f} hours)"
    )
    print("#" * 80)

    print("\nFinal Table 7 values (seconds):\n")
    print(display.to_string(index=False))

    print("\nSaved runtime/table files:")
    print(RAW_RUNTIME_CSV)
    print(OUT_CSV)
    print(OUT_TEX)
    print("\nTemporary instability PKL files were deleted after benchmarking.")


if __name__ == "__main__":
    main()
