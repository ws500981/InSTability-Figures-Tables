# use visiumhd3 environment

import multiprocessing as mp
import os
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_RESULTS_ROOT = (
    "/home/wuw15/data_dir/my_analysis_python/"
    "uncertainty_public_data/outs"
)



MAX_ITERATIONS = 50
RUN_COUNTS = list(range(5, MAX_ITERATIONS + 1, 5))

# How to choose clustering runs for each K:
#
# "nested":
#     Preserve the original behavior:
#     K=5  -> runs 0:5
#     K=10 -> runs 0:10
#     ...
#
# "sampled":
#     Generate random permutations of the 50 runs.
#     Within each replicate, K values are nested prefixes of the
#     same permutation.
RUN_SELECTION_MODE = "sampled"  # "nested" or "sampled"

OUTPUT_ROOT = (f"/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/{'2-sampled' if RUN_SELECTION_MODE == 'sampled' else '1-nested-non-sampling'}")

# Number of random run-order replicates in sampled mode.
N_SAMPLING_REPEATS = 25

# Fixed seed makes the sampling exactly reproducible.
SAMPLING_SEED = 20260907


# Only final uncertainty values and timing CSVs are written.
SAVE_UNCERTAINTY = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def elapsed_seconds(start):
    return time.perf_counter() - start

def make_run_selection_plans(
    max_iterations,
    mode,
    n_repeats=100,
    seed=20260907,
):
    """
    Return run-index orders used for run-count sensitivity analysis.

    nested:
        One deterministic order [0, 1, ..., R-1].

    sampled:
        n_repeats independent random permutations of all R runs.
        For each permutation, the first K entries are used for a
        K-run analysis. Thus K values remain nested within each
        sampling replicate.
    """
    if mode == "nested":
        return [
            np.arange(max_iterations, dtype=np.int32)
        ]

    if mode == "sampled":
        rng = np.random.default_rng(seed)

        return [
            rng.permutation(max_iterations).astype(
                np.int32,
                copy=False,
            )
            for _ in range(n_repeats)
        ]

    raise ValueError(
        f"Unknown RUN_SELECTION_MODE: {mode}. "
        "Expected 'nested' or 'sampled'."
    )

def make_output_paths(
    output_dir,
    dataset,
    method,
    sample_name,
    iterations,
    run_selection_mode="nested",
):
    stem = f"{dataset}{'_' + sample_name if sample_name else ''}"

    if run_selection_mode == "nested":

        # Preserve your original filenames exactly.
        uncertainty_filename = (
            f"{stem}_{method}_uncertainty_"
            f"{iterations}runs.pkl"
        )

        timing_filename = (
            f"{stem}_{method}_timing_"
            f"5to{MAX_ITERATIONS}runs.csv"
        )

    elif run_selection_mode == "sampled":

        uncertainty_filename = (
            f"{stem}_{method}_uncertainty_"
            f"{iterations}runs_sampled.pkl"
        )

        timing_filename = (
            f"{stem}_{method}_timing_"
            f"5to{MAX_ITERATIONS}runs_sampled.csv"
        )

        metadata_filename = (
            f"{stem}_{method}_sampling_metadata_"
            f"5to{MAX_ITERATIONS}runs.csv"
        )

    else:
        raise ValueError(
            f"Unknown run_selection_mode: "
            f"{run_selection_mode}"
        )

    return {
        "uncertainty": os.path.join(
            output_dir,
            uncertainty_filename,
        ),
        "timing": os.path.join(
            output_dir,
            timing_filename,
        ),
        "metadata": (
            os.path.join(output_dir, metadata_filename)
            if run_selection_mode == "sampled"
            else None
        ),
    }

def encode_labels(label_arrays):
    """
    Replace arbitrary cluster labels by compact int32 codes.

    Only the codes are retained, keeping memory linear in
    number_of_runs * number_of_spots.
    """
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


# ---------------------------------------------------------------------
# Exact direct uncertainty calculation
# ---------------------------------------------------------------------

def add_runs_to_counts(
    codes,
    row_sums_counts,
    intra_sum_counts,
    start_run,
    end_run,
):
    """
    Compute compact uncertainty counts for runs start_run to end_run.

    Each unordered pair of clustering runs is processed exactly once.
    No N x N co-association matrix is ever constructed.

    In this standalone-K script, this function is called with start_run=0
    and fresh arrays for every K, so each K is timed independently.
    """
    start = time.perf_counter()

    for q in range(start_run, end_run):
        cq = codes[q]

        # Sum_j coassociation(i, j), updated for the newly added run q.
        sizes_q = np.bincount(cq).astype(np.uint32, copy=False)
        row_sums_counts += sizes_q[cq]

        n_q_clusters = len(sizes_q)

        # Add all previously unseen run pairs (r, q), including (q, q).
        for r in range(q + 1):
            cr = codes[r]

            # One integer identifies each (cluster_in_r, cluster_in_q) pair.
            # int32 is enough in normal clustering problems and halves this
            # large temporary array. Fall back to int64 only if the theoretical
            # combined cluster id could overflow int32.
            n_r_clusters = int(cr.max()) + 1
            if n_r_clusters * n_q_clusters <= np.iinfo(np.int32).max:
                pair_key = cr * np.int32(n_q_clusters)
                pair_key += cq
            else:
                pair_key = cr.astype(np.int64) * n_q_clusters
                pair_key += cq

            # For each spot: number of OTHER spots in the same intersection.
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
    """
    Calculate LOO-normalized spot-level clustering instability
    for the first k clustering runs.

    The raw all-run formulation has a chance baseline of 1 - 1/k
    because the reference run contributes to its own co-association
    matrix. Multiplying each run-specific score by k / (k - 1)
    is exactly equivalent to excluding the reference run from the
    co-association calculation.

    Sample instability is defined as the mean spot-level instability.
    """
    start = time.perf_counter()

    if k < 2:
        raise ValueError(
            "At least two clustering runs are required "
            "to calculate LOO-normalized instability."
        )

    n = codes[0].size

    row_sums_f = row_sums_counts.astype(np.float32)
    spot_score_sum = np.zeros(n, dtype=np.float32)

    # Converts the original all-run score to the equivalent
    # leave-one-run-out score.
    loo_factor = np.float32(k / (k - 1))

    for run in range(k):
        labels = codes[run]
        intra_sum_f = intra_sum_counts[run].astype(np.float32)

        # These are the original all-run co-association sums.
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

        # If a spot is a singleton in this run, intra_count == 0.
        # By convention mean_intra = 1 in that case.
        mean_intra = np.divide(
            intra_sum,
            intra_count,
            out=np.ones_like(intra_sum),
            where=intra_count != 0,
        )

        # If the entire sample is one cluster, inter_count == 0.
        # By convention mean_inter = 0 in that case.
        #
        # IMPORTANT: For singleton clusters we KEEP inter_count = n - 1,
        # so their relationships with spots in other runs still contribute.
        mean_inter = np.divide(
            inter_sum,
            inter_count,
            out=np.zeros_like(inter_sum),
            where=inter_count != 0,
        )

        # Original all-run score:
        #
        #   1 - (mean_intra - mean_inter)
        #
        # Multiplication by k/(k-1) gives the exactly equivalent
        # leave-one-run-out-normalized score.
        run_score = 1.0 - (mean_intra - mean_inter)
        run_score *= loo_factor

        spot_score_sum += run_score

    # Average across reference runs for each spot.
    spot_instability = spot_score_sum / np.float32(k)

    # Sample-level instability is explicitly the arithmetic mean
    # of the spot-level instability scores.
    sample_instability = float(
        np.mean(spot_instability, dtype=np.float64)
    )

    return (
        spot_instability,
        sample_instability,
        elapsed_seconds(start),
    )

# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_clustering_results(
    dataset,
    method,
    sample_name,
    iterations,
    load_dir,
):
    if dataset != "dlpfc":
        result_path = (
            f"{load_dir}/{method}_results_{iterations}iterations.csv"
            if method == "bayesspace"
            else f"{load_dir}/{method}_results_{iterations}iterations.pkl"
        )
    else:
        result_path = (
            f"{load_dir}/{sample_name}_{method}_results_"
            f"{iterations}iterations.csv"
            if method == "bayesspace"
            else
            f"{load_dir}/{sample_name}_{method}_results_"
            f"{iterations}iterations.pkl"
        )

    if method == "bayesspace":
        frame = pd.read_csv(result_path)
        clustering_results = [
            frame[col].to_numpy() for col in frame.columns
        ]
    else:
        clustering_results = joblib.load(result_path)

    if len(clustering_results) < iterations:
        raise ValueError(
            f"{result_path} contains only {len(clustering_results)} runs; "
            f"{iterations} are required."
        )

    return clustering_results[:iterations], result_path


# ---------------------------------------------------------------------
# One dataset-sample-method task
# ---------------------------------------------------------------------

def process_dataset_method(
    dataset,
    method,
    sample_name=None,
    max_iterations=MAX_ITERATIONS,
    run_counts=RUN_COUNTS,
    save_uncertainty=SAVE_UNCERTAINTY,
    run_selection_mode=RUN_SELECTION_MODE,
    n_sampling_repeats=N_SAMPLING_REPEATS,
    sampling_seed=SAMPLING_SEED,
):
    task_start = time.perf_counter()

    sample_label = sample_name if sample_name else "single_sample"
    print(
        "\n"
        + "=" * 80
        + f"\nSTARTING: dataset={dataset}, sample={sample_label}, "
          f"method={method}\n"
        + "=" * 80
    )

    input_dir = f"{INPUT_RESULTS_ROOT}/{dataset}/{method}"
    output_dir = f"{OUTPUT_ROOT}/{dataset}/{method}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stem = f"{dataset}{'_' + sample_name if sample_name else ''}"

    load_start = time.perf_counter()
    clustering_results, clustering_path = load_clustering_results(
        dataset=dataset,
        method=method,
        sample_name=sample_name,
        iterations=max_iterations,
        load_dir=input_dir,
    )

    if dataset == "dlpfc" and method == "sedr_mclust":
        idx_path = (
            f"{INPUT_RESULTS_ROOT}/dlpfc/sedr_mclust/"
            f"{sample_name}_idx.pkl"
        )
        idx = joblib.load(idx_path)
        clustering_results = [
            np.asarray(labels)[idx]
            for labels in clustering_results
        ]

    # From here onward only compact int32 labels are needed.
    codes = encode_labels(clustering_results)
    del clustering_results

    clustering_load_seconds = elapsed_seconds(load_start)
    n = codes[0].size

    print(f"Loaded clustering runs from: {clustering_path}")
    print(f"Spots: {n:,}")
    print(
        f"Clustering-result loading/encoding runtime: "
        f"{clustering_load_seconds:.2f} seconds"
    )

    # uint32 is safe as long as max_iterations * n < 2**32.
    if max_iterations * n >= np.iinfo(np.uint32).max:
        raise ValueError(
            "max_iterations * n_spots exceeds uint32 capacity."
        )

    
    # Define which clustering runs will be used.
    #
    # nested:
    #     one plan: [0, 1, 2, ..., 49]
    #
    # sampled:
    #     multiple random permutations of [0, 1, 2, ..., 49]
    run_selection_plans = make_run_selection_plans(
        max_iterations=max_iterations,
        mode=run_selection_mode,
        n_repeats=n_sampling_repeats,
        seed=sampling_seed,
    )

    timing_rows = []
    metadata_rows = []

    if run_selection_mode == "sampled":
        spot_instability_values = {
            k: []
            for k in run_counts
        }

        sample_instability_values = {
            k: []
            for k in run_counts
        }

        sampling_counts = {
            k: 0
            for k in run_counts
        }

    for sampling_repeat, run_order in enumerate(run_selection_plans):

        print(
            "\n"
            + "*" * 80
            + f"\nRUN SELECTION: mode={run_selection_mode}, "
            f"repeat={sampling_repeat}\n"
            + "*" * 80
        )

        for k in run_counts:
            if k > max_iterations:
                raise ValueError(
                    f"Requested {k} runs, but max_iterations is "
                    f"{max_iterations}."
                )

            # In sampled mode, every permutation contains all runs when
            # k == max_iterations. The 50-run result is therefore identical
            # across repeats, so calculate it only once.
            if (
                run_selection_mode == "sampled"
                and k == max_iterations
                and sampling_repeat > 0
            ):
                continue

            # ----------------------------------------------------------
            # Select runs
            # ----------------------------------------------------------

            # nested mode:
            #   repeat 0, k=5  -> [0, 1, 2, 3, 4]
            #   repeat 0, k=10 -> [0, 1, ..., 9]
            #
            # sampled mode:
            #   each repeat has one random permutation;
            #   k values are nested prefixes of that permutation.
            selected_indices = run_order[:k]

            selected_codes = [
                codes[int(i)]
                for i in selected_indices
            ]

            k_start = time.perf_counter()

            print(
                "\n"
                + "-" * 80
                + f"\n{stem} | {method} | {k} runs | "
                f"mode={run_selection_mode} | "
                f"repeat={sampling_repeat}\n"
                + f"Selected run indices: "
                f"{selected_indices.tolist()}\n"
                + "-" * 80
            )

            # ----------------------------------------------------------
            # Fresh calculation for this exact set of k runs
            # ----------------------------------------------------------

            row_sums_counts = np.zeros(
                n,
                dtype=np.uint32,
            )

            intra_sum_counts = np.zeros(
                (k, n),
                dtype=np.uint32,
            )

            direct_count_seconds = add_runs_to_counts(
                codes=selected_codes,
                row_sums_counts=row_sums_counts,
                intra_sum_counts=intra_sum_counts,
                start_run=0,
                end_run=k,
            )

            (
                spot_uncertainty,
                uncertainty,
                uncertainty_compute_seconds,
            ) = calculate_uncertainty(
                codes=selected_codes,
                row_sums_counts=row_sums_counts,
                intra_sum_counts=intra_sum_counts,
                k=k,
            )

            # ----------------------------------------------------------
            # Output paths
            # ----------------------------------------------------------

            paths = make_output_paths(
                output_dir=output_dir,
                dataset=dataset,
                method=method,
                sample_name=sample_name,
                iterations=k,
                run_selection_mode=run_selection_mode,
            )

            # ----------------------------------------------------------
            # Handle instability result
            # ----------------------------------------------------------

            uncertainty_save_seconds = 0.0

            if run_selection_mode == "nested":

                # Nested mode: save each K normally.
                if save_uncertainty:
                    save_start = time.perf_counter()

                    joblib.dump(
                        {
                            "spot_uncertainty": spot_uncertainty,
                            "uncertainty": uncertainty,
                            "iterations": k,
                            "run_selection_mode": "nested",
                            "selected_run_indices": selected_indices.tolist(),
                        },
                        paths["uncertainty"],
                    )

                    uncertainty_save_seconds = elapsed_seconds(save_start)

            elif run_selection_mode == "sampled":
                # Save this replicate separately.
                # The 25 replicate vectors will be written together
                # into one PKL for this value of k.
                spot_instability_values[k].append(
                    spot_uncertainty.astype(np.float32, copy=True)
                )

                sample_instability_values[k].append(
                    float(uncertainty)
                )

                sampling_counts[k] += 1

            total_k_seconds = elapsed_seconds(k_start)
            if run_selection_mode == "sampled":
                metadata_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample_name if sample_name else "",
                        "method": method,
                        "iterations": k,
                        "sampling_repeat": sampling_repeat,
                        "sampling_seed": sampling_seed,
                        "selected_run_indices": ",".join(
                            map(str, selected_indices.tolist())
                        ),
                        "sample_instability": float(uncertainty),
                    }
                )

            # ----------------------------------------------------------
            # Record timing + selection metadata
            # ----------------------------------------------------------

            timing_rows.append(
                {
                    "dataset": dataset,
                    "sample": (
                        sample_name
                        if sample_name
                        else ""
                    ),
                    "method": method,
                    "iterations": k,
                    "n_spots": n,

                    # Run-selection information
                    "run_selection_mode": run_selection_mode,
                    "sampling_repeat": sampling_repeat,
                    # "sampling_seed": sampling_seed,
                    # "selected_run_indices": ",".join(
                    #     map(str,selected_indices.tolist())
                    # ),

                    # # Useful result to have directly in the CSV
                    # "sample_instability": uncertainty,

                    # Existing timing information
                    "coassoc_source": (
                        "not_used_direct_standalone_k"
                    ),
                    "clustering_load_seconds_shared": (
                        clustering_load_seconds
                    ),
                    "adata_load_seconds_shared": 0.0,
                    "coassoc_compute_seconds": 0.0,
                    "coassoc_load_seconds": 0.0,
                    "coassoc_save_seconds": 0.0,
                    "counts_load_seconds": 0.0,
                    "counts_precompute_seconds": (
                        direct_count_seconds
                    ),
                    "direct_count_compute_seconds": (
                        direct_count_seconds
                    ),
                    "counts_save_seconds": 0.0,
                    "uncertainty_compute_seconds": (
                        uncertainty_compute_seconds
                    ),
                    "effective_compute_seconds_for_k": (
                        direct_count_seconds
                        + uncertainty_compute_seconds
                    ),
                    "uncertainty_save_seconds": (
                        uncertainty_save_seconds
                    ),
                    "plot_seconds": 0.0,
                    "adata_write_seconds": 0.0,
                    "total_for_this_run_count_seconds": (
                        total_k_seconds
                    ),
                }
            )

            print(
                f"Standalone direct-count runtime: "
                f"{direct_count_seconds:.2f} seconds"
            )

            print(
                f"Uncertainty calculation runtime: "
                f"{uncertainty_compute_seconds:.2f} seconds"
            )

            print(
                f"TOTAL runtime for {k} runs: "
                f"{total_k_seconds:.2f} seconds"
            )

            del (
                spot_uncertainty,
                selected_codes,
                row_sums_counts,
                intra_sum_counts,
            )

    task_total_seconds = elapsed_seconds(task_start)
    if run_selection_mode == "sampled" and save_uncertainty:

        for k in run_counts:

            count = sampling_counts[k]

            spot_values = np.stack(
                spot_instability_values[k],
                axis=0,
            ).astype(np.float32, copy=False)

            sample_values = np.asarray(
                sample_instability_values[k],
                dtype=np.float64,
            )

            # k=50 is calculated only once because all 25 random
            # permutations contain exactly the same 50 runs.
            # Repeat it here purely so every sampled PKL has shape
            # (25, n_spots).
            if k == max_iterations and spot_values.shape[0] == 1:
                spot_values = np.repeat(
                    spot_values,
                    n_sampling_repeats,
                    axis=0,
                )

                sample_values = np.repeat(
                    sample_values,
                    n_sampling_repeats,
                )

            # Sanity check: every sampled PKL must contain 25 replicates.
            if spot_values.shape != (n_sampling_repeats, n):
                raise ValueError(
                    f"Unexpected spot uncertainty shape for k={k}: "
                    f"{spot_values.shape}; expected "
                    f"({n_sampling_repeats}, {n})"
                )

            if sample_values.shape != (n_sampling_repeats,):
                raise ValueError(
                    f"Unexpected sample uncertainty shape for k={k}: "
                    f"{sample_values.shape}; expected "
                    f"({n_sampling_repeats},)"
                )

            paths = make_output_paths(
                output_dir=output_dir,
                dataset=dataset,
                method=method,
                sample_name=sample_name,
                iterations=k,
                run_selection_mode="sampled",
            )

            joblib.dump(
                {
                    # shape = (25, n_spots)
                    "spot_uncertainty_repeats": spot_values,

                    # shape = (25,)
                    "uncertainty_repeats": sample_values,

                    "iterations": k,
                    "run_selection_mode": "sampled",
                    "n_sampling_repeats": n_sampling_repeats,
                    "sampling_seed": sampling_seed,

                    "sample_instability_mean": float(
                        sample_values.mean()
                    ),
                    "sample_instability_std": float(
                        sample_values.std(ddof=1)
                    ),
                    "sample_instability_median": float(
                        np.median(sample_values)
                    ),
                    "sample_instability_q025": float(
                        np.quantile(sample_values, 0.025)
                    ),
                    "sample_instability_q975": float(
                        np.quantile(sample_values, 0.975)
                    ),
                },
                paths["uncertainty"],
            )

    timing_frame = pd.DataFrame(timing_rows)
    timing_frame["total_dataset_sample_method_seconds"] = (
        task_total_seconds
    )


    timing_path = make_output_paths(
        output_dir=output_dir,
        dataset=dataset,
        method=method,
        sample_name=sample_name,
        iterations=run_counts[-1],
        run_selection_mode=run_selection_mode,
    )["timing"]
    timing_frame.to_csv(timing_path, index=False)
    if run_selection_mode == "sampled":

        metadata_frame = pd.DataFrame(metadata_rows)

        metadata_path = make_output_paths(
            output_dir=output_dir,
            dataset=dataset,
            method=method,
            sample_name=sample_name,
            iterations=run_counts[-1],
            run_selection_mode="sampled",
        )["metadata"]

        metadata_frame.to_csv(
            metadata_path,
            index=False,
        )

        print(f"Sampling metadata: {metadata_path}")

    print(
        "\n"
        + "=" * 80
        + f"\nFINISHED: dataset={dataset}, sample={sample_label}, "
        f"method={method}\n"
        + f"TOTAL dataset-sample-method runtime: "
        f"{task_total_seconds:.2f} seconds\n"
        + f"Timing file: {timing_path}\n"
        + "=" * 80
    )

    return timing_frame.to_dict(orient="records")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    full_script_start = time.perf_counter()

    datasets = [
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
        "SCAF4093_3229997_A1",
    ]

    dlpfc_sample_names = [
        "151507",
        "151508",
        "151509",
        "151510",
        "151669",
        "151670",
        "151671",
        "151672",
        "151673",
        "151674",
        "151675",
        "151676",
    ]

    methods = [
        "leiden",
        "louvain",
        "sedr",
        "graphst",
        "bayesspace",
        "stagate",
        "spicemix",
        "sedr_mclust",
    ]

    tasks = []

    for dataset in datasets:
        for method in methods:
            if (
                "full" in dataset
                or dataset == "Visium_HD_Human_Colon_Cancer"
            ) and method not in ("leiden", "louvain"):
                continue

            if (
                method == "stagate"
                and dataset == "mouse_brain_cerebellum"
            ):
                continue

            if method == "sedr_mclust" and dataset != "dlpfc":
                continue

            if dataset == "dlpfc":
                for sample_name in dlpfc_sample_names:
                    tasks.append((dataset, method, sample_name))
            else:
                tasks.append((dataset, method, None))

    n_jobs = min(len(tasks), mp.cpu_count(), 64)

    print(f"Submitting {len(tasks)} jobs with n_jobs={n_jobs}")
    print(f"Run counts: {RUN_COUNTS}")
    print(f"All outputs will be written below: {OUTPUT_ROOT}")
    print("Co-association matrices: disabled (direct exact calculation)")
    print("Each K is recomputed independently from fresh count arrays")

    all_timing_records = Parallel(
        n_jobs=n_jobs,
        verbose=10,
    )(
        delayed(process_dataset_method)(
            dataset=dataset,
            method=method,
            sample_name=sample_name,
            max_iterations=MAX_ITERATIONS,
            run_counts=RUN_COUNTS,
            save_uncertainty=SAVE_UNCERTAINTY,
            run_selection_mode=RUN_SELECTION_MODE,
            n_sampling_repeats=N_SAMPLING_REPEATS,
            sampling_seed=SAMPLING_SEED,
        )
        for dataset, method, sample_name in tasks
    )

    full_script_seconds = elapsed_seconds(full_script_start)

    completed_run_count_jobs = sum(
        len(records) for records in all_timing_records
    )

    # Save whole-script wall time into each task's timing CSV.
    unique_tasks = set()
    for task_records in all_timing_records:
        if not task_records:
            continue
        first = task_records[0]
        unique_tasks.add(
            (first["dataset"], first["sample"], first["method"])
        )

    for dataset, sample_name, method in unique_tasks:
        output_dir = f"{OUTPUT_ROOT}/{dataset}/{method}"
        timing_path = make_output_paths(
            output_dir=output_dir,
            dataset=dataset,
            method=method,
            sample_name=sample_name or None,
            iterations=MAX_ITERATIONS,
            run_selection_mode=RUN_SELECTION_MODE,
        )["timing"]

        timing_frame = pd.read_csv(timing_path)
        timing_frame["full_script_wall_clock_seconds"] = full_script_seconds
        timing_frame["full_script_wall_clock_minutes"] = (
            full_script_seconds / 60.0
        )
        timing_frame["full_script_wall_clock_hours"] = (
            full_script_seconds / 3600.0
        )
        timing_frame["completed_run_count_jobs_in_full_script"] = (
            completed_run_count_jobs
        )
        timing_frame.to_csv(timing_path, index=False)

    print(
        "\n"
        + "#" * 80
        + "\nALL CALCULATIONS FINISHED\n"
        + f"Completed dataset-sample-method-run-count calculations: "
          f"{completed_run_count_jobs}\n"
        + f"TOTAL WALL-CLOCK TIME FOR THE ENTIRE SCRIPT: "
          f"{full_script_seconds:.2f} seconds "
          f"({full_script_seconds / 60:.2f} minutes; "
          f"{full_script_seconds / 3600:.2f} hours)\n"
        + "#" * 80
    )