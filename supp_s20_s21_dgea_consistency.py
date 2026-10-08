from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

plt.rcParams['font.family'] = 'Arial'


DGEA_DIR = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig4/alignment_dgea_data')
OUTPUT_DIR = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/supp/s20_s21_dgea_consistency')

METHODS = [
    "leiden",
    "louvain",
    "bayesspace",
    "graphst",
    "stagate",
    "spicemix",
    # "sedr", # sedr is used in main text, no need to be in supp
    "sedr_mclust"
]

METHOD_DISPLAY = {
    "leiden": "Leiden",
    "louvain": "Louvain",
    "sedr": "SEDR",
    "graphst": "GraphST",
    "bayesspace": "BayesSpace",
    "stagate": "STAGATE",
    "spicemix": "SpiceMix",
    "sedr_mclust": "SEDR (mclust)",
}

DATASET_SAMPLES = [
    *(('dlpfc', sample) for sample in [
        "151507", "151508", "151509", "151510", "151669", "151670",
        "151671", "151672", "151673", "151674", "151675", "151676",
    ]),
    ("mouse_brain", None),
    ("mouse_brain_cerebellum", None),
    ("human_breast_cancer", None),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None),
    ("ov_ffpe", None),
    ("coad_ffpe", None),
]
TOP_KS = [10, 20, 50, 200]
BIN_COLORS = {
    "[0.0,0.1)": "#08306b", "[0.1,0.2)": "#4292c6",
    "[0.2,0.3)": "#9ecae1", "[0.3,0.4)": "#deebf7",
    "[0.4,0.5)": "#fee391", "[0.5,0.6)": "#fec44f",
    "[0.6,0.7)": "#fdae6b", "[0.7,0.8)": "#fd8d3c",
    "[0.8,0.9)": "#fb6a4a", "[0.9,1.0]": "#cb181d",
}
FINE_BINS = list(BIN_COLORS)
FIG_DPI = 600
SAVE_PDF = True

METRICS = {
    "jaccard": {
        "column": "jaccard_mean",
        "ylabel": "Jaccard",
        "ylim": (0, 1),
    },
    "abs_rank_diff": {
        "column": "rank_abs_diff_shared_mean",
        "ylabel": "Abs rank diff", #(shared)
        "ylim": None,
    },
}



def safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def result_path(dataset, sample, method, suffix):
    sample_part = f"_{safe_name(sample)}" if sample is not None else ""
    return DGEA_DIR / (
        f"{safe_name(dataset)}{sample_part}_{safe_name(method)}_{suffix}"
    )


def topk(df, k):
    return df.sort_values("rank").head(int(k)).drop_duplicates("gene").copy()


def shared_rank_metrics(dgea_long):
    # For each run, mean absolute rank difference is calculated over shared genes only
    # these run-level means are then averaged across all non-example runs per contrast.
    rows = []

    for comparison, sub in dgea_long.groupby("comparison", sort=False):
        example_run = int(sub["example_run_idx"].iloc[0])  # this is the example run id
        runs = sorted(sub["run"].astype(int).unique())
        if example_run not in runs:
            continue

        # so i can find the total number of data points by comparing
        # and counting across 'run' and group_a_fine and group_b_fine
        # select the other run under 'run' label
        # but then what about cluster pairs?
        by_run = {run: sub.loc[sub["run"].eq(run)] for run in runs}
        for top_k in TOP_KS:
            example = topk(by_run[example_run], top_k).set_index("gene")["rank"]
            rank_diffs = []
            n_shared = []
            n_example_missing = []
            n_run_missing = []

            for run in runs:
                if run == example_run:
                    continue
                current = topk(by_run[run], top_k).set_index("gene")["rank"]
                shared = example.index.intersection(current.index)

                # Missing genes are excluded from rank differences.
                n_example_missing.append(len(example.index.difference(current.index)))
                n_run_missing.append(len(current.index.difference(example.index)))
                n_shared.append(len(shared))

                if len(shared):
                    rank_diffs.append(float(np.mean(np.abs(
                        example.loc[shared].to_numpy(float)
                        - current.loc[shared].to_numpy(float)
                    ))))

            rows.append({
                "comparison": comparison,
                "top_k": int(top_k),
                "rank_abs_diff_shared_mean": (
                    float(np.mean(rank_diffs)) if rank_diffs else np.nan
                ),
                "n_shared_genes_mean": (
                    float(np.mean(n_shared)) if n_shared else np.nan
                ),
                "n_example_genes_missing_in_run_mean": (
                    float(np.mean(n_example_missing)) if n_example_missing else np.nan
                ),
                "n_run_genes_missing_in_example_mean": (
                    float(np.mean(n_run_missing)) if n_run_missing else np.nan
                ),
                # Total unshared genes: missing from either top-K list.
                "n_missing_genes_mean": (
                    float(np.mean(np.asarray(n_example_missing) + np.asarray(n_run_missing)))
                    if n_example_missing else np.nan
                ),
            })

    return pd.DataFrame(rows)


def load_method(method):
    pooled = []
    summary_required = {
        "comparison", "top_k", "group_a_fine_bin", "group_b_fine_bin",
        "jaccard_mean",
    }
    long_required = {
        "comparison", "example_run_idx", "run", "gene", "rank",
    }

    for dataset, sample in DATASET_SAMPLES:
        summary_path = result_path(
            dataset, sample, method, "contrast_dgea_consistency.csv"
        )
        long_path = result_path(dataset, sample, method, "dgea_long_top200.csv")
        if not summary_path.exists() or not long_path.exists():
            print(f"[missing] {summary_path.name} or {long_path.name}")
            continue
        try:
            df = pd.read_csv(summary_path)
            dgea_long = pd.read_csv(long_path)
        except pd.errors.EmptyDataError:
            print(f"[empty] {summary_path.name} or {long_path.name}")
            continue

        missing = summary_required - set(df.columns)
        missing_long = long_required - set(dgea_long.columns)
        if missing or missing_long:
            print(
                f"[skip invalid schema] {summary_path.name}: "
                f"summary missing {sorted(missing)}, DGEA missing {sorted(missing_long)}"
            )
            continue

        dgea_long["run"] = pd.to_numeric(dgea_long["run"], errors="coerce")
        dgea_long["rank"] = pd.to_numeric(dgea_long["rank"], errors="coerce")
        dgea_long = dgea_long.dropna(subset=["run", "rank", "gene"]).copy()
        dgea_long["run"] = dgea_long["run"].astype(int)
        dgea_long["gene"] = dgea_long["gene"].astype(str)

        new_rank_metrics = shared_rank_metrics(dgea_long)
        df = df.drop(
            columns=[
                "rank_abs_diff_shared_mean", "n_shared_genes_mean",
                "n_missing_genes_mean",
                "n_example_genes_missing_in_run_mean",
                "n_run_genes_missing_in_example_mean",
            ],
            errors="ignore",
        ).merge(new_rank_metrics, on=["comparison", "top_k"], how="left")

        a_bin = df["group_a_fine_bin"].astype(str)  # use 10 numerical bins, rather than 4 grouped categories
        b_bin = df["group_b_fine_bin"].astype(str)
        df = df.loc[a_bin.eq(b_bin) & a_bin.isin(FINE_BINS)].copy()
        if df.empty:
            print(f"[no usable contrasts] {summary_path.name}")
            continue

        df["contrast_fine_bin"] = df["group_a_fine_bin"].astype(str)
        df["source_dataset"] = dataset
        df["source_sample"] = "" if sample is None else str(sample)
        df["source_file"] = summary_path.name
        pooled.append(df)
        print(f"[loaded] {summary_path.name}: {len(df)} rows")

    if not pooled:
        return pd.DataFrame()

    pooled = pd.concat(pooled, ignore_index=True)
    for column in [
        "top_k", "jaccard_mean", "rank_abs_diff_shared_mean",
        "n_shared_genes_mean", "n_missing_genes_mean",  # n_missing_genes is the ones that get removed on average across all comparisons
        "n_example_genes_missing_in_run_mean",
        "n_run_genes_missing_in_example_mean",
    ]:
        pooled[column] = pd.to_numeric(pooled[column], errors="coerce")
    return pooled.loc[
        pooled["top_k"].isin(TOP_KS)
        & pooled["contrast_fine_bin"].isin(FINE_BINS)  # contrast bin is just the bin being compared because a,b have same bin
    ].copy()


def finite_values(df, fine_bin, column):
    values = df.loc[df["contrast_fine_bin"].eq(fine_bin), column].to_numpy(float)
    return values[np.isfinite(values)]


def draw_panel(mi, ax, pooled, top_k, metric_name, metric, show_ylabel):
    column = metric["column"]
    sub = pooled.loc[pooled["top_k"].eq(top_k)]
    populated_bins = [
        b for b in FINE_BINS
        if finite_values(sub, b, column).size
    ]

    if not populated_bins:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No Top-{top_k} data",
            ha="center",
            va="center",
        )
        return

    max_bin_idx = max(FINE_BINS.index(b) for b in populated_bins)
    bins = FINE_BINS[:max_bin_idx + 1]

    positions = np.arange(1, len(bins) + 1)
    box_data = [finite_values(sub, b, column) for b in bins]
    colors = [BIN_COLORS[b] for b in bins]
    # Boxes show Q1–Q3, median lines, 1.5×IQR whiskers, and white-square means, outliers are hidden.

    boxplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.64,
        patch_artist=True,
        showfliers=True, #False,
        manage_ticks=False,
        whis=1.5,
    )
    
    for box, fine_bin in zip(boxplot["boxes"], bins):
        box.set(facecolor=BIN_COLORS[fine_bin], alpha=0.82,
                edgecolor="black", linewidth=1.1)
    
    for key in ["whiskers", "caps"]:
        for line in boxplot[key]:
            line.set(color="black", linewidth=1.1)
    
    for median in boxplot["medians"]:
        median.set(color="black", linewidth=1.5)

    for flier in boxplot["fliers"]:
        flier.set(color="black", markersize=4)

    # White square marks the arithmetic mean.
    #ax.scatter(
    #    positions,
    #    [np.mean(values) for values in box_data],
    #    marker="s",
    #    s=18,
    #    facecolor="white",
    #    edgecolor=colors,
    #    linewidth=1.0,
    #    zorder=5,
    #)

    ax.tick_params(axis='x', labelsize=12)
    ax.set_xticks(positions)

    if mi == len(METHODS) - 1:
        ax.set_xticklabels(bins,
            rotation=45,
            ha="right")
    else:
        ax.set_xticklabels([],
            rotation=45,
            ha="right")
    ax.set_xlim(0.25, len(bins) + 0.75)

    ax.tick_params(axis='y', labelsize=14)
    if metric_name == "jaccard":
        ax.set_ylim(0, 1.0)
        if show_ylabel:
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        else:
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels([])
    else:
        ymax_data = max(np.max(finite_values(sub, b, column)) for b in populated_bins)
        #ax.set_ylim(0, max(1.0, ymax_data)) # * 1.08))

        if top_k == 10:
            ymax = 6
            ax.set_ylim(0, 6)
            ax.set_yticks([0, 2, 4, 6])
        elif top_k == 20:
            ymax = 9
            ax.set_ylim(0, 9)
            ax.set_yticks([0, 3, 6, 9])
        elif top_k == 50:
            ymax = 25
            ax.set_ylim(0, 25)
            ax.set_yticks([0, 5, 10, 15, 20, 25])
        else:
            ymax = 80
            ax.set_ylim(0, 80)
            ax.set_yticks([0, 20, 40, 60, 80])

        if ymax_data > ymax:
            print(ymax_data)
            print(ymax)
            sys.exit("Error!")


    if mi == 0:
        ax.set_title(f"Top-{top_k} DEGs", fontsize=18, fontweight="bold")

    if show_ylabel:
        ax.set_ylabel(metric["ylabel"], fontsize=14)

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric_name == "jaccard":
        Ns = [str(len(box)) for box in box_data]
        text = "(" + r"$N=$" + ",".join(Ns) + ")"
        ax.text(0.75, 0.02, text, fontsize=8, ha="left", va="bottom", clip_on=False)

        if top_k == 10:
            text = f"{METHOD_DISPLAY[METHODS[mi]]}"
            ax.text(-3.5, 0.5, text, fontsize=19, fontweight="bold", ha="right", va="center", clip_on=False)
    else:
        if top_k == 10:
            text = f"{METHOD_DISPLAY[METHODS[mi]]}"
            ax.text(-3, 3, text, fontsize=19, fontweight="bold", ha="right", va="center", clip_on=False)


def main(metric_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric_name={metric_name!r}. "
            f"Choose from {list(METRICS)}."
        )

    metric = METRICS[metric_name]

    fig, axes = plt.subplots(
        nrows=len(METHODS),
        ncols=len(TOP_KS),
        figsize=(9.78, 18 * len(METHODS) / 7),
        constrained_layout=True,
)

    for mi, method in enumerate(METHODS):
        print(f"\nProcessing {method}")
        pooled = load_method(method)
        if pooled.empty:
            print(f"[skip] No usable nearest-match data for {method}")
            continue
        pooled.to_csv(
            OUTPUT_DIR / f"{safe_name(method)}_dgea_consistency_data.csv",
            index=False,
        )

        for index, top_k in enumerate(TOP_KS):
            print(top_k)
            text = draw_panel(mi,
                axes[mi][index],
                pooled,
                top_k,
                metric_name,
                metric,
                show_ylabel=index == 0,
            )

    plt.tight_layout(rect=(0, 0.33, 1, 1))     # needed if not plotting p-values

    plt.subplots_adjust(wspace=0.2, hspace=0.4)  # adjusts spacing between subplots

    stem = OUTPUT_DIR / f"s20_21_dgea_consistency_{metric_name}"
    fig.savefig(stem.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main("jaccard")
    main("abs_rank_diff")
