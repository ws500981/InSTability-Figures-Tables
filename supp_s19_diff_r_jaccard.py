#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RESULTS_DIR = Path('./data/fig4/alignment_dgea_data')

OUTPUT_DIR = Path('./data/supp/s19_diff_r_jaccard')

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
    ("dlpfc", "151507"),
    ("dlpfc", "151508"),
    ("dlpfc", "151509"),
    ("dlpfc", "151510"),
    ("dlpfc", "151669"),
    ("dlpfc", "151670"),
    ("dlpfc", "151671"),
    ("dlpfc", "151672"),
    ("dlpfc", "151673"),
    ("dlpfc", "151674"),
    ("dlpfc", "151675"),
    ("dlpfc", "151676"),
    ("mouse_brain", None),
    ("mouse_brain_cerebellum", None),
    ("human_breast_cancer", None),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None),
    ("ov_ffpe", None),
    ("coad_ffpe", None),
]

BIN_COLORS = {
    "[0.0,0.1)": "#08306b",
    "[0.1,0.2)": "#4292c6",
    "[0.2,0.3)": "#9ecae1",
    "[0.3,0.4)": "#deebf7",
    "[0.4,0.5)": "#fee391",
    "[0.5,0.6)": "#fec44f",
    "[0.6,0.7)": "#fdae6b",
    "[0.7,0.8)": "#fd8d3c",
    "[0.8,0.9)": "#fb6a4a",
    "[0.9,1.0]": "#cb181d",
}
FINE_BINS = list(BIN_COLORS)

FIG_DPI = 600
SAVE_PDF = True
RANDOM_SEED = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["pdf.compression"] = 9


# -----------------------------------------------------------------------------
# Loading and summarization
# -----------------------------------------------------------------------------
def safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def result_path(dataset, sample, method):
    sample_part = f"_{safe_name(sample)}" if sample is not None else ""
    return RESULTS_DIR / (
        f"{safe_name(dataset)}{sample_part}_{safe_name(method)}"
        "_best_cluster_matches.csv"
    )


def load_method_matches(method):
    parts = []

    for dataset, sample in DATASET_SAMPLES:
        path = result_path(dataset, sample, method)
        if not path.exists():
            print(f"[missing] {path.name}")
            continue

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"[empty] {path.name}")
            continue

        required = {
            "example_cluster",
            "run",
            "example_run_idx",
            "best_jaccard",
            "overlap",
            "example_cluster_size",
            "run_cluster_size",
            "example_mean_instability",
            "example_fine_bin",
        }
        missing = required - set(df.columns)
        if missing:
            print(f"[invalid] {path.name}: missing {sorted(missing)}")
            continue

        for column in [
            "run",
            "example_run_idx",
            "best_jaccard",
            "example_mean_instability",
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Exclude the example run because its self-match has Jaccard = 1.
        df = df.loc[
            df["run"].ne(df["example_run_idx"])
            & df["best_jaccard"].notna()
            & df["example_mean_instability"].notna()
            & df["example_fine_bin"].astype(str).isin(FINE_BINS)
        ].copy()
        if df.empty:
            print(f"[no non-example matches] {path.name}")
            continue

        df["source_dataset"] = dataset
        df["source_sample"] = "" if sample is None else str(sample)
        df["source_method"] = method
        df["example_fine_bin"] = df["example_fine_bin"].astype(str)
        parts.append(df)
        print(f"[loaded] {path.name}: {len(df)} non-example matches")

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_clusters(matches):
    #one row per example cluster
    id_columns = [
        "source_dataset",
        "source_sample",
        "source_method",
        "example_cluster",
        "example_mean_instability",
        "example_fine_bin",
        "example_cluster_size",
    ]

    return (
        matches.groupby(id_columns, dropna=False, as_index=False)
        .agg(
            n_nonexample_runs=("run", "nunique"),
            best_jaccard_mean=("best_jaccard", "mean"),
            best_jaccard_median=("best_jaccard", "median"),
            best_jaccard_sd=("best_jaccard", "std"),
            best_jaccard_min=("best_jaccard", "min"),
            best_jaccard_q25=("best_jaccard", lambda x: x.quantile(0.25)),
            best_jaccard_q75=("best_jaccard", lambda x: x.quantile(0.75)),
            best_jaccard_max=("best_jaccard", "max"),
            overlap_mean=("overlap", "mean"),
            matched_run_cluster_size_mean=("run_cluster_size", "mean"),
        )
    )


def correlation_results(cluster_summary):
    valid = cluster_summary[
        ["example_mean_instability", "best_jaccard_mean"]
    ].dropna()

    if len(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(valid)

    x = valid["example_mean_instability"]
    y = valid["best_jaccard_mean"]

    spearman = spearmanr(x, y)
    pearson = pearsonr(x, y)

    return (
        float(spearman.statistic),
        float(spearman.pvalue),
        float(pearson.statistic),
        float(pearson.pvalue),
        len(valid),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def style_boxplot(boxplot, colors):
    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor("black")
        box.set_linewidth(1.1)

    for key in ["whiskers", "caps"]:
        for line in boxplot[key]:
            line.set_color("black")
            line.set_linewidth(1.1)

    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    #for flier in boxplot["fliers"]:
    #    flier.set(color="black", markersize=4)


def plot_method(method, matches, cluster_summary, mi, ax_scatter, ax_box):
    populated_bins = [
        fine_bin
        for fine_bin in FINE_BINS
        if matches.loc[
            matches["example_fine_bin"].eq(fine_bin), "best_jaccard"
        ].notna().any()
    ]
    if not populated_bins:
        return

    # left plot: each bin combines a boxplot and right-sided half violin (KDE).
    box_data = [
        matches.loc[
            matches["example_fine_bin"].eq(fine_bin), "best_jaccard"
        ].dropna().to_numpy(float)
        for fine_bin in populated_bins
    ]
    positions = np.arange(1, len(populated_bins) + 1, dtype=float)
    box_positions = positions - 0.14
    violin_positions = positions + 0.10
    colors = [BIN_COLORS[b] for b in populated_bins]

    # Boxes show Q1–Q3, median lines,
    # 1.5×IQR whiskers, and 
    # white-square means, 
    # outliers are hidden.
    boxplot = ax_box.boxplot(
        box_data,
        positions=box_positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        manage_ticks=False,
    )
    style_boxplot(boxplot, colors)

    # Mean marker inside each box.
    #ax_box.scatter(
    #    box_positions,
    #    [np.mean(values) for values in box_data],
    #    marker="s",
    #    s=18,
    #    facecolor="white",
    #    edgecolor=colors,
    #    linewidth=1.0,
    #    zorder=5,
    #)

    for values, color, violin_position in zip(
        box_data, colors, violin_positions
    ):
        # Matplotlib creates a full violin.
        #Clamp its left half to the center line to obtain the right-sided density.
        if len(values) >= 2 and np.ptp(values) > 0:
            violin = ax_box.violinplot(
                [values],
                positions=[violin_position],
                widths=0.55,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
            )
            body = violin["bodies"][0]
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], violin_position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(1.1)
            body.set_alpha(0.6)
            body.set_zorder(1)

    ax_box.tick_params(axis='x', labelsize=13.5)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(populated_bins, rotation=45, ha="right")
    ax_box.set_xlim(0.45, len(populated_bins) + 0.65)
    ax_box.set_ylim(0, 1)
    #ax_box.set_title(
    #    "(distribution)",
    #    fontweight="bold",
    #    fontsize=19,
    #)

    ax_box.tick_params(axis='y', labelsize=13.5)
    ax_box.set_ylim(0, 1)
    ax_box.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_box.set_yticklabels([])

    ax_box.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)

    # right plot: one point per example cluster.
    Ns = []
    rng = np.random.default_rng(RANDOM_SEED)
    for fine_bin in populated_bins:
        sub = cluster_summary[
            cluster_summary["example_fine_bin"].eq(fine_bin)
        ]
        if sub.empty:
            continue
        jitter = rng.normal(0, 0.0025, size=len(sub))

        Ns.append(str(len(sub["example_mean_instability"].to_numpy(float))))

        ax_scatter.scatter(
            sub["example_mean_instability"].to_numpy(float) + jitter,
            sub["best_jaccard_mean"],
            s=34,
            color=BIN_COLORS[fine_bin],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
        )

    # Linear least-squares trend line using one point per example cluster.
    fit_data = cluster_summary[
        ["example_mean_instability", "best_jaccard_mean"]
    ].dropna()
    if len(fit_data) >= 2 and fit_data["example_mean_instability"].nunique() >= 2:
        x_fit = fit_data["example_mean_instability"].to_numpy(float)
        y_fit = fit_data["best_jaccard_mean"].to_numpy(float)
        slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = slope * x_line + intercept
        ax_scatter.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=1.75,
            #label="Linear best fit",
            zorder=4,
        )
        #ax_scatter.legend(frameon=False, loc="upper right", fontsize=8)

    rho, spearman_p, pearson_r, pearson_p, n_clusters = correlation_results(
        cluster_summary
    )

    correlation_text = (
        f"Spearman ρ = {rho:.3f}, p = {spearman_p:.3g}\n"
        f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}\n"
        f"n = {n_clusters} clusters"
    )
    print(correlation_text)

    #ax_scatter.text(
    #    0.04,
    #    0.05,
    #    correlation_text,
    #    transform=ax_scatter.transAxes,
    #    ha="left",
    #    va="bottom",
    #    fontsize=9,
    #    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    #)
    
    ax_scatter.set_xlim(0, 0.7)
    ax_scatter.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    ax_scatter.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    ax_scatter.tick_params(axis='x', labelsize=13.5)
    ax_scatter.tick_params(axis='y', labelsize=13.5)

    #xlab_text = "Example Cluster Instability\n(" + r"$N=$" + ",".join(Ns) + ")"    
    #ax_scatter.set_xlabel(xlab_text, fontsize=15)
    ax_scatter.set_xlabel("Cluster Instability - Example Run", fontsize=15)
    ax_scatter.set_ylabel("Jaccard w.r.t. Match", fontsize=15)  # Mean across runs
    ax_scatter.set_title(f"Cross-run Consistency of {METHOD_DISPLAY[method]}",
                         fontweight="bold",
                         fontsize=19, x=1.0) #,
                         #color="white")
    
    ax_scatter.grid(linestyle=":", linewidth=0.6, alpha=0.4)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    # Use so that can copy into illustrator...
    text = "(" + r"$N=$" + ",".join(Ns) + ")" 
    ax_scatter.text(0.35, -0.37, text, fontsize=16, ha="center", va="bottom", clip_on=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(METHODS),
        ncols=2,
        figsize=(9.78, 24 * len(METHODS) / 7),
        constrained_layout=True,
    )

    for mi, method in enumerate(METHODS):
        print(f"\nProcessing {method}")

        matches = load_method_matches(method)
        if matches.empty:
            print(f"[skip] No usable matching data for {method}")
            continue

        cluster_summary = summarize_clusters(matches)
        matches.to_csv(
            OUTPUT_DIR / f"{safe_name(method)}_pooled_best_matches.csv",
            index=False,
        )
        cluster_summary.to_csv(
            OUTPUT_DIR / f"{safe_name(method)}_cluster_jaccard_summary.csv",
            index=False,
        )

        ax_scatter = axes[mi][0]
        ax_box = axes[mi][1]

        plot_method(method, matches, cluster_summary, mi, ax_scatter, ax_box)

        print(f"[finished] {method}")

    plt.tight_layout(rect=(0, 0, 1, 1)) # Uncomment if not plotting p-values

    #plt.subplots_adjust(wspace=1.0, hspace=1.0)  # adjusts spacing between subplots

    output_stem = OUTPUT_DIR / f"s19_cross-run-consistency-vs-instability"
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    if SAVE_PDF:
        fig.savefig(
            output_stem.with_suffix(".pdf"),
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)



if __name__ == "__main__":
    main()
