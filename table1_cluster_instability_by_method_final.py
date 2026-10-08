#!/usr/bin/env python3
"""Generate Table 1 and its final LaTeX table.

Uses current 50-run non-sampled spot instability and current Figure 5
clusterwise metrics. Unstable spots have instability >= 0.4. Cluster categories
are s/ms/u/hu = [0,.2)/[.2,.4)/[.4,.6)/[.6,1].
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = Path("/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling")
INSTABILITY_ROOT = BASE / "1-nested-non-sampling"
CLUSTER_CSV = BASE / "zzz-all_plots/fig5/clusterwise_metrics_by_number_of_runs_new.csv"
OUT_DIR = BASE / "zzz-all_plots/supp/t1"
OUT_CSV = OUT_DIR / "table1_cluster_instability_by_method.csv"
OUT_TEX = OUT_DIR / "table1_cluster_instability_by_method.tex"

DLPFC = [
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]
SAMPLES = [
    ("mouse_brain_cerebellum", None),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None),
    ("human_breast_cancer", None),
    ("mouse_brain", None),
    ("coad_ffpe", None),
    ("ov_ffpe", None),
] + [("dlpfc", sample) for sample in DLPFC]

# Main methods, then the DLPFC-only SEDR backend comparison.
METHOD_ROWS = [
    ("sedr", "all", "SEDR"),
    ("graphst", "all", "GraphST"),
    ("stagate", "all", "STAGATE"),
    ("bayesspace", "all", "BayesSpace"),
    ("louvain", "all", "Louvain"),
    ("leiden", "all", "Leiden"),
    ("spicemix", "all", "SpiceMix"),
    ("sedr", "dlpfc", "Leiden"),
    ("sedr_mclust", "dlpfc", "mclust"),
]
METRICS = {
    "Purity R": "clusterwise purity (precision)",
    "Recall R": "clusterwise recall",
    "Jaccard R": "clusterwise jaccard",
    "Exp. Coherence R": "expression coherence",
    "Exp. Variance R": "expression variance",
}
# Preserve the emphasis used in the supplied manuscript table.
BOLD_CELLS = {
    ("STAGATE", "Recall R"), ("STAGATE", "Jaccard R"),
    ("BayesSpace", "Purity R"), ("BayesSpace", "Recall R"),
    ("BayesSpace", "Jaccard R"), ("SpiceMix", "Purity R"),
}


def tag(dataset, sample):
    return f"{dataset}_{sample}" if sample else dataset


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan


def benchmark_cluster_data():
    """Keep the 18 benchmark samples and the 50-run example-run cluster rows."""
    df = pd.read_csv(CLUSTER_CSV)
    allowed = {tag(dataset, sample) for dataset, sample in SAMPLES}
    df = df[(df["iterations"] == 50) & df["dataset-sample"].isin(allowed)].copy()
    if "reference_mode" in df.columns:
        df = df[df["reference_mode"] == "example_50"].copy()
    return df


def spot_fraction_data():
    """One fraction of spots with instability >= 0.4 per available sample-method pair."""
    rows = []
    methods = sorted({method for method, _, _ in METHOD_ROWS})
    for dataset, sample in SAMPLES:
        sample_tag = tag(dataset, sample)
        for method in methods:
            path = INSTABILITY_ROOT / dataset / method / f"{sample_tag}_{method}_uncertainty_50runs.pkl"
            if not path.exists():
                continue
            values = np.asarray(joblib.load(path)["spot_uncertainty"], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                rows.append({
                    "dataset-sample": sample_tag,
                    "method": method,
                    "frac_unstable": float(np.mean(values >= 0.4)),
                })
    return pd.DataFrame(rows)


def category(value):
    if not np.isfinite(value):
        return None
    if value < 0.2:
        return "s"
    if value < 0.4:
        return "ms"
    if value < 0.6:
        return "u"
    return "hu"


def cluster_fraction_text(group):
    total = len(group)
    if total == 0:
        return "--"
    categories = group["cluster_category"]
    return "/".join(f"{((categories == x).sum() / total):.2f}" for x in ("s", "ms", "u", "hu"))


def finite_pearson_r(group, metric):
    x = pd.to_numeric(group["clusterwise mean instability"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(pearsonr(x, y).statistic)


def format_mean_sd(mean, sd):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.2f} $\\pm$ --"
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def format_r(row, metric):
    r = row[metric]
    r2 = row[f"{metric}^2"]
    text = "--" if pd.isna(r) else f"{r:.2f} ({r2:.2f})"
    if (row["Method"], metric) in BOLD_CELLS and text != "--":
        text = rf"\textbf{{{text}}}"
    return text


def latex_table(table):
    lines = []
    for i, row in table.iterrows():
        if i == 7:
            lines += [
                r"\midrule",
                r"\multicolumn{9}{l}{\textbf{SEDR with different backend clustering algorithms (DLPFC samples only)}} \\",
            ]
        cells = [
            row["Method"],
            format_mean_sd(row["mean frac. unstable spots"], row["sd frac. unstable spots"]),
            str(int(row["total # clusters"])),
            row["fraction of clusters (s/ms/u/hu)"],
        ] + [format_r(row, metric) for metric in METRICS]
        lines.append(" & ".join(cells) + r"\\")

    body = "\n".join(lines)
    return rf"""\begin{{sidewaystable}}[p]
\centering
\scriptsize
\caption{{Cluster instability  by method.}}
\label{{tab:bymethod_new}}

\begin{{tabular}}{{
>{{\raggedright\arraybackslash}}m{{1.6cm}}%1.4cm}}
>{{\centering\arraybackslash}}m{{2cm}}%1.58cm}}
>{{\centering\arraybackslash}}m{{1.2cm}}%0.5cm}}
>{{\centering\arraybackslash}}m{{3cm}}%2.65cm}}
>{{\raggedleft\arraybackslash}}m{{1.6cm}}
>{{\raggedleft\arraybackslash}}m{{1.6cm}}
>{{\raggedleft\arraybackslash}}m{{1.6cm}}
>{{\raggedleft\arraybackslash}}m{{2.6cm}}%1.6cm}}
>{{\raggedleft\arraybackslash}}m{{2.3cm}}%1.6cm}}
}}
\toprule
Method & mean frac. unstable spots & total \# clusters & fraction of clusters (s/ms/u/hu) & Purity R & Recall R & Jaccard R & Exp. Coherence R & Exp. Variance R \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\vspace{{0.3em}}
\parbox{{0.95\textheight}}{{
\scriptsize \textit{{Note.}}
Rows correspond to different clustering methods.
Second column is the fraction of unstable spots (with scores greater than or equal to 0.4) in each analysis (sample-method pair) averaged across all samples keeping the method fixed (standard deviation also shown).
Third column is the total number of clusters produced from running the method on up to 18 samples.
Fourth column is the fraction of those clusters in the four categories: $s=$ stable, $ms=$ modestly stable, $u=$ unstable, $hu=$ highly unstable.
The remaining columns show the Pearson $R$ ($R^2$) correlation between cluster instability (mean score across spots) versus another cluster-level metric, e.g., cluster purity with respect to the reference annotation (all clusters produced by the method, combining across the different samples, are used in the correlation analysis).
}}
%\end{{sidewaystable}}
"""


def main():
    clusters = benchmark_cluster_data()
    clusters["cluster_category"] = clusters["clusterwise mean instability"].map(category)
    spot_fractions = spot_fraction_data()

    rows = []
    for method, scope, display in METHOD_ROWS:
        cluster_group = clusters[clusters["method"] == method].copy()
        spot_group = spot_fractions[spot_fractions["method"] == method].copy()
        if scope == "dlpfc":
            cluster_group = cluster_group[cluster_group["dataset-sample"].str.startswith("dlpfc_")]
            spot_group = spot_group[spot_group["dataset-sample"].str.startswith("dlpfc_")]

        frac_mean, frac_sd = mean_sd(spot_group["frac_unstable"])
        row = {
            "Method": display,
            "analysis_scope": scope,
            "mean frac. unstable spots": frac_mean,
            "sd frac. unstable spots": frac_sd,
            "total # clusters": len(cluster_group),
            "fraction of clusters (s/ms/u/hu)": cluster_fraction_text(cluster_group),
        }
        for name, column in METRICS.items():
            r = finite_pearson_r(cluster_group, column)
            row[name] = r
            row[f"{name}^2"] = r * r if np.isfinite(r) else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, float_format="%.6f", na_rep="")
    OUT_TEX.write_text(latex_table(table), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nSaved:\n{OUT_CSV}\n{OUT_TEX}")


if __name__ == "__main__":
    main()
