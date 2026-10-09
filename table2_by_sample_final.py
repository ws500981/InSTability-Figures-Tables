#!/usr/bin/env python3
"""Generate Table 2 and its final LaTeX table.

For each sample, average the method-specific fractions of spots with current
50-run instability >= 0.4. Cluster counts/fractions pool current 50-run
example-run clusters across available methods.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path("./data")
INSTABILITY_ROOT = BASE / "1-nested-non-sampling"
CLUSTER_CSV = BASE / "fig5/clusterwise_metrics_by_number_of_runs_new.csv"
OUT_DIR = BASE / "supp/t2"
OUT_CSV = OUT_DIR / "table2_by_sample.csv"
OUT_TEX = OUT_DIR / "table2_by_sample.tex"

METHODS = ["sedr", "graphst", "stagate", "bayesspace", "louvain", "leiden", "spicemix", "sedr_mclust"]
SAMPLES = [
    ("mouse_brain_cerebellum", None, "MBC"),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None, "CRC"),
    ("human_breast_cancer", None, "HBC"),
    ("mouse_brain", None, "MB"),
    ("coad_ffpe", None, "COAD"),
    ("ov_ffpe", None, "OV"),
] + [("dlpfc", sample, sample) for sample in [
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]]


def tag(dataset, sample):
    return f"{dataset}_{sample}" if sample else dataset


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


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan


def unstable_fraction(dataset, sample, method):
    sample_tag = tag(dataset, sample)
    path = INSTABILITY_ROOT / dataset / method / f"{sample_tag}_{method}_uncertainty_50runs.pkl"
    if not path.exists():
        return np.nan
    values = np.asarray(joblib.load(path)["spot_uncertainty"], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values >= 0.4)) if values.size else np.nan


def cluster_fraction_text(group):
    total = len(group)
    if total == 0:
        return "--"
    categories = group["cluster_category"]
    return "/".join(f"{((categories == x).sum() / total):.2f}" for x in ("s", "ms", "u", "hu"))


def format_mean_sd(mean, sd):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.2f} $\\pm$ --"
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def latex_table(table):
    lines = []
    for i, row in table.iterrows():
        if i == 6:
            lines += [r"\midrule", r"\multicolumn{4}{l}{\textbf{DLPFC}}\\"]
        lines.append(
            f"{row['Sample']} & {format_mean_sd(row['mean frac. unstable spots'], row['sd frac. unstable spots'])} & "
            f"{int(row['# clusters'])} & {row['fraction of clusters (s/ms/u/hu)']} \\\\"
        )
    body = "\n".join(lines)
    return rf"""\vspace{{2em}}

\caption{{Cluster instability by sample.}}
\label{{tab:bysample_new}}

\begin{{tabular}}{{
>{{\raggedright\arraybackslash}}m{{1.6cm}}%1.0cm}}
>{{\centering\arraybackslash}}m{{2cm}}%1.58cm}}
>{{\centering\arraybackslash}}m{{1.2cm}}%0.5cm}}
>{{\centering\arraybackslash}}m{{3cm}}%2.65cm}}
}}
\toprule
Sample & mean frac. unstable spots & \# clusters & fraction of clusters (s/ms/u/hu) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\vspace{{0.3em}}
\parbox{{0.95\textheight}}{{
\scriptsize \textit{{Note.}}
Rows correspond to different benchmark samples.
Second column is the fraction of unstable spots (with scores greater than or equal to 0.4) in each analysis (sample-method pair) averaged across all methods keeping the sample fixed (standard deviation also shown).
Third column is the total number of clusters produced from running up to 8 methods on the sample.
Fourth column is the fraction of those clusters in the four categories: $s=$ stable, $ms=$ modestly stable, $u=$ unstable, $hu=$ highly unstable.
%The remaining columns show the Pearson $R$ ($R^2$) correlation between cluster instability (mean score across spots) versus another cluster-level metric, e.g., cluster purity with respect to the reference annotation (all clusters produced for the sample, combining across the different methods, are used in the correlation analysis).
%Data underlying correlation analyses for cluster purity, expression Coherence, and expression Variance are shown in subfigure D of Figure 2 of the main text and Supplementary Figures S1--S16,S18,S20.
}}
\end{{sidewaystable}}
"""


def main():
    clusters = pd.read_csv(CLUSTER_CSV)
    allowed = {tag(dataset, sample) for dataset, sample, _ in SAMPLES}
    clusters = clusters[(clusters["iterations"] == 50) & clusters["dataset-sample"].isin(allowed)].copy()
    if "reference_mode" in clusters.columns:
        clusters = clusters[clusters["reference_mode"] == "example_50"].copy()
    clusters["cluster_category"] = clusters["clusterwise mean instability"].map(category)

    rows = []
    for dataset, sample, display in SAMPLES:
        sample_tag = tag(dataset, sample)
        group = clusters[clusters["dataset-sample"] == sample_tag]
        fractions = [unstable_fraction(dataset, sample, method) for method in METHODS]
        frac_mean, frac_sd = mean_sd(fractions)
        rows.append({
            "Sample": display,
            "mean frac. unstable spots": frac_mean,
            "sd frac. unstable spots": frac_sd,
            "# clusters": len(group),
            "fraction of clusters (s/ms/u/hu)": cluster_fraction_text(group),
        })

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, float_format="%.6f", na_rep="")
    OUT_TEX.write_text(latex_table(table), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nSaved:\n{OUT_CSV}\n{OUT_TEX}")


if __name__ == "__main__":
    main()
