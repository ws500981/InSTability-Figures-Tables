#!/usr/bin/env python3
"""Generate Table 3 and its final LaTeX table from current Figure 3 metrics."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling")
METRICS_CSV = BASE / "zzz-all_plots/fig3/all_sample_method_metrics_by_number_of_runs.csv"
OUT_DIR = BASE / "zzz-all_plots/supp/t3"
OUT_CSV = OUT_DIR / "table3_mean_ari_sample_method.csv"
SUMMARY_CSV = OUT_DIR / "table3_mean_ari_method_summary.csv"
OUT_TEX = OUT_DIR / "table3_mean_ari_sample_method.tex"

METHODS = [
    ("sedr", "SEDR"), ("graphst", "GraphST"), ("stagate", "STAGATE"),
    ("bayesspace", "BayesSpace"), ("louvain", "Louvain"),
    ("leiden", "Leiden"), ("spicemix", "SpiceMix"),
    ("sedr_mclust", "SEDR (mclust)"),
]
SAMPLE_ORDER = [
    "MBC", "CRC", "HBC", "MB", "COAD", "OV",
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]
DLPFC = set(SAMPLE_ORDER[6:])


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan


def fmt_value(value, bold=False):
    if not np.isfinite(value):
        return "--"
    text = f"{value:.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_mean_sd(mean, sd):
    if not np.isfinite(mean):
        return "--"
    sd_text = "--" if not np.isfinite(sd) else f"{sd:.2f}"
    return rf"${mean:.2f} \pm {sd_text}$"


def method_summary(table, samples):
    sub = table[table["Sample"].isin(samples)]
    row = {}
    for _, display in METHODS:
        row[display] = mean_sd(pd.to_numeric(sub[display], errors="coerce"))
    return row


def latex_table(table, all_summary, dlpfc_summary):
    lines = []
    method_names = [display for _, display in METHODS]
    for i, row in table.iterrows():
        if i == 6:
            lines += [r"\midrule", r"\multicolumn{10}{l}{\textbf{DLPFC}}\\"]

        numeric = np.asarray([row[name] for name in method_names], dtype=float)
        finite = np.isfinite(numeric)
        max_value = np.nanmax(numeric) if finite.any() else np.nan
        cells = [
            fmt_value(value, np.isfinite(value) and np.isclose(value, max_value, rtol=0, atol=1e-12))
            for value in numeric
        ]
        mean, sd = mean_sd(numeric)
        lines.append(" & ".join([row["Sample"], *cells, fmt_mean_sd(mean, sd)]) + r" \\")

    lines += [r"\midrule", r"\multicolumn{10}{l}{\textbf{Mean $\pm$ S.D.}}\\"]

    all_cells = []
    for method, display in METHODS:
        all_cells.append("--" if method == "sedr_mclust" else fmt_mean_sd(*all_summary[display]))
    lines.append(" & ".join(["All", *all_cells, ""]) + r" \\")

    dlpfc_cells = [fmt_mean_sd(*dlpfc_summary[display]) for _, display in METHODS]
    lines.append(" & ".join(["DLPFC only", *dlpfc_cells, ""]) + r" \\")

    body = "\n".join(lines)
    return rf"""\begin{{table}}[p]
\centering
\scriptsize
\caption{{Mean ARI across 50 runs for each sample--method pair.}}
\label{{tab:mean_ari_sample_method}}
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrr}}
\toprule
Sample & SEDR & GraphST & STAGATE & Bayes & Louvain & Leiden & SpiceMix & SEDR & Mean $\pm$ S.D. \\
&  &  & & Space &  & & & mclust &  \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def main():
    df = pd.read_csv(METRICS_CSV)
    df = df[df["iterations"] == 50].copy()
    if "example_mode" in df.columns:
        df = df[df["example_mode"] == "example_50"].copy()
    df = df.drop_duplicates("sample_label", keep="last").set_index("sample_label")

    rows = []
    for sample in SAMPLE_ORDER:
        source = df.loc[sample]
        row = {"Sample": sample}
        for method, display in METHODS:
            row[display] = pd.to_numeric(source[f"{method}_mean_ari_all_runs"], errors="coerce")
        rows.append(row)
    table = pd.DataFrame(rows)

    all_summary = method_summary(table, SAMPLE_ORDER)
    dlpfc_summary = method_summary(table, DLPFC)
    summary_rows = []
    for scope, summary in [("All", all_summary), ("DLPFC only", dlpfc_summary)]:
        for _, display in METHODS:
            mean, sd = summary[display]
            summary_rows.append({"Scope": scope, "Method": display, "Mean": mean, "SD": sd})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, float_format="%.6f", na_rep="")
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False, float_format="%.6f", na_rep="")
    OUT_TEX.write_text(latex_table(table, all_summary, dlpfc_summary), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nSaved:\n{OUT_CSV}\n{SUMMARY_CSV}\n{OUT_TEX}")


if __name__ == "__main__":
    main()
