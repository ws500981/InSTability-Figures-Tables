#!/usr/bin/env python3
"""Generate Table 5 and its final LaTeX table from current Figure 3 metrics."""
from pathlib import Path

import pandas as pd

BASE = Path("/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling")
METRICS_CSV = BASE / "zzz-all_plots/fig3/all_sample_method_metrics_by_number_of_runs.csv"
OUT_DIR = BASE / "zzz-all_plots/supp/t5"
OUT_CSV = OUT_DIR / "table5_bayesspace_entropy_instability.csv"
OUT_TEX = OUT_DIR / "table5_bayesspace_entropy_instability.tex"

SAMPLE_ORDER = [
    "MBC", "CRC", "HBC", "MB", "COAD", "OV",
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]


def latex_table(table):
    lines = []
    for i, row in table.iterrows():
        if i == 6:
            lines += [r"\midrule", r"\multicolumn{3}{l}{\textbf{DLPFC}}\\"]
        line = f"{row['Sample']} & {row['Mean Instability']:.2f} & {row['Mean Entropy']:.2f} "
        lines.append(line + r"\\")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[p]
\centering
\small
\caption[BayesSpace Entropy vs. Instability]{{\textbf{{BayesSpace Entropy vs. Instability.}} Mean instability of BayesSpace is the instability score averaged across all spots in the sample.
Mean entropy is the entropy averaged across all spots in the sample, then averaged across all 50 BayesSpace runs.
}}
\label{{tab:bayesspace_entropy_ari_instability}}
\setlength{{\tabcolsep}}{{7pt}}
\begin{{tabular}}{{lcc}}
\toprule
Sample
& Mean Instability & Mean Entropy \\
\midrule
{body}
\bottomrule
\end{{tabular}}
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
        rows.append({
            "Sample": sample,
            "Mean Instability": pd.to_numeric(source["bayesspace_mean_instability"], errors="coerce"),
            "Mean Entropy": pd.to_numeric(source["bayesspace_mean_entropy_all_runs"], errors="coerce"),
        })

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, float_format="%.6f", na_rep="")
    OUT_TEX.write_text(latex_table(table), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nSaved:\n{OUT_CSV}\n{OUT_TEX}")


if __name__ == "__main__":
    main()
