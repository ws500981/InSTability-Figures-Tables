from pathlib import Path
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np

sampled = True

if sampled:
    ROOT = Path("./data/2-sampled")
else:
    # first k runs
    ROOT = Path("./data/1-nested-non-sampling")

UNCERTAINTY_KEY = (
    "spot_uncertainty_repeats"
    if sampled
    else "spot_uncertainty"
)

RUN_COUNTS = list(range(5, 51, 5))

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

METHOD_NAMES = {
    "leiden": "Leiden",
    "louvain": "Louvain",
    "bayesspace": "BayesSpace",
    "graphst": "GraphST",
    "stagate": "STAGATE",
    "spicemix": "SpiceMix",
    "sedr": "SEDR",
    "sedr_mclust": "SEDR (mclust)",
}

SAMPLES = [
    ("dlpfc", sample)
    for sample in [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676",
    ]
] + [
    ("mouse_brain", None),
    ("mouse_brain_cerebellum", None),
    ("human_breast_cancer", None),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None),
    ("ov_ffpe", None),
    ("coad_ffpe", None),
]


def get_path(dataset, sample, method, runs):
    tag = f"{dataset}_{sample}" if sample else dataset

    return (
        ROOT
        / dataset
        / method
        / f"{tag}_{method}_uncertainty_{runs}runs{'_sampled'if sampled else ''}.pkl"
    )

def combination_available(dataset, method):
    if method == "sedr_mclust" and dataset != "dlpfc":
        return False

    if method == "stagate" and dataset == "mouse_brain_cerebellum":
        return False

    return True


# ------------------------------------------------------------
# Calculate mean absolute difference relative to 50 runs
# ------------------------------------------------------------
data = defaultdict(list)

for method in METHODS:
    for dataset, sample in SAMPLES:

        # Skip method/dataset combinations that were never run
        if not combination_available(dataset, method):
            continue

        reference_path = get_path(dataset, sample, method, 50)

        if not reference_path.exists():
            print(f"[missing reference] {reference_path}")
            continue

        reference = np.asarray(
            joblib.load(reference_path)[UNCERTAINTY_KEY],
            dtype=float,
        )

        # For sampled=True, the 50-run file may also contain 25 replicates.
        # Since sampling 50 out of 50 runs should give the same result each time,
        # use one replicate as the 50-run reference.
        if sampled and reference.ndim == 2:
            if reference.shape[0] != 25:
                raise ValueError(
                    f"Expected 25 replicates in 50-run reference, "
                    f"got {reference.shape}: {reference_path}"
                )

            max_diff = np.max(np.abs(reference - reference[0]))

            if not np.allclose(reference, reference[0]):
                raise ValueError(
                    f"50-run replicates are not identical. "
                    f"Max difference = {max_diff}: {reference_path}"
                )

            reference = reference[0]

        values = []

        for runs in RUN_COUNTS:
            path = get_path(dataset, sample, method, runs)

            if not path.exists():
                print(f"[missing] {path}")
                values = None
                break

            scores = np.asarray(
                joblib.load(path)[UNCERTAINTY_KEY],
                dtype=float,
            )

            if sampled:
                if scores.ndim != 2:
                    raise ValueError(
                        f"Expected sampled scores to be 2D "
                        f"(25, n_spots), got {scores.shape}: {path}"
                    )

                if scores.shape[0] != 25:
                    raise ValueError(
                        f"Expected 25 replicates, got {scores.shape[0]}: {path}"
                    )

                if scores.shape[1] != reference.shape[0]:
                    raise ValueError(
                        f"Spot count mismatch: scores={scores.shape}, "
                        f"reference={reference.shape}: {path}"
                    )

                replicate_mae = np.mean(
                    np.abs(scores - reference[None, :]),
                    axis=1,
                )

                if not np.all(np.isfinite(replicate_mae)):
                    raise ValueError(
                        f"Non-finite replicate MAE found: {path}"
                    )

                values.append(replicate_mae.mean())
                if runs == 50:
                    print(
                        method,
                        dataset,
                        sample,
                        "50-run MAE:",
                        replicate_mae.mean(),
                    )

            else:
                values.append(
                    np.mean(np.abs(scores - reference))
                )

        if values is not None:
            data[method].append(values)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    4,
    figsize=(10, 5),
    sharex=True,
    sharey=True,
)

for ax, method in zip(axes.flat, METHODS):

    values = np.asarray(data[method], dtype=float)

    if len(values) == 0:
        print(f"[no data] {method}")
        ax.set_title(METHOD_NAMES[method], fontweight="bold")
        # ax.set_xticks(RUN_COUNTS)
        ax.set_xticks(range(10, 51, 10)) # label only multiples of 10 on x axis
        ax.grid(alpha=0.2, linewidth=0.5)
        continue

    print(method, values.shape)

    # Individual datasets/samples
    for y in values:
        ax.plot(
            RUN_COUNTS,
            y,
            color="gray",
            alpha=0.25,
            linewidth=0.7,
        )

    # Mean ± SEM
    mean = values.mean(axis=0)

    if len(values) > 1:
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    else:
        sem = np.zeros_like(mean)

    ax.fill_between(
        RUN_COUNTS,
        mean - sem,
        mean + sem,
        color="#2166ac",
        alpha=0.20,
    )

    ax.plot(
        RUN_COUNTS,
        mean,
        color="#2166ac",
        linewidth=2,
        marker="o",
        markersize=3,
    )

    ax.set_title(
        METHOD_NAMES[method],
        fontweight="bold",
    )

    # ax.set_xticks(RUN_COUNTS)
    ax.set_xticks(range(10, 51, 10)) # label only multiples of 10 on x axis
    ylim = -0.001 if sampled else -0.005
    ax.set_ylim(ylim, None)
    ax.grid(alpha=0.2, linewidth=0.5)


for ax in axes[-1]:
    ax.set_xlabel("Number of runs")

# for ax in axes[:, 0]:
#     ax.set_ylabel(
#         "|Mean spotwise diff.|\n"
#         "w.r.t. 50 runs"
#     )

fig.supylabel(
    #"|Mean spotwise diff.|\nw.r.t. 50 runs"
    "Mean absolute spotwise difference relative to 50 runs"
)

fig.tight_layout()

plt.savefig(f"./data/supp/s27_diff_r_convergence/s27_continuous_score_convergence{'_sampled' if sampled else ''}.png", dpi=600, bbox_inches="tight")
plt.savefig(f"./data/supp/s27_diff_r_convergence/s27_continuous_score_convergence{'_sampled' if sampled else ''}.pdf", dpi=600, bbox_inches="tight")
