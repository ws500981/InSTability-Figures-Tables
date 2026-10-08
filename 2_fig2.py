
# The three full samples need fixing.

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
DATA_PATH = Path('/home/wuw15/data_dir/ST_datasets/AAA_with_uncert')
OUT_DIR = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig2')
OUT_DIR.mkdir(parents=True, exist_ok=True)
METHODS = [('leiden', 'Leiden'), ('louvain', 'Louvain'), ('bayesspace', 'BayesSpace'), ('graphst', 'GraphST'), ('stagate', 'STAGATE'), ('spicemix', 'SpiceMix'), ('sedr', 'SEDR'), ('sedr_mclust', 'SEDR (mclust)')]
DATASETS = ['dlpfc', 'mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe', 'Visium_HD_Human_Colon_Cancer', 'coad_ffpe_full', 'ov_ffpe_full', 'SCAF4093_3229997_A1']
DLPFC_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
DATASET_DISPLAY = {'dlpfc': 'DLPFC', 'mouse_brain': 'Mouse Brain', 'mouse_brain_cerebellum': 'Mouse Brain Cerebellum', 'human_breast_cancer': 'Human Breast Cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square': 'CRC', 'Visium_HD_Human_Colon_Cancer': 'CRC (full sample)', 'coad_ffpe': 'COAD', 'ov_ffpe': 'OV', 'coad_ffpe_full': 'COAD (full sample)', 'ov_ffpe_full': 'OV (full sample)', 'SCAF4093_3229997_A1': 'SCAF4093-3229997-A1'}
CLUSTER_COLORS = ['#3049ad', '#fe8011', '#1b7837', '#fa0000', '#ab43fc', '#8d574c', '#ff00d9', '#bcbd22', '#17becf', '#8baaf3', '#ffbb79', '#99df8b', '#fe7775', '#c6b1d4', '#c49d95', '#ff80c6', '#dcdb91', '#a7d1e6', '#393b79', '#8c6d31', '#0aac00', '#982109', '#7b4173', '#713230', '#ff008c', '#637939', '#e7cb94', '#ccefc5', '#efcece', '#f7b6d2', '#eeedc8']
# Full 0.0-1.0 instability legend.
BINS = [('[0.0,0.1)', 0.0, 0.1, '#08306b'), ('[0.1,0.2)', 0.1, 0.2, '#4292c6'), ('[0.2,0.3)', 0.2, 0.3, '#9ecae1'), ('[0.3,0.4)', 0.3, 0.4, '#deebf7'), ('[0.4,0.5)', 0.4, 0.5, '#fee391'), ('[0.5,0.6)', 0.5, 0.6, '#fec44f'), ('[0.6,0.7)', 0.6, 0.7, '#fdae6b'), ('[0.7,0.8)', 0.7, 0.8, '#fd8d3c'), ('[0.8,0.9)', 0.8, 0.9, '#fb6a4a'), ('[0.9,1.0]', 0.9, 1.0, '#cb181d')]
BIN_COLORS = [x[3] for x in BINS]
HD_DATASETS = {'Visium_HD_Human_Colon_Cancer_cropped_square', 'Visium_HD_Human_Colon_Cancer', 'ov_ffpe', 'coad_ffpe', 'ov_ffpe_full', 'coad_ffpe_full', 'SCAF4093_3229997_A1'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 9, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'pdf.fonttype': 42, 'ps.fonttype': 42})

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def pretty_dataset(ds):
    return DATASET_DISPLAY.get(ds, ds.replace('_', ' ').title())

def make_runs():
    runs = []
    for ds in DATASETS:
        if ds == 'dlpfc':
            runs.extend(((ds, sample) for sample in DLPFC_SAMPLES))
        else:
            runs.append((ds, None))
    return runs

def bin_indices(values):
    """Return 0..9 instability-bin indices."""
    values = np.asarray(values, dtype=float)
    idx = np.digitize(values, bins=np.arange(0.1, 1.0, 0.1), right=False)
    idx = np.clip(idx, 0, len(BINS) - 1)
    idx[~np.isfinite(values)] = -1
    return idx

def bin_counts(values):
    idx = bin_indices(values)
    return np.array([(idx == i).sum() for i in range(len(BINS))], dtype=int)

def category_order(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories]
    return [str(x) for x in pd.unique(series.dropna().astype(str))]

# -----------------------------------------------------------------------------
# Color alignment
# -----------------------------------------------------------------------------

def reference_palette(annotation):
    cats = category_order(annotation)
    return {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(cats)}

def aligned_prediction_palette(annotation, prediction, gt_palette):
    """
    Globally match predicted clusters to reference labels using
    maximum Jaccard overlap.

    Colors are still taken only from the existing gt_palette /
    CLUSTER_COLORS. No blending or new colors are created.
    """
    pred = pd.Series(prediction, index=annotation.index).astype(str)
    gt = annotation.astype('string')
    valid = gt.notna()
    pred_cats = list(pd.unique(pred))
    gt_cats = list(gt_palette.keys())
    palette = {}
    if valid.sum() == 0:
        return {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(pred_cats)}
    # Jaccard matrix: predicted clusters x reference labels.
    scores = np.zeros((len(pred_cats), len(gt_cats)))
    for i, p in enumerate(pred_cats):
        p_mask = (pred == p) & valid
        for j, g in enumerate(gt_cats):
            g_mask = (gt == g) & valid
            intersection = np.sum(p_mask & g_mask)
            union = np.sum(p_mask | g_mask)
            if union > 0:
                scores[i, j] = intersection / union
    # Globally optimal one-to-one assignment.
    row_ind, col_ind = linear_sum_assignment(-scores)
    assigned_pred = set()
    for i, j in zip(row_ind, col_ind):
        if scores[i, j] <= 0:
            continue
        pred_label = pred_cats[i]
        gt_label = gt_cats[j]
        color = gt_palette[gt_label]
        palette[pred_label] = color
        assigned_pred.add(pred_label)
    # Extra predicted clusters use colors not already assigned to references.
    reference_colors = set(gt_palette.values())
    unused_colors = [c for c in CLUSTER_COLORS if c not in reference_colors]
    extra_i = 0
    for pred_label in pred_cats:
        if pred_label in assigned_pred:
            continue
        palette[pred_label] = unused_colors[extra_i % len(unused_colors)]
        extra_i += 1
    return palette

# -----------------------------------------------------------------------------
# Spatial drawing
# -----------------------------------------------------------------------------

def get_spatial_context(adata, dataset):
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    image = None
    spatial = adata.uns.get('spatial', {})
    if isinstance(spatial, dict) and spatial:
        library = spatial[next(iter(spatial))]
        images = library.get('images', {})
        scales = library.get('scalefactors', {})
        for key in ('hires', 'lowres'):
            if key in images:
                image = np.asarray(images[key])
                scale = float(scales.get(f'tissue_{key}_scalef', 1.0))
                coords = coords * scale
                break
    unique = np.unique(coords, axis=0)
    if len(unique) > 1:
        distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
        distances = distances[np.isfinite(distances) & (distances > 0)]
        spacing = float(np.median(distances))
    else:
        spacing = 1.0
    # Build spot geometry once and reuse it in every panel.
    if dataset in HD_DATASETS:
        half = spacing / 2
        offsets = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    elif dataset == 'mouse_brain':
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    elif dataset in {'dlpfc', 'human_breast_cancer'}:
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        radius = 0.48 * spacing
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    vertices = coords[:, None, :] + offsets[None, :, :]
    return {'coords': coords, 'image': image, 'vertices': vertices, 'spacing': spacing}

def grayscale(image):
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    rgb = image[..., :3].astype(float)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

def draw_spatial(ax, ctx, labels, palette):
    image = ctx['image']
    coords = ctx['coords']
    if image is not None:
        ax.imshow(grayscale(image), cmap='gray', alpha=0.5, origin='upper')
    labels = pd.Series(labels).astype('string')
    colors = [palette.get(str(x), '#d9d9d9') if not pd.isna(x) else '#d9d9d9' for x in labels]
    collection = PolyCollection(ctx['vertices'], facecolors=colors, edgecolors='none', linewidths=0, rasterized=True)
    ax.add_collection(collection)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    px = 0.035 * max(x1 - x0, 1)
    py = 0.035 * max(y1 - y0, 1)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y1 + py, y0 - py)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color('black')

def blank_panel(ax, text=''):
    ax.set_xticks([])
    ax.set_yticks([])
    if text:
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center', fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

# -----------------------------------------------------------------------------
# Instability legend
# -----------------------------------------------------------------------------

def draw_instability_legend(fig, left, right):
    ax = fig.add_axes([left, 0.1, right - left, 0.145])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis('off')
    groups = [('Stable', 0, 2), ('Modestly Stable', 2, 4), ('Unstable', 4, 6), ('Highly Unstable', 6, 10)]
    for group_name, start, stop in groups:
        x0 = start + 0.03
        width = stop - start - 0.06
        ax.add_patch(Rectangle((x0, 0.5), width, 0.28, fill=False, edgecolor='0.35', linewidth=0.7))
        for i in range(start, stop):
            item_x = i + 0.12
            ax.add_patch(Rectangle((item_x, 0.57), 0.15, 0.14, facecolor=BINS[i][3], edgecolor='none'))
            ax.text(item_x + 0.19, 0.64, BINS[i][0], ha='left', va='center', fontsize=8, fontweight='bold')
        ax.text((start + stop) / 2, 0.22, group_name, ha='center', va='center', fontsize=10.5, fontweight='bold')

# -----------------------------------------------------------------------------
# One complete Figure 2 output for a dataset/sample
# -----------------------------------------------------------------------------

def plot_one(dataset, sample=None):
    suffix = f'_{sample}' if sample else ''
    infile = DATA_PATH / f'{dataset}{suffix}.h5ad'
    print(f'Reading {infile}')
    adata = ad.read_h5ad(infile)
    ctx = get_spatial_context(adata, dataset)
    has_reference = 'annotation' in adata.obs and adata.obs['annotation'].notna().any()
    if has_reference:
        gt_palette = reference_palette(adata.obs['annotation'])
    else:
        gt_palette = {}
    # Plot only methods that have uncertainty values for this sample.
    plot_methods = [(method, display) for method, display in METHODS if f'{method}_uncertainty' in adata.obs]
    n_methods = len(plot_methods)
    spot_counts = np.zeros((n_methods, len(BINS)), dtype=int)
    cluster_counts = np.zeros((n_methods, len(BINS)), dtype=int)
    fig_width = 5.0 + 1.25 * n_methods
    fig = plt.figure(figsize=(fig_width, 4.05))
    ncols = n_methods + 4
    gs = fig.add_gridspec(2, ncols, width_ratios=[1.0, 0.3, *[1.0] * n_methods, 0.4, 1.18], hspace=0.025, wspace=0.08)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.91, bottom=0.235)
    ax_ref = fig.add_subplot(gs[0, 0])
    if has_reference:
        draw_spatial(ax_ref, ctx, adata.obs['annotation'], gt_palette)
    else:
        blank_panel(ax_ref, 'No reference')
    ax_ref.set_title('Reference', fontsize=11, pad=4, fontweight='bold')
    ax_ref.text(-0.19, 1.08, 'A', transform=ax_ref.transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    ax_unused = fig.add_subplot(gs[1, 0])
    ax_unused.axis('off')
    top_axes = []
    bottom_axes = []
    for method_i, (method, display) in enumerate(plot_methods):
        col = method_i + 2
        ax_top = fig.add_subplot(gs[0, col])
        ax_bottom = fig.add_subplot(gs[1, col])
        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)
        ax_top.set_title(display, fontsize=10, pad=4, fontweight='bold')
        uncertainty_key = f'{method}_uncertainty'
        results_key = f'{method}_results'
        u = adata.obs[uncertainty_key].to_numpy(dtype=float)
        u_bin = bin_indices(u)
        instability_labels = pd.Series([BINS[i][0] if i >= 0 else pd.NA for i in u_bin], index=adata.obs_names, dtype='string')
        instability_palette = {name: color for name, _, _, color in BINS}
        draw_spatial(ax_bottom, ctx, instability_labels, instability_palette)
        spot_counts[method_i] = bin_counts(u)
        if results_key not in adata.uns:
            blank_panel(ax_top, 'No runs')
            continue
        method_results = list(adata.uns[results_key])
        if not method_results:
            blank_panel(ax_top, 'No runs')
            continue
        # Example run = run with the fewest unique clusters.
        example_i = min(range(len(method_results)), key=lambda i: len(np.unique(np.asarray(method_results[i]).astype(str))))
        example_run = np.asarray(method_results[example_i]).astype(str)
        if has_reference:
            pred_palette = aligned_prediction_palette(adata.obs['annotation'], example_run, gt_palette)
        else:
            cats = list(pd.unique(example_run))
            pred_palette = {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(cats)}
        draw_spatial(ax_top, ctx, example_run, pred_palette)
        cluster_means = []
        for cluster in np.unique(example_run):
            mask = example_run == cluster
            if mask.any():
                cluster_means.append(np.nanmean(u[mask]))
        cluster_counts[method_i] = bin_counts(np.asarray(cluster_means))
    top_axes[0].set_ylabel('Example Run', fontsize=11, fontweight='bold', labelpad=3)
    bottom_axes[0].set_ylabel('Instability Map', fontsize=11, fontweight='bold', labelpad=3)
    top_axes[0].text(-0.22, 1.08, 'B', transform=top_axes[0].transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    bottom_axes[0].text(-0.22, 1.08, 'C', transform=bottom_axes[0].transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    d_col = n_methods + 3
    ax_spots = fig.add_subplot(gs[0, d_col])
    ax_clusters = fig.add_subplot(gs[1, d_col])
    pos = ax_clusters.get_position()
    ax_clusters.set_position([pos.x0, 0.2, pos.width, pos.height])
    x = np.arange(n_methods)
    totals = spot_counts.sum(axis=1)
    percent = np.divide(spot_counts, totals[:, None], out=np.zeros_like(spot_counts, dtype=float), where=totals[:, None] > 0) * 100
    bottoms = np.zeros(n_methods)
    for bin_i, color in enumerate(BIN_COLORS):
        ax_spots.bar(x, percent[:, bin_i], bottom=bottoms, width=0.72, color=color, edgecolor='white', linewidth=0.4)
        bottoms += percent[:, bin_i]
    ax_spots.set_ylim(0, 100)
    ax_spots.set_yticks(np.arange(0, 101, 20))
    ax_spots.set_ylabel('Percent spots', fontsize=10, fontweight='bold')
    ax_spots.set_xticks([])
    bottoms = np.zeros(n_methods)
    for bin_i, color in enumerate(BIN_COLORS):
        ax_clusters.bar(x, cluster_counts[:, bin_i], bottom=bottoms, width=0.72, color=color, edgecolor='white', linewidth=0.4)
        bottoms += cluster_counts[:, bin_i]
    ax_clusters.set_ylabel('Number of clusters', fontsize=10, fontweight='bold')
    ax_clusters.set_xticks([])
    ax_clusters.set_xlabel('Clustering Methods\n(same order as maps)', fontsize=10, fontweight='bold', labelpad=3)
    ax_clusters.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for ax in (ax_spots, ax_clusters):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7, width=0.7, length=2)
    ax_spots.text(-0.35, 1.08, 'D', transform=ax_spots.transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    fig.canvas.draw()
    legend_left = top_axes[0].get_position().x0
    legend_right = top_axes[-1].get_position().x1
    draw_instability_legend(fig, legend_left, legend_right)
    out_base = OUT_DIR / f'{dataset}{suffix}_methods_uncertainty'
    fig.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{out_base}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
# -----------------------------------------------------------------------------
# Generate all Figure 2 panels
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    for dataset, sample in make_runs():
        plot_one(dataset, sample)
        #break
# -----------------------------------------------------------------------------
# Caption statistics: ARI and mean instability
# This intentionally remains top-level to preserve the original execution behavior.
# -----------------------------------------------------------------------------

from sklearn.metrics import adjusted_rand_score

def all_samples():
    for dataset in DATASETS:
        if dataset == 'dlpfc':
            for sample in DLPFC_SAMPLES:
                yield (dataset, sample)
        else:
            yield (dataset, None)


for dataset, sample in all_samples():
    suffix = f'_{sample}' if sample else ''
    path = DATA_PATH / f'{dataset}{suffix}.h5ad'
    adata = ad.read_h5ad(path)
    print(f"\n{'=' * 70}")
    print(f'{dataset}{suffix}')
    print(f"{'=' * 70}")
    if 'annotation' not in adata.obs:
        print('No reference annotation.')
        continue
    ref = adata.obs['annotation']
    valid = ref.notna().to_numpy()
    ref_valid = ref[valid].astype(str).to_numpy()
    print(f"{'Method':<18} {'ARI':>10} {'Mean instability':>18}")
    for method, display in METHODS:
        uncertainty_key = f'{method}_uncertainty'
        results_key = f'{method}_results'
        if uncertainty_key not in adata.obs:
            continue
        uncertainty = adata.obs[uncertainty_key].to_numpy(dtype=float)
        mean_instability = np.nanmean(uncertainty)
        ari = np.nan
        if results_key in adata.uns:
            runs = list(adata.uns[results_key])
            if runs:
                example_i = min(range(len(runs)), key=lambda i: len(np.unique(np.asarray(runs[i]).astype(str))))
                example_run = np.asarray(runs[example_i]).astype(str)
                ari = adjusted_rand_score(ref_valid, example_run[valid])
        print(f'{display:<18} {ari:>10.4f} {mean_instability:>18.4f}')
    del adata
