#!/usr/bin/env python3

import gc
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
import geopandas as gpd
import scanpy as sc
import spatialdata as sd
import spatialdata_plot  # registers sdata.pl
from anndata import AnnData
from scipy import stats
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, box
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from matplotlib.lines import Line2D
# Figure-wide styling and paths.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
DATA_DIR = '/home/wuw15/data_dir/ST_datasets/AAA_with_uncert'
FIG_DIR = '/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig3'
METRICS_CSV = os.path.join(FIG_DIR, 'all_sample_method_metrics_by_number_of_runs.csv')
OUT_STEM = os.path.join(FIG_DIR, 'figure_3')
DPI = 300
NRUNS = 50
COLUMN_WSPACE = 0.06
ROW_HSPACE = 0.02
CROP_PADDING = 0.035
METHODS = ['leiden', 'louvain', 'bayesspace', 'graphst', 'stagate', 'spicemix', 'sedr', 'sedr_mclust']
METHOD_DISPLAY = {'leiden': 'Leiden', 'louvain': 'Louvain', 'bayesspace': 'Bayes\nSpace', 'graphst': 'GraphST', 'stagate': 'STAGATE', 'spicemix': 'SpiceMix', 'sedr': 'SEDR', 'sedr_mclust': 'SEDR\n(mclust)'}
DATASET_DISPLAY = {'dlpfc': 'DLPFC', 'mouse_brain': 'MB', 'mouse_brain_cerebellum': 'MBC', 'human_breast_cancer': 'HBC', 'Visium_HD_Human_Colon_Cancer_cropped_square': 'CRC', 'ov_ffpe': 'OV', 'coad_ffpe': 'COAD'}
SAMPLES = [('dlpfc', sample) for sample in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']] + [('mouse_brain', None), ('mouse_brain_cerebellum', None), ('human_breast_cancer', None), ('Visium_HD_Human_Colon_Cancer_cropped_square', None), ('ov_ffpe', None), ('coad_ffpe', None)]
LABELS = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,1.0]']
BINS = np.r_[np.arange(0.0, 1.0, 0.1), np.nextafter(1.0, np.inf)]
PALETTE = dict(zip(LABELS, ['#08306b', '#4292c6', '#9ecae1', '#deebf7', '#fee391', '#fec44f', '#fdae6b', '#fd8d3c', '#fb6a4a', '#cb181d']))
TABLEAU_20 = plt.cm.tab20.colors


# -----------------------------------------------------------------------------
# Spatial data helpers
# -----------------------------------------------------------------------------
def h5ad_path(dataset, sample):
    suffix = f'_{sample}' if sample is not None else ''
    return f'{DATA_DIR}/{dataset}{suffix}.h5ad'

def sample_name(dataset, sample):
    name = DATASET_DISPLAY.get(dataset, dataset)
    if sample is not None:
        return f'{name}\n{sample}'
    return name

def image_and_coordinates(adata):
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    spatial = adata.uns.get('spatial', {})
    if not spatial:
        return (None, coords)
    library = spatial[next(iter(spatial))]
    images = library.get('images', {})
    scales = library.get('scalefactors', {})
    for key in ('hires', 'lowres'):
        if key in images:
            scale = float(scales.get(f'tissue_{key}_scalef', 1.0))
            return (np.asarray(images[key]), coords * scale)
    return (None, coords)

def grayscale(image):
    if image.ndim == 2:
        return image
    rgb = image[..., :3].astype(float)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(gray, 0, np.iinfo(image.dtype).max)
    return gray.astype(image.dtype)

def nearest_spacing(coords):
    unique = np.unique(np.asarray(coords, dtype=float), axis=0)
    if len(unique) < 2:
        return 1.0
    distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size:
        return float(np.median(distances))
    return 1.0

def make_shapes(coords, index, dataset):
    spacing = nearest_spacing(coords)
    # Match each platform's spot geometry.
    if dataset == 'mouse_brain_cerebellum':
        radius = 0.48 * spacing
        shapes = gpd.GeoDataFrame(geometry=[Point(x, y).buffer(radius) for x, y in coords], index=index)
    elif 'Visium_HD' in dataset or dataset in {'coad_ffpe', 'ov_ffpe'}:
        half = spacing / 2.0
        shapes = gpd.GeoDataFrame(geometry=[box(x - half, y - half, x + half, y + half) for x, y in coords], index=index)
    elif dataset == 'mouse_brain':
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        shapes = gpd.GeoDataFrame(geometry=[Polygon([(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]) for x, y in coords], index=index)
    else:
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        shapes = gpd.GeoDataFrame(geometry=[Polygon([(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]) for x, y in coords], index=index)
    return ShapesModel.parse(shapes)

def build_spatialdata(adata, dataset):
    image, coords = image_and_coordinates(adata)
    index = pd.Index(adata.obs_names.astype(str), name='instance_id')
    shapes = make_shapes(coords, index, dataset)
    obs = pd.DataFrame(index=index)
    obs['region'] = pd.Categorical(['spots'] * adata.n_obs)
    obs['instance_id'] = index
    available_methods = set()
    # Figure 3 uses final per-method uncertainty stored in adata.obs.
    for method in set(METHODS):
        key = f'{method}_uncertainty'
        if key not in adata.obs:
            continue
        obs[f'{method}_bin'] = pd.cut(adata.obs[key].to_numpy(dtype=float), bins=BINS, labels=LABELS, include_lowest=True, right=False, ordered=True)
        available_methods.add(method)
    table = TableModel.parse(AnnData(obs=obs), region='spots', region_key='region', instance_key='instance_id')
    images = {}
    if image is not None:
        images['histology'] = Image2DModel.parse(grayscale(image)[None, ...], dims=('c', 'y', 'x'), c_coords=['gray'])
    sdata = sd.SpatialData(images=images, shapes={'spots': shapes}, tables={'table': table})
    return (sdata, sdata.shapes['spots'].total_bounds, available_methods)

def square_limits(bounds):
    min_x, min_y, max_x, max_y = bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    span = max(max_x - min_x, max_y - min_y)
    span *= 1 + 2 * CROP_PADDING
    half = span / 2
    return (center_x - half, center_x + half, center_y - half, center_y + half)

def blank_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_subplot(ax, sdata, bounds, method):
    plot = sdata
    if 'histology' in sdata.images:
        plot = plot.pl.render_images(element='histology', cmap='gray', alpha=0.45)
    plot.pl.render_shapes(element='spots', color=f'{method}_bin', palette=PALETTE, fill_alpha=1.0, outline_width=0, table_name='table', method='matplotlib').pl.show(ax=ax, title='', frameon=True, legend_loc=None, colorbar=False, show=False)

    # Rasterize dense spatial shapes in PDF, while keeping
    # text, axes, legends, etc. as vector.
    for collection in ax.collections:
        collection.set_rasterized(True)

    x0, x1, y0, y1 = square_limits(bounds)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('C')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.35)


# -----------------------------------------------------------------------------
# Panel A: spatial instability maps
# -----------------------------------------------------------------------------
def add_spatial_headers(axes):
    for j, (dataset, sample) in enumerate(SAMPLES):
        y = 1.3
        axes[0, j].text(0.5, y, sample_name(dataset, sample), ha='center', va='center', fontsize=20, fontweight='bold', linespacing=0.95, transform=axes[0, j].transAxes, clip_on=False)
    for i, method in enumerate(METHODS):
        ax = axes[i, 0]
        if method in {'bayesspace', 'sedr'}:
            background = FancyBboxPatch((-1.15, 0.18), 1.1, 0.68, boxstyle='round,pad=0.02', facecolor='#a5cbe7', edgecolor='black', linewidth=1.5, linestyle='--', transform=ax.transAxes, clip_on=False, zorder=1)
            ax.add_patch(background)
        ax.text(-0.045, 0.5, METHOD_DISPLAY[method], ha='right', va='center', fontsize=22, fontweight='bold', transform=ax.transAxes, clip_on=False, zorder=2)

def add_instability_legend(fig, axes):
    handles = [Patch(facecolor=PALETTE[label], edgecolor='none', label=label) for label in LABELS]
    # Matplotlib fills multi-column legends column-first; this yields two rows.
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    handles = [handles[i] for i in order]
    labels = [LABELS[i] for i in order]
    left_pos = axes[-1, 12].get_position()
    right_pos = axes[-1, -1].get_position()
    legend_x = (left_pos.x0 + right_pos.x1) / 2
    legend_y = (right_pos.y0 + right_pos.y1) / 2
    legend = fig.legend(handles=handles, labels=labels, title='Instability Score Bins', loc='center', bbox_to_anchor=(legend_x, legend_y), bbox_transform=fig.transFigure, ncol=5, frameon=False, fontsize=14, title_fontsize=16, handlelength=1.3, handleheight=0.8, handletextpad=0.5, columnspacing=1.5, labelspacing=0.9, borderaxespad=0.0)
    legend.get_title().set_fontweight('bold')
    try:
        legend._legend_box.align = 'left'
    except AttributeError:
        pass

def plot_spatial_panel(fig, axes):
    for i, (dataset, sample) in enumerate(SAMPLES):
        path = h5ad_path(dataset, sample)
        print(f"[spatial {i + 1:02d}/{len(SAMPLES)}] {sample_name(dataset, sample).replace(chr(10), ' ')}")
        if not os.path.exists(path):
            for j in range(len(METHODS)):
                blank_axis(axes[j, i])
            continue
        adata = sc.read_h5ad(path)
        sdata, bounds, available_methods = build_spatialdata(adata, dataset)
        for j, method in enumerate(METHODS):
            ax = axes[j, i]
            # SEDR-mclust is only available for DLPFC.
            unavailable = method not in available_methods or (method == 'sedr_mclust' and dataset != 'dlpfc')
            if unavailable:
                blank_axis(ax)
                continue
            plot_subplot(ax, sdata, bounds, method)
        del adata, sdata
        gc.collect()
    add_spatial_headers(axes)
    add_instability_legend(fig, axes)


# -----------------------------------------------------------------------------
# Panel B: sample instability/entropy vs. ARI
# -----------------------------------------------------------------------------
def plot_scatter_panel(fig, axes):
    df = pd.read_csv(METRICS_CSV)
    df = df[df['iterations'] == NRUNS]
    mthds = ['leiden', 'louvain', 'bayesspace', 'bayesspace', 'graphst', 'stagate', 'spicemix', 'sedr', 'sedr_mclust']
    mnams = ['Leiden', 'Louvain', 'BayesSpace', 'BayesSpace', 'GraphST', 'STAGATE', 'SpiceMix', 'SEDR', 'SEDR (mclust)']
    color_index = {'mouse_brain_cerebellum': 0, 'mouse_brain': 8, 'Visium_HD_Human_Colon_Cancer_cropped_square': 2, 'human_breast_cancer': 12, 'coad_ffpe': 16, 'ov_ffpe': 18, 'dlpfc': 14}
    for axi, ax in enumerate(axes):
        method = mthds[axi]
        # The second BayesSpace panel uses entropy instead of instability.
        if axi == 3:
            xs = df['bayesspace_mean_entropy_all_runs'].values
            ax.set_xlabel('Sample Entropy', fontsize=16)
        else:
            xs = df[method + '_mean_instability'].values
            ax.set_xlabel('Sample Instability', fontsize=16)
        ys = df[method + '_mean_ari_all_runs'].values
        ds = df['dataset'].values
        ii = np.flatnonzero(~np.isnan(xs) & ~np.isnan(ys))
        xx = xs[ii]
        yy = ys[ii]
        datasets = ds[ii]
        ax.set_title(mnams[axi], fontsize=20, fontweight='bold')
        if axi == 0:
            ax.set_ylabel('ARI w.r.t. Ref.', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.0)
        r, p = stats.pearsonr(xx, yy)
        r_sq = r ** 2
        print(f'{method}: r={r:.4f}, r2={r_sq:.4f}, p={p:.4g}')
        # Preserve the original behavior: no regression line/stat box for SEDR-mclust.
        if method != 'sedr_mclust':
            slope, intercept = np.polyfit(xx, yy, 1)
            fit_x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, color='black', linestyle='--', linewidth=2)
            info = f'r = {r:.2f}\nr² = {r_sq:.2f}\np = {p:.3g}'
            ax.text(0.5, -0.4, info, transform=ax.transAxes, ha='center', va='top', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
        for x, y, dataset in zip(xx, yy, datasets):
            col = color_index.get(dataset, 14)
            alpha = 0.3 if dataset == 'dlpfc' else 1.0
            ax.scatter(x, y, color=TABLEAU_20[col], s=140, alpha=alpha)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.canvas.draw()
    add_scatter_boxes(fig, axes)


# Grouping boxes and shared sample legend for Panel B.
def add_group_box(fig, first_ax, last_ax):
    p1 = first_ax.get_position()
    p2 = last_ax.get_position()
    height = p1.height
    x0 = p1.x0 - 0.006
    x1 = p2.x1 + 0.006
    y0 = p1.y0 - 0.8 * height
    y1 = p1.y1 + 0.15 * height
    patch = FancyBboxPatch((x0, y0), width=x1 - x0, height=y1 - y0, boxstyle='round,pad=0.005', edgecolor='black', facecolor='none', linestyle='--', linewidth=1.5, transform=fig.transFigure, clip_on=False, zorder=10)
    fig.add_artist(patch)

def add_scatter_boxes(fig, axes):
    add_group_box(fig, axes[2], axes[3])
    add_group_box(fig, axes[7], axes[7])

def add_sample_legend(fig, scatter_axes):
    labels = ['DLPFC', 'MB', 'MBC', 'HBC', 'CRC', 'OV', 'COAD']
    colors = [TABLEAU_20[14], TABLEAU_20[8], TABLEAU_20[0], TABLEAU_20[12], TABLEAU_20[2], TABLEAU_20[18], TABLEAU_20[16]]
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=12, markerfacecolor=color, markeredgecolor='none', label=label) for label, color in zip(labels, colors)]
    p0 = scatter_axes[0].get_position()
    p1 = scatter_axes[-1].get_position()
    x_center = (p0.x0 + p1.x1) / 2
    y = p0.y0 - 0.14
    legend = fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(x_center, y), bbox_transform=fig.transFigure, ncol=7, frameon=False, fontsize=16, handletextpad=0.4, columnspacing=1.7, borderaxespad=0)
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bbox.x0 - 0.015, y, 'Sample', fontsize=16, fontweight='bold', ha='right', va='center')

def add_panel_letters(fig, spatial_axes, scatter_axes):
    fig.canvas.draw()
    x = 0.03
    y_A = spatial_axes[0, 0].get_position().y1 + 0.035
    y_B = scatter_axes[0].get_position().y1 + 0.035
    fig.text(x, y_A, 'A', fontsize=28, fontweight='bold', ha='left', va='top')
    fig.text(x, y_B, 'B', fontsize=28, fontweight='bold', ha='left', va='top')


# -----------------------------------------------------------------------------
# Build and save the integrated Figure 3
# -----------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig = plt.figure(figsize=(22, 13))
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 4], hspace=0.1, left=0.08, right=0.995, top=0.96, bottom=0.08)
    top_grid = outer[0].subgridspec(nrows=len(METHODS), ncols=len(SAMPLES), wspace=COLUMN_WSPACE, hspace=ROW_HSPACE)
    spatial_axes = np.empty((len(METHODS), len(SAMPLES)), dtype=object)
    for i in range(len(METHODS)):
        for j in range(len(SAMPLES)):
            spatial_axes[i, j] = fig.add_subplot(top_grid[i, j])
    bottom_grid = outer[1].subgridspec(nrows=1, ncols=9, wspace=0.28)
    scatter_axes = np.array([fig.add_subplot(bottom_grid[0, j]) for j in range(9)])
    plot_spatial_panel(fig, spatial_axes)
    plot_scatter_panel(fig, scatter_axes)
    add_sample_legend(fig, scatter_axes)
    add_panel_letters(fig, spatial_axes, scatter_axes)
    # Save raster and editable-vector versions.
    png_path = OUT_STEM + '.png'
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    print(f'[saved] {png_path}')
    pdf_path = OUT_STEM + '.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    print(f'[saved] {pdf_path}')
    plt.close(fig)
if __name__ == '__main__':
    main()
