"""
Figure 5, with panels A-E integrated using Figure 3's general styling.

Important semantic difference retained intentionally:
- Figure 3 reads final uncertainty from adata.obs["<method>_uncertainty"].
- Figure 5A reads run-count-specific spot_uncertainty pickle files.

Panel A contains the spatial instability maps, panel B contains the
category-transition heatmaps, panel C contains the direct k-run-vs-50-run
hexbin comparisons, panel D contains sample instability vs ARI, and panel E
contains cluster instability vs clusterwise Jaccard with respect to the reference.
"""
import gc
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import spatialdata_plot  # registers sdata.pl
from anndata import AnnData
from scipy import stats
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, box
from spatialdata.models import Image2DModel, ShapesModel, TableModel
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Final Figure 5 canvas.
FIGSIZE = (22, 13)
# Set to (dataset, sample) for fast single-sample development.
PLOT_ONLY_SAMPLE = None
FIG5_LEFT = 0.08
FIG5_RIGHT = 0.995
FIG5_TOP = 0.96
FIG5_BOTTOM = 0.08
FIG5_HEIGHT_RATIOS = [1.7, 0.13, 1.2, 0.25, 0.75]
FIG5_OUTER_WSPACE = 0.16
FIG5_BC_WSPACE = 0.22
DATA_DIR = Path('/home/wuw15/data_dir/ST_datasets/AAA_with_uncert')
INSTABILITY_ROOT = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/1-nested-non-sampling')
OUT_DIR = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig5')
FIG3_DIR = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig3')
METRICS_CSV = FIG3_DIR / 'all_sample_method_metrics_by_number_of_runs.csv'
CLUSTERWISE_METRICS_CSV = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig5/clusterwise_metrics_by_number_of_runs_new.csv')
RUN_COUNTS = [5, 10, 20, 40]
D_RUN_COUNTS = RUN_COUNTS.copy()
E_RUN_COUNTS = RUN_COUNTS.copy()
# Panels B/C compare each requested run count with the 50-run reference.
REFERENCE_RUNS = 50
HEATMAP_RUN_COUNTS = RUN_COUNTS.copy()
HEATMAP_PLOT_ONLY_SAMPLE = PLOT_ONLY_SAMPLE
HEXBIN_PLOT_ONLY_SAMPLE = PLOT_ONLY_SAMPLE
HEXBIN_RUN_COUNTS = RUN_COUNTS.copy()
MAX_HEXBIN_SPOTS_PER_SAMPLE = 10000
HEXBIN_GRIDSIZE = 45
# Panel B uses five broad transition bins; panel A uses ten 0.1-wide bins.
TRANSITION_BIN_LABELS = ['[0.0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,1.0]']
TRANSITION_BINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, np.nextafter(1.0, np.inf)])
HEATMAP_ANNOTATION_THRESHOLD = 0.01
HEATMAP_ANNOTATION_FONTSIZE = 7
DPI = 300
TABLEAU_20 = plt.cm.tab20.colors
D_DATASET_COLOR_INDEX = {'mouse_brain_cerebellum': 0, 'mouse_brain': 8, 'Visium_HD_Human_Colon_Cancer_cropped_square': 2, 'human_breast_cancer': 12, 'coad_ffpe': 16, 'ov_ffpe': 18, 'dlpfc': 14}
D_SCATTER_SIZE = 140
E_SCATTER_SIZE = 20
E_NON_DLPFC_ALPHA = 0.8
E_DLPFC_ALPHA = 0.3
E_DEFAULT_XMAX = 0.7
E_XMAX_STEP = 0.1
E_XCOL = 'clusterwise mean instability'
E_METRIC = 'clusterwise jaccard'
E_SINGLETON_SAMPLE_TAGS = {'mouse_brain_cerebellum', 'mouse_brain', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'human_breast_cancer', 'coad_ffpe', 'ov_ffpe'}
COLUMN_WSPACE = 0.06
ROW_HSPACE = 0.01
CROP_PADDING = 0.035
SAMPLE_FONTSIZE = 20
ROW_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 14
LEGEND_TITLE_FONTSIZE = 16
PANEL_LETTER_FONTSIZE = 28
METHODS = ['leiden', 'louvain', 'bayesspace', 'graphst', 'stagate', 'spicemix', 'sedr', 'sedr_mclust']
METHOD_DISPLAY = {'leiden': 'Leiden', 'louvain': 'Louvain', 'bayesspace': 'BayesSpace', 'graphst': 'GraphST', 'stagate': 'STAGATE', 'spicemix': 'SpiceMix', 'sedr': 'SEDR', 'sedr_mclust': 'SEDR (mclust)'}
DATASET_DISPLAY = {'dlpfc': 'DLPFC', 'mouse_brain': 'MB', 'mouse_brain_cerebellum': 'MBC', 'human_breast_cancer': 'HBC', 'Visium_HD_Human_Colon_Cancer_cropped_square': 'CRC', 'ov_ffpe': 'OV', 'coad_ffpe': 'COAD'}
SAMPLES = [('dlpfc', sample) for sample in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']] + [('mouse_brain', None), ('mouse_brain_cerebellum', None), ('human_breast_cancer', None), ('Visium_HD_Human_Colon_Cancer_cropped_square', None), ('ov_ffpe', None), ('coad_ffpe', None)]
LABELS = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,1.0]']
BINS = np.r_[np.arange(0.0, 1.0, 0.1), np.nextafter(1.0, np.inf)]
PALETTE = dict(zip(LABELS, ['#08306b', '#4292c6', '#9ecae1', '#deebf7', '#fee391', '#fec44f', '#fdae6b', '#fd8d3c', '#fb6a4a', '#cb181d']))

# ---- Paths and naming ----
def dataset_tag(dataset, sample):
    return f'{dataset}_{sample}' if sample is not None else dataset

def h5ad_path(dataset, sample):
    return DATA_DIR / f'{dataset_tag(dataset, sample)}.h5ad'

def instability_path(dataset, sample, method, k):
    tag = dataset_tag(dataset, sample)
    return INSTABILITY_ROOT / dataset / method / f'{tag}_{method}_uncertainty_{k}runs.pkl'

def sample_name(dataset, sample):
    name = DATASET_DISPLAY.get(dataset, dataset)
    if sample is not None:
        return f'{name}\n{sample}'
    return name

# ---- Spatial image/geometry helpers ----
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
    """Figure 3 version, including the missing-distance fallback."""
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

# ---- Figure 5A spatial data ----
def build_spatialdata(adata, dataset, sample, method):
    image, coords = image_and_coordinates(adata)
    index = pd.Index(adata.obs_names.astype(str), name='instance_id')
    shapes = make_shapes(coords, index, dataset)
    obs = pd.DataFrame(index=index)
    obs['region'] = pd.Categorical(['spots'] * adata.n_obs)
    obs['instance_id'] = index
    available_runs = set()
    # Figure 5A intentionally reads run-specific spot_uncertainty pickles,
    # unlike Figure 3, which reads final uncertainty from AnnData.obs.
    for k in RUN_COUNTS:
        path = instability_path(dataset, sample, method, k)
        if not path.exists():
            continue
        result = joblib.load(path)
        instability = np.asarray(result['spot_uncertainty'], dtype=float)
        if len(instability) != adata.n_obs:
            print(f'[length mismatch] {path}: {len(instability)} values for {adata.n_obs} spots')
            continue
        obs[f'instability_{k}runs'] = pd.cut(instability, bins=BINS, labels=LABELS, include_lowest=True, right=False, ordered=True)
        available_runs.add(k)
    table = TableModel.parse(AnnData(obs=obs), region='spots', region_key='region', instance_key='instance_id')
    images = {}
    if image is not None:
        images['histology'] = Image2DModel.parse(grayscale(image)[None, ...], dims=('c', 'y', 'x'), c_coords=['gray'])
    sdata = sd.SpatialData(images=images, shapes={'spots': shapes}, tables={'table': table})
    return (sdata, sdata.shapes['spots'].total_bounds, available_runs)

# ---- Shared plotting helpers ----
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

def plot_subplot(ax, sdata, bounds, k):
    plot = sdata
    if 'histology' in sdata.images:
        plot = plot.pl.render_images(element='histology', cmap='gray', alpha=0.45)
    plot.pl.render_shapes(element='spots', color=f'instability_{k}runs', palette=PALETTE, fill_alpha=1.0, outline_width=0, table_name='table', method='matplotlib').pl.show(ax=ax, title='', frameon=True, legend_loc=None, colorbar=False, show=False)

    # Rasterize dense spatial spot geometry in PDF.
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

# ---- Figure 5A ----
def add_spatial_headers(axes):
    """Use Figure 3's explicit axes-coordinate text, not set_title/pad."""
    for j, (dataset, sample) in enumerate(SAMPLES):
        axes[0, j].text(0.5, 1.3, sample_name(dataset, sample), ha='center', va='center', fontsize=SAMPLE_FONTSIZE, fontweight='bold', linespacing=0.95, transform=axes[0, j].transAxes, clip_on=False)
    for i, k in enumerate(RUN_COUNTS):
        axes[i, 0].text(-0.045, 0.5, f'{k}\nruns', ha='right', va='center', fontsize=ROW_LABEL_FONTSIZE, fontweight='bold', transform=axes[i, 0].transAxes, clip_on=False)

def add_instability_legend(fig, axes):
    """
    Draw the instability-bin key as one horizontal row below panel A.

    "Instability Score Bins" is inline *before* the first color handle rather
    than being a Matplotlib legend title above the handles. The title plus the
    legend entries are centered as one combined group beneath the spatial grid.
    """
    handles = [Patch(facecolor=PALETTE[label], edgecolor='none', label=label) for label in LABELS]
    fig.canvas.draw()
    left_pos = axes[-1, 0].get_position()
    right_pos = axes[-1, -1].get_position()
    group_center_x = (left_pos.x0 + right_pos.x1) / 2
    legend_y = left_pos.y0 - 0.032
    legend = fig.legend(handles=handles, labels=LABELS, loc='center', bbox_to_anchor=(group_center_x, legend_y), bbox_transform=fig.transFigure, ncol=len(LABELS), frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=1.3, handleheight=0.8, handletextpad=0.5, columnspacing=1.5, labelspacing=0.9, borderaxespad=0.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    legend_width = legend_bbox.width
    temp_title = fig.text(0.0, 0.0, 'Instability Score Bins', fontsize=LEGEND_TITLE_FONTSIZE, fontweight='bold', ha='left', va='center')
    fig.canvas.draw()
    title_bbox = temp_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    title_width = title_bbox.width
    temp_title.remove()
    title_legend_gap = 0.01
    total_width = title_width + title_legend_gap + legend_width
    group_left = group_center_x - total_width / 2
    legend_center_x = group_left + title_width + title_legend_gap + legend_width / 2
    legend.set_bbox_to_anchor((legend_center_x, legend_y), transform=fig.transFigure)
    fig.text(group_left, legend_y, 'Instability Score Bins', fontsize=LEGEND_TITLE_FONTSIZE, fontweight='bold', ha='left', va='center')
    return legend

def plot_fig5a(fig, parent_spec, method):
    """
    Draw Figure 5A inside parent_spec and return its axes and legend.

    parent_spec should be a SubplotSpec from the *final* Figure 5 layout.
    The final Figure 5 canvas should be created with figsize=(22, 14), as in
    Figure 3. This keeps Figure 5A independent of whatever panels are later
    added below it.
    """
    top_grid = parent_spec.subgridspec(nrows=len(RUN_COUNTS), ncols=len(SAMPLES), wspace=COLUMN_WSPACE, hspace=ROW_HSPACE)
    axes = np.empty((len(RUN_COUNTS), len(SAMPLES)), dtype=object)
    for i in range(len(RUN_COUNTS)):
        for j in range(len(SAMPLES)):
            axes[i, j] = fig.add_subplot(top_grid[i, j])
    for j, (dataset, sample) in enumerate(SAMPLES):
        if PLOT_ONLY_SAMPLE is not None and (dataset, sample) != PLOT_ONLY_SAMPLE:
            for i in range(len(RUN_COUNTS)):
                blank_axis(axes[i, j])
            continue
        print(f"[{METHOD_DISPLAY[method]}] {sample_name(dataset, sample).replace(chr(10), ' ')}")
        path = h5ad_path(dataset, sample)
        unavailable = not path.exists() or (method == 'sedr_mclust' and dataset != 'dlpfc')
        if unavailable:
            for i in range(len(RUN_COUNTS)):
                blank_axis(axes[i, j])
            continue
        adata = sc.read_h5ad(path)
        sdata, bounds, available_runs = build_spatialdata(adata, dataset, sample, method)
        for i, k in enumerate(RUN_COUNTS):
            ax = axes[i, j]
            if k not in available_runs:
                blank_axis(ax)
                continue
            plot_subplot(ax, sdata, bounds, k)
        del adata, sdata
        gc.collect()
    add_spatial_headers(axes)
    legend = add_instability_legend(fig, axes)
    return (axes, legend)

# ---- Figure 5B: category-transition heatmaps ----
def combination_available(dataset, method):
    """Return whether this dataset/method combination is expected to exist."""
    if method == 'sedr_mclust' and dataset != 'dlpfc':
        return False
    if method == 'stagate' and dataset == 'mouse_brain_cerebellum':
        return False
    return True

def transition_category(values):
    """Map instability values into the five Figure 5B transition bins."""
    return np.clip(np.digitize(values, TRANSITION_BINS[1:-1], right=False), 0, len(TRANSITION_BIN_LABELS) - 1)

def analyze_transition_pair(dataset, sample, method):
    """
    Build 50-run -> current-run transition counts for one sample/method pair.

    Only 5, 10, 20, and 40 runs are plotted, but the 50-run pickle is loaded
    as the reference. Missing required files cause this pair to be skipped with
    an explicit message rather than crashing the entire figure build.
    """
    required_runs = [REFERENCE_RUNS] + HEATMAP_RUN_COUNTS
    paths = {runs: instability_path(dataset, sample, method, runs) for runs in required_runs}
    missing = [runs for runs, path in paths.items() if not path.exists()]
    if missing:
        print(f'[transition missing] {dataset_tag(dataset, sample)}, {method}: missing runs {missing}')
        return []
    scores = {runs: np.asarray(joblib.load(paths[runs])['spot_uncertainty'], dtype=float) for runs in required_runs}
    lengths = {runs: len(values) for runs, values in scores.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f'Spot-count mismatch for transition heatmap: {dataset_tag(dataset, sample)}, {method}, {lengths}')
    rows = []
    n_categories = len(TRANSITION_BIN_LABELS)
    reference_values = scores[REFERENCE_RUNS]
    for runs in HEATMAP_RUN_COUNTS:
        current_values = scores[runs]
        valid = np.isfinite(reference_values) & np.isfinite(current_values)
        reference_category = transition_category(reference_values[valid])
        current_category = transition_category(current_values[valid])
        counts = np.zeros((n_categories, n_categories), dtype=int)
        np.add.at(counts, (reference_category, current_category), 1)
        for reference_code in range(n_categories):
            for current_code in range(n_categories):
                rows.append({'dataset': dataset, 'sample': sample, 'method': method, 'runs': runs, 'reference_category': reference_code, 'current_category': current_code, 'count': counts[reference_code, current_code]})
    return rows

def build_transition_dataframe(selected_methods):
    """
    Build only the transition data panel B actually needs.

    This is intentionally more efficient than the old standalone script, which
    calculated every method and only filtered to HEATMAP_METHODS afterward.
    """
    selected_methods = list(dict.fromkeys(selected_methods))
    if not selected_methods:
        raise ValueError('HEATMAP_METHODS cannot be empty.')
    unknown = sorted(set(selected_methods) - set(METHODS))
    if unknown:
        raise ValueError(f'Unknown HEATMAP_METHODS: {unknown}')
    if HEATMAP_PLOT_ONLY_SAMPLE is None:
        samples_to_use = SAMPLES
    else:
        if HEATMAP_PLOT_ONLY_SAMPLE not in SAMPLES:
            raise ValueError(f'HEATMAP_PLOT_ONLY_SAMPLE is not in SAMPLES: {HEATMAP_PLOT_ONLY_SAMPLE}')
        samples_to_use = [HEATMAP_PLOT_ONLY_SAMPLE]
    rows = []
    for dataset, sample in samples_to_use:
        for method in selected_methods:
            if not combination_available(dataset, method):
                continue
            print(f'[transition] {dataset_tag(dataset, sample)}, {method}')
            rows.extend(analyze_transition_pair(dataset, sample, method))
    if not rows:
        raise ValueError('No transition data were available for Figure 5B.')
    return pd.DataFrame(rows)

def match_colorbar_height(fig, cbar_ax, plot_axes):
    """
    Make the colorbar start and end at exactly the same vertical
    coordinates as the square plotting axes.
    """
    fig.canvas.draw()
    plot_pos = plot_axes[0].get_position()
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([cbar_pos.x0, plot_pos.y0, cbar_pos.width, plot_pos.height])

def plot_fig5b(fig, parent_spec, transition_df, selected_methods):
    """
    Draw Figure 5B: four 50-vs-run transition heatmaps plus one colorbar.

    The colorbar gets a dedicated axis; no heatmap is drawn underneath it.
    This avoids the residual blue artifact produced by hiding a previously
    populated ninth subplot and then asking pyplot to place a colorbar there.
    """
    heatmap_df = transition_df[transition_df['method'].isin(selected_methods)]
    if heatmap_df.empty:
        raise ValueError('No transition data for selected Figure 5B methods.')
    grid = parent_spec.subgridspec(nrows=1, ncols=len(HEATMAP_RUN_COUNTS) + 1, width_ratios=[1.0] * len(HEATMAP_RUN_COUNTS) + [0.09], wspace=0.2)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(HEATMAP_RUN_COUNTS))])
    cbar_ax = fig.add_subplot(grid[0, -1])
    image = None
    n_categories = len(TRANSITION_BIN_LABELS)
    for axi, (ax, runs) in enumerate(zip(axes, HEATMAP_RUN_COUNTS)):
        subset = heatmap_df[heatmap_df['runs'] == runs]
        counts = subset.groupby(['reference_category', 'current_category'])['count'].sum().unstack(fill_value=0).reindex(index=range(n_categories), columns=range(n_categories), fill_value=0).astype(float)
        matrix = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        image = ax.imshow(matrix.to_numpy(), cmap='Blues', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
        for i in range(n_categories):
            for j in range(n_categories):
                value = matrix.iloc[i, j]
                if np.isfinite(value) and value >= HEATMAP_ANNOTATION_THRESHOLD:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=HEATMAP_ANNOTATION_FONTSIZE, color='white' if value >= 0.5 else 'black')
        ax.set_title(f'{REFERENCE_RUNS} vs. {runs}', fontsize=20, fontweight='bold', pad=8)
        ax.set_xticks(range(n_categories))
        ax.set_xticklabels(TRANSITION_BIN_LABELS, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(n_categories))
        if axi == 0:
            ax.set_yticklabels(TRANSITION_BIN_LABELS)
            ax.set_ylabel(f'Bins {REFERENCE_RUNS} runs', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label('Fraction of spots', fontsize=16, labelpad=10)
    match_colorbar_height(fig, cbar_ax, axes)
    return (axes, colorbar)

# ---- Figure 5C: k-run vs 50-run instability ----
def build_hexbin_dataframe(method):
    """
    Load only the data Figure 5C actually needs.

    - one selected Figure 5 method only
    - only 5, 10, 20, 40, and 50 runs
    - either one development sample or all SAMPLES
    - sample at most MAX_HEXBIN_SPOTS_PER_SAMPLE spots per sample

    The same spot indices are retained across all four k-run comparisons so the
    four panels are directly comparable.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown FIG5_METHOD: {method}')
    if HEXBIN_PLOT_ONLY_SAMPLE is None:
        samples_to_use = SAMPLES
    else:
        if HEXBIN_PLOT_ONLY_SAMPLE not in SAMPLES:
            raise ValueError(f'HEXBIN_PLOT_ONLY_SAMPLE is not in SAMPLES: {HEXBIN_PLOT_ONLY_SAMPLE}')
        samples_to_use = [HEXBIN_PLOT_ONLY_SAMPLE]
    required_runs = [REFERENCE_RUNS] + HEXBIN_RUN_COUNTS
    rng = np.random.default_rng(42)
    frames = []
    for dataset, sample in samples_to_use:
        if not combination_available(dataset, method):
            continue
        tag = dataset_tag(dataset, sample)
        paths = {runs: instability_path(dataset, sample, method, runs) for runs in required_runs}
        missing = [runs for runs, path in paths.items() if not path.exists()]
        if missing:
            print(f'[hexbin missing] {tag}, {method}: missing runs {missing}')
            continue
        print(f'[hexbin] {tag}, {method}')
        scores = {runs: np.asarray(joblib.load(paths[runs])['spot_uncertainty'], dtype=float) for runs in required_runs}
        lengths = {runs: len(values) for runs, values in scores.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f'Spot-count mismatch for Figure 5C: {tag}, {method}, {lengths}')
        valid = np.ones(next(iter(lengths.values())), dtype=bool)
        for runs in required_runs:
            valid &= np.isfinite(scores[runs])
        keep = np.flatnonzero(valid)
        if keep.size == 0:
            print(f'[hexbin empty] {tag}, {method}: no finite spots')
            continue
        if keep.size > MAX_HEXBIN_SPOTS_PER_SAMPLE:
            keep = rng.choice(keep, size=MAX_HEXBIN_SPOTS_PER_SAMPLE, replace=False)
        frame = pd.DataFrame({'dataset-sample': np.repeat(tag, len(keep))})
        for runs in required_runs:
            frame[f'instability_{runs}runs'] = scores[runs][keep]
        frames.append(frame)
    if not frames:
        raise ValueError('No Figure 5C data were available.')
    return pd.concat(frames, ignore_index=True)

def plot_fig5c(fig, parent_spec, spot_df, method):
    """
    Draw Figure 5C: four direct instability comparisons plus one colorbar.

    x = instability score for k runs
    y = instability score for 50 runs

    All four hexbin artists are created with raw bin counts first. Their global
    maximum count is then used to construct ONE shared LogNorm, and that exact
    same norm is assigned to every panel. Therefore a given color means the same
    spot density in all four comparisons.
    """
    grid = parent_spec.subgridspec(nrows=1, ncols=len(HEXBIN_RUN_COUNTS) + 1, width_ratios=[1.0] * len(HEXBIN_RUN_COUNTS) + [0.09], wspace=0.2)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(HEXBIN_RUN_COUNTS))])
    cbar_ax = fig.add_subplot(grid[0, -1])
    images = []
    for axi, (ax, runs) in enumerate(zip(axes, HEXBIN_RUN_COUNTS)):
        x = spot_df[f'instability_{runs}runs'].to_numpy(dtype=float)
        y = spot_df[f'instability_{REFERENCE_RUNS}runs'].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        #image = ax.hexbin(x[valid], y[valid], gridsize=HEXBIN_GRIDSIZE, extent=(0, 1, 0, 1), mincnt=1, cmap='viridis')
        image = ax.hexbin(
                    x[valid],
                    y[valid],
                    gridsize=HEXBIN_GRIDSIZE,
                    extent=(0, 1, 0, 1),
                    mincnt=1,
                    cmap='viridis',
                    edgecolors='white',
                    linewidths=0.1,
                )
        image.set_rasterized(True)
        images.append(image)
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{REFERENCE_RUNS} vs. {runs}', fontsize=20, fontweight='bold', pad=8)
        ax.set_xlabel(f'Score {runs} runs', fontsize=16)
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'], rotation=45, ha='right')
        ax.set_yticks(ticks)
        if axi == 0:
            ax.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'])
            ax.set_ylabel(f'Score {REFERENCE_RUNS} runs', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
    global_vmax = max((float(np.max(image.get_array())) for image in images if image.get_array().size))
    global_vmax = max(2.0, global_vmax)
    shared_norm = LogNorm(vmin=1.0, vmax=global_vmax)
    for image in images:
        image.set_norm(shared_norm)
    colorbar = fig.colorbar(images[-1], cax=cbar_ax)
    max_exponent = int(np.floor(np.log10(global_vmax)))
    colorbar_ticks = 10.0 ** np.arange(max_exponent + 1)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([str(exponent) for exponent in range(max_exponent + 1)])
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label('Spot density (log10)', fontsize=16, labelpad=10)
    match_colorbar_height(fig, cbar_ax, axes)
    return (axes, colorbar, shared_norm)

# ---- Figure 5D: sample instability vs ARI ----
def plot_fig5d(fig, parent_spec, metrics_df, method):
    """
    Draw Figure 5D for one method and four run counts: 5, 10, 20, and 40.

    x = sample mean instability
    y = mean ARI with respect to the reference

    BayesSpace is treated exactly like the other methods here: instability only.
    No entropy branch is used in Figure 5D.
    """
    x_col = f'{method}_mean_instability'
    y_col = f'{method}_mean_ari_all_runs'
    required_columns = {'iterations', 'dataset', x_col, y_col}
    missing_columns = sorted(required_columns - set(metrics_df.columns))
    if missing_columns:
        raise ValueError(f'Missing Figure 5D metrics columns: {missing_columns}')
    grid = parent_spec.subgridspec(nrows=1, ncols=len(D_RUN_COUNTS), wspace=0.3)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(D_RUN_COUNTS))])
    for axi, (ax, runs) in enumerate(zip(axes, D_RUN_COUNTS)):
        tmp = metrics_df[metrics_df['iterations'] == runs]
        xs = tmp[x_col].to_numpy(dtype=float)
        ys = tmp[y_col].to_numpy(dtype=float)
        datasets = tmp['dataset'].to_numpy()
        valid = np.isfinite(xs) & np.isfinite(ys)
        xx = xs[valid]
        yy = ys[valid]
        plot_datasets = datasets[valid]
        ax.set_title(f'{runs} runs', fontsize=20, fontweight='bold', pad=8)
        ax.set_xlabel('Sample Instability', fontsize=16)
        if axi == 0:
            ax.set_ylabel('ARI w.r.t. Ref.', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.0)
        can_correlate = len(xx) >= 2 and np.ptp(xx) > 0 and (np.ptp(yy) > 0)
        if can_correlate:
            r, p = stats.pearsonr(xx, yy)
            r_sq = r ** 2
            print(f'[panel D] {method}, {runs} runs: r={r:.4f}, r2={r_sq:.4f}, p={p:.4g}')
            if method != 'sedr_mclust':
                slope, intercept = np.polyfit(xx, yy, 1)
                fit_x = np.linspace(0, 0.5, 100)
                fit_y = slope * fit_x + intercept
                ax.plot(fit_x, fit_y, color='black', linestyle='--', linewidth=2)
                info = f'r = {r:.2f}\nr² = {r_sq:.2f}\np = {p:.3g}'
                ax.text(0.5, -0.4, info, transform=ax.transAxes, ha='center', va='top', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
        else:
            print(f'[panel D] {method}, {runs} runs: insufficient/non-varying finite data for Pearson correlation')
        for x, y, dataset in zip(xx, yy, plot_datasets):
            color_index = D_DATASET_COLOR_INDEX.get(dataset, 14)
            alpha = 0.3 if dataset == 'dlpfc' else 1.0
            ax.scatter(x, y, color=TABLEAU_20[color_index], s=D_SCATTER_SIZE, alpha=alpha)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return axes

# ---- Figure 5E: cluster instability vs Jaccard ----
def prepare_clusterwise_metrics(cluster_df, method):
    """
    Prepare the cluster-level table used by Figure 5E for one Figure 5 method.

    The intended unit of observation is a cluster. Only RUN_COUNTS =
    5, 10, 20, and 40 are retained. Rows from obsolete/full-size dataset tags
    excluded by the original standalone script are removed explicitly.
    """
    required_columns = {'dataset-sample', 'method', 'iterations', E_XCOL, E_METRIC}
    missing_columns = sorted(required_columns - set(cluster_df.columns))
    if missing_columns:
        raise ValueError(f'Missing Figure 5E clusterwise metrics columns: {missing_columns}')
    excluded_tags = {'ov_ffpe_full', 'coad_ffpe_full', 'Visium_HD_Human_Colon_Cancer'}
    out = cluster_df[~cluster_df['dataset-sample'].isin(excluded_tags) & (cluster_df['method'] == method) & cluster_df['iterations'].isin(E_RUN_COUNTS)].copy()
    sample_tags = out['dataset-sample'].astype(str)
    supported_sample = sample_tags.str.startswith('dlpfc_', na=False) | sample_tags.isin(E_SINGLETON_SAMPLE_TAGS)
    out = out[supported_sample].copy()
    if out.empty:
        raise ValueError(f'No Figure 5E data for method {method}.')
    return out

def figure5e_xmax(cluster_df):
    """
    Choose one shared x-axis maximum for all four Figure 5E panels.

    - If every finite plotted cluster instability is <= 0.7, retain 0.7.
    - Otherwise use the smallest 0.1 increment that contains the observed
      maximum. No data values are clipped or modified.
    """
    xs = cluster_df[E_XCOL].to_numpy(dtype=float)
    ys = cluster_df[E_METRIC].to_numpy(dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid):
        raise ValueError('Figure 5E has no finite cluster instability/Jaccard pairs.')
    observed_max = float(np.max(xs[valid]))
    if observed_max <= E_DEFAULT_XMAX:
        x_max = E_DEFAULT_XMAX
    else:
        x_max = np.ceil((observed_max - 1e-12) / E_XMAX_STEP) * E_XMAX_STEP
        x_max = float(x_max)
    print(f'[panel E] maximum cluster instability={observed_max:.4f}; shared x-axis maximum={x_max:.2f}')
    return x_max

def dataset_mask_for_clusterwise(ydf, dataset):
    """Return rows for one dataset without defining DLPFC by exclusion."""
    sample_tags = ydf['dataset-sample'].astype(str)
    if dataset == 'dlpfc':
        return sample_tags.str.startswith('dlpfc_', na=False)
    return sample_tags == dataset

def plot_fig5e(fig, parent_spec, cluster_df, method):
    """
    Draw Figure 5E for one method and 5, 10, 20, and 40 runs.

    x = clusterwise mean instability
    y = clusterwise Jaccard with respect to the reference

    Each point is a cluster. Correlation and regression are therefore also
    cluster-level analyses, as intended.
    """
    xdf = prepare_clusterwise_metrics(cluster_df, method)
    x_max = figure5e_xmax(xdf)
    grid = parent_spec.subgridspec(nrows=1, ncols=len(E_RUN_COUNTS), wspace=0.3)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(E_RUN_COUNTS))])
    for axi, (ax, runs) in enumerate(zip(axes, E_RUN_COUNTS)):
        ydf = xdf[xdf['iterations'] == runs]
        xs = ydf[E_XCOL].to_numpy(dtype=float)
        ys = ydf[E_METRIC].to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xx = xs[valid]
        yy = ys[valid]
        ax.set_title(f'{runs} runs', fontsize=20, fontweight='bold', pad=8)
        ax.set_xlabel('Cluster Instability', fontsize=16)
        if axi == 0:
            ax.set_ylabel('Jaccard w.r.t. Ref.', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        x_ticks = np.arange(0.0, x_max + 1e-09, 0.2)
        ax.set_xticks(x_ticks)
        if axi == 0:
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        can_correlate = len(xx) >= 2 and np.ptp(xx) > 0 and (np.ptp(yy) > 0)
        if can_correlate:
            r, p = stats.pearsonr(xx, yy)
            r_sq = r ** 2
            print(f'[panel E] {method}, {runs} runs: r={r:.4f}, r2={r_sq:.4f}, p={p:.4g}')
            slope, intercept = np.polyfit(xx, yy, 1)
            fit_x = np.linspace(0, x_max, 100)
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, color='black', linestyle='--', linewidth=2)
            info = f'r = {r:.2f}\nr² = {r_sq:.2f}\np = {p:.3g}'
            ax.text(0.5, -0.4, info, transform=ax.transAxes, ha='center', va='top', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
        else:
            print(f'[panel E] {method}, {runs} runs: insufficient/non-varying finite data for Pearson correlation')
        datasets_to_plot = ['dlpfc', 'mouse_brain_cerebellum', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'human_breast_cancer', 'mouse_brain', 'coad_ffpe', 'ov_ffpe']
        for dataset in datasets_to_plot:
            zdf = ydf[dataset_mask_for_clusterwise(ydf, dataset)]
            zx = zdf[E_XCOL].to_numpy(dtype=float)
            zy = zdf[E_METRIC].to_numpy(dtype=float)
            point_valid = np.isfinite(zx) & np.isfinite(zy)
            zx = zx[point_valid]
            zy = zy[point_valid]
            if zx.size == 0:
                continue
            color_index = D_DATASET_COLOR_INDEX[dataset]
            alpha = E_DLPFC_ALPHA if dataset == 'dlpfc' else E_NON_DLPFC_ALPHA
            ax.scatter(zx, zy, color=TABLEAU_20[color_index], s=E_SCATTER_SIZE, alpha=alpha)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return (axes, x_max)

def add_sample_legend(fig, scatter_axes):
    """
    Shared dataset/sample legend centered below panels D and E.

    Legend marker area matches the Panel D scatter-point area exactly:
    scatter uses s=D_SCATTER_SIZE (points^2), whereas Line2D markersize
    is a diameter in points, hence sqrt(D_SCATTER_SIZE).
    """
    labels = ['DLPFC', 'MB', 'MBC', 'HBC', 'CRC', 'OV', 'COAD']
    datasets = ['dlpfc', 'mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe']
    colors = [TABLEAU_20[D_DATASET_COLOR_INDEX[dataset]] for dataset in datasets]
    legend_marker_size = np.sqrt(D_SCATTER_SIZE)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=legend_marker_size, markerfacecolor=color, markeredgecolor='none', label=label) for label, color in zip(labels, colors)]
    fig.canvas.draw()
    left = min((ax.get_position().x0 for ax in scatter_axes))
    right = max((ax.get_position().x1 for ax in scatter_axes))
    bottom = min((ax.get_position().y0 for ax in scatter_axes))
    x_center = (left + right) / 2
    y = bottom - 0.14
    legend = fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(x_center, y), bbox_transform=fig.transFigure, ncol=len(labels), frameon=False, fontsize=16, handletextpad=0.4, columnspacing=1.7, borderaxespad=0.0)
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bbox.x0 - 0.015, y, 'Sample', fontsize=16, fontweight='bold', ha='right', va='center')
    return legend

# ---- Integrated layout ----
def add_panel_letters(fig, spatial_axes, heatmap_axes, hexbin_axes, d_axes, e_axes):
    """
    Align panel letters as:
        A, B, D -> same x
        C, E    -> same x
        B, C    -> same y
        D, E    -> same y
    """
    fig.canvas.draw()
    x_left = 0.035
    x_right = e_axes[0].get_position().x0 - 0.035
    y_a = spatial_axes[0, 0].get_position().y1 + 0.035
    y_bc = max(max((ax.get_position().y1 for ax in heatmap_axes)), max((ax.get_position().y1 for ax in hexbin_axes))) + 0.03
    y_de = max(max((ax.get_position().y1 for ax in d_axes)), max((ax.get_position().y1 for ax in e_axes))) + 0.033
    positions = {'A': (x_left, y_a), 'B': (x_left, y_bc), 'C': (x_right, y_bc), 'D': (x_left, y_de), 'E': (x_right, y_de)}
    for letter, (x, y) in positions.items():
        fig.text(x, y, letter, fontsize=PANEL_LETTER_FONTSIZE, fontweight='bold', ha='left', va='top')

def add_placeholder_panel(fig, spec, letter):
    """Create an empty panel placeholder and place its panel letter."""
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    fig.canvas.draw()
    pos = ax.get_position()
    fig.text(pos.x0 - 0.025, pos.y1 + 0.012, letter, fontsize=PANEL_LETTER_FONTSIZE, fontweight='bold', ha='left', va='top')
    return ax

def create_figure5(method, transition_df, hexbin_df, metrics_df, clusterwise_df):
    """
    Draw Figure 5 on the actual final-size canvas.

    Current layout:
        A A
        B C
        D E

    A contains the spatial panel, B contains the four transition heatmaps,
    C contains the four direct k-run-vs-50-run hexbin comparisons, D contains
    sample instability versus ARI, and E contains cluster instability versus
    clusterwise Jaccard.
    """
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(nrows=5, ncols=2, height_ratios=FIG5_HEIGHT_RATIOS, hspace=0.08, wspace=FIG5_OUTER_WSPACE, left=FIG5_LEFT, right=FIG5_RIGHT, top=FIG5_TOP, bottom=FIG5_BOTTOM)
    bc_grid = outer[2, :].subgridspec(nrows=1, ncols=2, wspace=FIG5_BC_WSPACE)
    spatial_axes, legend = plot_fig5a(fig, outer[0, :], method)
    heatmap_axes, heatmap_colorbar = plot_fig5b(fig, bc_grid[0, 0], transition_df, [method])
    hexbin_axes, hexbin_colorbar, hexbin_norm = plot_fig5c(fig, bc_grid[0, 1], hexbin_df, method)
    d_axes = plot_fig5d(fig, outer[4, 0], metrics_df, method)
    e_axes, e_xmax = plot_fig5e(fig, outer[4, 1], clusterwise_df, method)
    add_panel_letters(fig, spatial_axes, heatmap_axes, hexbin_axes, d_axes, e_axes)
    add_sample_legend(fig, np.concatenate([d_axes, e_axes]))
    return (fig, spatial_axes, legend, heatmap_axes, heatmap_colorbar, hexbin_axes, hexbin_colorbar, hexbin_norm, d_axes, e_axes, e_xmax)

def save_figure5(fig, method):
    output_stem = OUT_DIR / f'figure5_{method}'
    png_path = output_stem.with_suffix('.png')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    print(f'[saved] {png_path}')
    pdf_path = output_stem.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    print(f'[saved] {pdf_path}')
    plt.close(fig)

# ---- Main ----
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for method in METHODS: # One method controls panels A-E.
        #method= 'sedr'
        transition_df = build_transition_dataframe([method])
        hexbin_df = build_hexbin_dataframe(method)
        if not METRICS_CSV.exists():
            raise FileNotFoundError(f'Figure 5D metrics CSV not found: {METRICS_CSV}')
        metrics_df = pd.read_csv(METRICS_CSV)
        if not CLUSTERWISE_METRICS_CSV.exists():
            raise FileNotFoundError(f'Figure 5E clusterwise metrics CSV not found: {CLUSTERWISE_METRICS_CSV}')
        clusterwise_df = pd.read_csv(CLUSTERWISE_METRICS_CSV)
        fig = create_figure5(method, transition_df, hexbin_df, metrics_df, clusterwise_df)[0]
        save_figure5(fig, method)
        #break
if __name__ == '__main__':
    main()
