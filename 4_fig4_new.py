#!/usr/bin/env python3
# Run in the visiumhd3 environment.
# Refactored for compactness; plotting/data semantics are intentionally preserved.

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import anndata as ad
import geopandas as gpd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
# Import registers the sdata.pl plotting accessor.
import spatialdata_plot
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from shapely import box as shapely_box
from shapely.geometry import Polygon
from spatialdata.models import Image2DModel, ShapesModel, TableModel

# ============================================================================
# GLOBAL FIGURE SETTINGS
# ============================================================================
plt.rcParams['font.family'] = 'Arial'
# Keep text editable in Illustrator-exported PDF/PS files.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
FIGSIZE = (22, 13.5)
DPI = 300

# ============================================================================
# FIGURE 4 LAYOUT
# ============================================================================
FIG4_LEFT = 0.08
FIG4_RIGHT = 0.995
FIG4_TOP = 0.96
FIG4_BOTTOM = 0.08
FIG4_OUTER_HSPACE = 0.08
FIG4_OUTER_WSPACE = 0.16
FIG4_HEIGHT_RATIOS = [0.8, 0.8, 1, 1, 1.1]

# ============================================================================
# OPTIONAL PANEL GUIDES
# ============================================================================
SHOW_PANEL_GUIDES = False

# ============================================================================
# INPUT / OUTPUT PATHS
# ============================================================================
DATA_ROOT = Path('/home/wuw15/data_dir/ST_datasets/AAA_with_uncert')
RESULT_ROOT = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs')
OUT_DIR = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig4')
ALIGNMENT_DIR = OUT_DIR / 'alignment_dgea_data'
OUT_STEM = OUT_DIR / 'figure4_dgea'
FIG4_METHOD = 'sedr'

# ============================================================================
# FIGURE 4A SETTINGS
# ============================================================================
N_RUNS = 50
EXAMPLE_RUN_INDEX = 14
SELECTED_EXAMPLE_CLUSTERS = ['0', '1', '4', '7', '8', '11']
RUN_ROWS = [list(range(0, 16)), list(range(16, 33)), list(range(33, 50))]
RUN_ROW_LABELS = ['runs\n1-16', 'runs\n17-33', 'runs\n34-50']
RUN_GRID_COLUMNS = 17

# ============================================================================
# FIGURE 4B SETTINGS
# ============================================================================
FIG4B_DATASET_SAMPLES = [('dlpfc', '151507'), ('dlpfc', '151508'), ('dlpfc', '151509'), ('dlpfc', '151510'), ('dlpfc', '151669'), ('dlpfc', '151670'), ('dlpfc', '151671'), ('dlpfc', '151672'), ('dlpfc', '151673'), ('dlpfc', '151674'), ('dlpfc', '151675'), ('dlpfc', '151676'), ('mouse_brain', None), ('mouse_brain_cerebellum', None), ('human_breast_cancer', None), ('Visium_HD_Human_Colon_Cancer_cropped_square', None), ('ov_ffpe', None), ('coad_ffpe', None)]
FIG4B_RANDOM_SEED = 42
FIG4B_INNER_WSPACE = 0.16
FIG4B_TITLE_FONTSIZE = 19
FIG4B_AXIS_LABEL_FONTSIZE = 15
FIG4B_TICK_FONTSIZE = 13.5
FIG4B_SCATTER_SIZE = 34
FIG4B_HEIGHT_RATIOS = [1.0, 0.32]

# ============================================================================
# FIGURE 4C SETTINGS
# ============================================================================
FIG4C_DATASET_SAMPLES = FIG4B_DATASET_SAMPLES.copy()
FIG4C_TOP_KS = [10, 20, 50, 200]
FIG4C_METRICS = {'jaccard': {'column': 'jaccard_mean', 'ylabel': 'Jaccard'}, 'abs_rank_diff': {'column': 'rank_abs_diff_shared_mean', 'ylabel': 'Abs rank diff (shared)'}}
FIG4C_INNER_WSPACE = 0.3
FIG4C_INNER_HSPACE = 0.05
FIG4C_TITLE_FONTSIZE = 20
FIG4C_AXIS_LABEL_FONTSIZE = 16
FIG4C_TICK_FONTSIZE = 14
FIG4C_SHARED_XLABEL_FONTSIZE = 16
FIG4C_HEIGHT_RATIOS = [1.0, 0.18, 1.0, 0.42, 0.2]

# ============================================================================
# FIGURE 4D SETTINGS
# ============================================================================
FIG4D_METRICS_CSV = Path('/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs_new_diff_r_faster_standalonek_sampling/zzz-all_plots/fig5/clusterwise_metrics_by_number_of_runs_new.csv')
FIG4D_METRICS = [('clusterwise jaccard', 'Jaccard'), ('clusterwise recall', 'Recall'), ('clusterwise purity (precision)', 'Purity')]
FIG4D_XCOL = 'clusterwise mean instability'
FIG4D_ITERATIONS = 50
TABLEAU_20 = plt.cm.tab20.colors
FIG4D_DATASET_COLOR_INDEX = {'mouse_brain_cerebellum': 0, 'mouse_brain': 8, 'Visium_HD_Human_Colon_Cancer_cropped_square': 2, 'human_breast_cancer': 12, 'coad_ffpe': 16, 'ov_ffpe': 18, 'dlpfc': 14}
FIG4D_SCATTER_SIZE = 20
FIG4D_NON_DLPFC_ALPHA = 0.8
FIG4D_DLPFC_ALPHA = 0.3
FIG4D_TITLE_FONTSIZE = 20
FIG4D_AXIS_LABEL_FONTSIZE = 16
FIG4D_TICK_FONTSIZE = 14
FIG4D_STATS_FONTSIZE = 14
FIG4D_INNER_WSPACE = 0.18
FIG4D_INNER_HSPACE = 0.04
FIG4D_HEIGHT_RATIOS = [1.0, 0.26, 0.3]

# ============================================================================
# SPATIAL DISPLAY SETTINGS
# ============================================================================
COLUMN_WSPACE = 0.06
ROW_HSPACE = 0.02
CROP_PADDING = 0.035
SPATIAL_IMAGE_ALPHA = 0.45
SPATIAL_CIRCLE_SIZE = 0.48
SPATIAL_SQUARE_SIZE = 1.0
SPATIAL_FLAT_HEX_SIZE = 1.0
SPATIAL_POINTY_HEX_SIZE = 1.0
SPATIAL_FRAME_LINEWIDTH = 0.35
ROW_LABEL_FONTSIZE = 18
CONTEXT_TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 14
REFERENCE_LEGEND_FONTSIZE = 12
EXAMPLE_LABEL_FONTSIZE = 15
CONTEXT_FRAME_LINEWIDTH = 3.0
CONTEXT_WORKFLOW_FONTSIZE = 13
# CONTEXT_BINNED_CLUSTER_BINS = ['[0.0,0.1)', '[0.3,0.4)']
CONTEXT_BINNED_CLUSTER_BINS = ['[0.0,0.1)', '[0.3,0.4)', '[0.5,0.6)']

# ============================================================================
# DATA SETTINGS
# ============================================================================
ANNOTATION_COLUMN = 'annotation'
OTHER_SPOTS_COLOR = '#d9d9d9'
COLLISION_COLOR = '#000000'
INSTABILITY_LABELS = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,1.0]']
INSTABILITY_BINS = np.r_[np.arange(0.0, 1.0, 0.1), np.nextafter(1.0, np.inf)]
BIN_COLORS = {'[0.0,0.1)': '#08306b', '[0.1,0.2)': '#4292c6', '[0.2,0.3)': '#9ecae1', '[0.3,0.4)': '#deebf7', '[0.4,0.5)': '#fee391', '[0.5,0.6)': '#fec44f', '[0.6,0.7)': '#fdae6b', '[0.7,0.8)': '#fd8d3c', '[0.8,0.9)': '#fb6a4a', '[0.9,1.0]': '#cb181d'}
COLOR_LIST = ['#3049ad', '#fe8011', '#1b7837', '#fa0000', '#ab43fc', '#8d574c', '#ff00d9', '#bcbd22', '#17becf', '#8baaf3', '#ffbb79', '#99df8b', '#fe7775', '#c6b1d4', '#c49d95', '#ff80c6', '#dcdb91', '#a7d1e6', '#393b79', '#8c6d31', '#0aac00', '#982109', '#7b4173', '#713230', '#ff008c', '#637939', '#e7cb94', '#ccefc5', '#efcece', '#f7b6d2', '#eeedc8']
VISIUM_HD_DATASETS = {'Visium_HD_Human_Colon_Cancer_cropped_square', 'Visium_HD_Human_Colon_Cancer', 'ov_ffpe', 'ov_ffpe_full', 'coad_ffpe', 'coad_ffpe_full', 'SCAF4093_3229997_A1'}


# ============================================================================
# PATH HELPERS
# ============================================================================
def safe_name(value):
    return str(value).replace('/', '_').replace('\\', '_').replace(' ', '_')

def dataset_tag(dataset, sample):
    if sample is None:
        return dataset
    return f'{dataset}_{sample}'

def get_paths(dataset, sample, method):
    """
    Paths retained from the existing Figure 4A script.
    """
    prefix = f'_{sample}_' if sample is not None else ''
    adata_path = DATA_ROOT / f'{dataset}{prefix}.h5ad'
    result_dir = RESULT_ROOT / dataset / method
    suffix = 'csv' if method == 'bayesspace' else 'pkl'
    result_path = result_dir / f'{prefix}{method}_results_50iterations.{suffix}'
    return (adata_path, result_path)

def uncertainty_h5ad_path(dataset, sample):
    """
    Figure 3-style h5ad containing <method>_uncertainty.

    Example:
        human_breast_cancer.h5ad
        dlpfc_151507.h5ad
    """
    return DATA_ROOT / f'{dataset_tag(dataset, sample)}.h5ad'

def alignment_path(dataset, sample, method):
    sample_part = f'_{safe_name(sample)}' if sample is not None else ''
    return ALIGNMENT_DIR / f'{safe_name(dataset)}{sample_part}_{safe_name(method)}_best_cluster_matches.csv'


# ============================================================================
# RESULT LOADING
# ============================================================================
def load_results(path, method):
    if method == 'bayesspace':
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
        results = [df[col].to_numpy() for col in df.columns]
    else:
        results = [np.asarray(x) for x in joblib.load(path)]
    if len(results) < N_RUNS:
        raise ValueError(f'{path} contains only {len(results)} runs.')
    return results[:N_RUNS]

def cluster_string(value):
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


# ============================================================================
# ALIGNMENT DATA
# ============================================================================
def load_alignment(dataset, sample, method):
    path = alignment_path(dataset, sample, method)
    if not path.exists():
        raise FileNotFoundError(f'Alignment file not found: {path}')
    matches = pd.read_csv(path)
    required = {'example_cluster', 'run', 'example_run_idx', 'matched_run_cluster', 'best_jaccard', 'example_mean_instability', 'example_fine_bin'}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f'{path.name} is missing columns: {sorted(missing)}')
    matches['example_cluster'] = matches['example_cluster'].map(cluster_string)
    matches['matched_run_cluster'] = matches['matched_run_cluster'].map(cluster_string)
    matches['run'] = pd.to_numeric(matches['run'], errors='raise').astype(int)
    matches['example_run_idx'] = pd.to_numeric(matches['example_run_idx'], errors='raise').astype(int)
    matches['best_jaccard'] = pd.to_numeric(matches['best_jaccard'], errors='coerce')
    matches['example_mean_instability'] = pd.to_numeric(matches['example_mean_instability'], errors='coerce')
    matches['example_fine_bin'] = matches['example_fine_bin'].astype(str)
    if matches.duplicated(['example_cluster', 'run']).any():
        raise ValueError(f'{path.name} contains multiple selected matches for the same example-cluster/run pair.')
    return matches


# ============================================================================
# IMAGE / GEOMETRY HELPERS
# ============================================================================
def get_image_and_coords(adata, preferred_key='hires'):
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    spatial = adata.uns.get('spatial', {})
    if not isinstance(spatial, dict) or not spatial:
        return (None, coords)
    library = spatial[next(iter(spatial))]
    images = library.get('images', {})
    scales = library.get('scalefactors', {})
    keys = ('hires', 'lowres') if preferred_key == 'hires' else ('lowres', 'hires')
    for key in keys:
        if key in images:
            scale = float(scales.get(f'tissue_{key}_scalef', 1.0))
            return (np.asarray(images[key]), coords * scale)
    return (None, coords)

def to_grayscale(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    rgb = image[..., :3].astype(float)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(gray, 0, np.iinfo(image.dtype).max)
    else:
        gray = np.clip(gray, 0.0, 1.0)
    return gray.astype(image.dtype)

def nearest_spacing(coords):
    unique_coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    if len(unique_coords) < 2:
        return 1.0
    distances = cKDTree(unique_coords).query(unique_coords, k=2)[0][:, 1]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size:
        return float(np.median(distances))
    return 1.0

def make_hexagons(coords, radius, angles):
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    return [Polygon(np.column_stack((x + radius * cos_angles, y + radius * sin_angles))) for x, y in coords]

def make_shapes(coords, index, dataset):
    spacing = nearest_spacing(coords)
    if dataset == 'mouse_brain_cerebellum':
        gdf = gpd.GeoDataFrame({'radius': np.full(len(coords), SPATIAL_CIRCLE_SIZE * spacing)}, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]), index=index)
    elif dataset in VISIUM_HD_DATASETS:
        half_size = SPATIAL_SQUARE_SIZE * spacing / 2
        gdf = gpd.GeoDataFrame(geometry=shapely_box(coords[:, 0] - half_size, coords[:, 1] - half_size, coords[:, 0] + half_size, coords[:, 1] + half_size), index=index)
    elif dataset == 'mouse_brain':
        radius = SPATIAL_FLAT_HEX_SIZE * spacing / np.sqrt(3)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        gdf = gpd.GeoDataFrame(geometry=make_hexagons(coords, radius, angles), index=index)
    elif dataset in {'dlpfc', 'human_breast_cancer'}:
        radius = SPATIAL_POINTY_HEX_SIZE * spacing / np.sqrt(3)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        gdf = gpd.GeoDataFrame(geometry=make_hexagons(coords, radius, angles), index=index)
    else:
        gdf = gpd.GeoDataFrame({'radius': np.full(len(coords), SPATIAL_CIRCLE_SIZE * spacing)}, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]), index=index)
    return ShapesModel.parse(gdf)

def make_spatialdata(adata, dataset):
    image, coords = get_image_and_coords(adata)
    index = pd.Index(adata.obs_names.astype(str), name='instance_id')
    shapes = make_shapes(coords, index, dataset)
    table_obs = pd.DataFrame(index=index)
    table_obs['region'] = pd.Categorical(['spots'] * adata.n_obs)
    table_obs['instance_id'] = index
    table = TableModel.parse(ad.AnnData(obs=table_obs), region='spots', region_key='region', instance_key='instance_id')
    images = {}
    if image is not None:
        images['histology'] = Image2DModel.parse(to_grayscale(image)[None, ...], dims=('c', 'y', 'x'), c_coords=['gray'])
    sdata = sd.SpatialData(images=images, shapes={'spots': shapes}, tables={'table': table})
    return (sdata, (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()))


# ============================================================================
# SPATIAL VIEW HELPERS
# ============================================================================
def square_limits(bounds):
    min_x, min_y, max_x, max_y = bounds
    x_span = max(max_x - min_x, 1)
    y_span = max(max_y - min_y, 1)
    px = CROP_PADDING * x_span
    py = CROP_PADDING * y_span
    return (min_x - px, max_x + px, min_y - py, max_y + py)

def apply_spatial_view(ax, bounds):
    x0, x1, y0, y1 = square_limits(bounds)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('C')

def keep_black_frame(ax, linewidth=SPATIAL_FRAME_LINEWIDTH):
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(linewidth)
        spine.set_linestyle('-')

def style_example_frame(ax):
    """
    Thick dotted, slightly rounded frame
    for the example run.
    """
    style_rounded_spatial_frame(ax, linewidth=3.0, linestyle=(0, (3, 2)), rounding=0.025)

def style_rounded_spatial_frame(ax, linewidth=3.0, linestyle='-', rounding=0.025):
    """
    Slightly rounded frame for spatial plots.
    Can be solid or dotted/dashed.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    old_frame = getattr(ax, '_rounded_spatial_frame', None)
    if old_frame is not None:
        old_frame.remove()
    frame = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, boxstyle=f'round,pad=0,rounding_size={rounding}', fill=False, edgecolor='black', linewidth=linewidth, linestyle=linestyle, clip_on=False, zorder=20)
    ax.add_patch(frame)
    ax._rounded_spatial_frame = frame
    return frame


# ============================================================================
# GENERIC SPATIALDATA RENDERER
# ============================================================================
def render_categorical_map(ax, sdata, bounds, color, palette, groups=None, fill_alpha=1.0, show_histology=True):
    plot = sdata
    # Histology is optional; categorical shapes render either way.
    if show_histology and 'histology' in sdata.images:
        plot = plot.pl.render_images(element='histology', cmap='gray', alpha=SPATIAL_IMAGE_ALPHA)
    render_kwargs = dict(element='spots', color=color, palette=palette, fill_alpha=fill_alpha, outline_width=0, table_name='table', method='matplotlib')
    if groups is not None:
        render_kwargs['groups'] = groups
        render_kwargs['na_color'] = '#00000000'
    plot.pl.render_shapes(**render_kwargs).pl.show(ax=ax, title='', frameon=True, legend_loc=None, colorbar=False, show=False)

    for collection in ax.collections:
        collection.set_rasterized(True)

    apply_spatial_view(ax, bounds)
    keep_black_frame(ax)


# ============================================================================
# SPOT INSTABILITY
# ============================================================================
def load_spot_instability(base_adata, dataset, sample, method):
    key = f'{method}_uncertainty'
    if key in base_adata.obs.columns:
        print(f'[spot instability] using {key} from Figure 4 input h5ad')
        return base_adata.obs[key].to_numpy(dtype=float)
    path = uncertainty_h5ad_path(dataset, sample)
    if not path.exists():
        raise FileNotFoundError(f"Spot instability could not be loaded.\n'{key}' is absent from the Figure 4 input h5ad and the Figure 3-style h5ad does not exist:\n{path}")
    uncertainty_adata = sc.read_h5ad(path)
    if key not in uncertainty_adata.obs.columns:
        raise ValueError(f"{path} does not contain adata.obs['{key}'].")
    source_names = pd.Index(uncertainty_adata.obs_names.astype(str))
    target_names = pd.Index(base_adata.obs_names.astype(str))
    missing_ids = target_names.difference(source_names)
    if len(missing_ids):
        raise ValueError(f'{len(missing_ids)} Figure 4 spots are absent from {path.name}. First few: {missing_ids[:10].tolist()}')
    uncertainty_series = pd.Series(uncertainty_adata.obs[key].to_numpy(dtype=float), index=source_names)
    aligned = uncertainty_series.reindex(target_names).to_numpy(dtype=float)
    del uncertainty_adata
    print(f'[spot instability] using {key} from {path}')
    return aligned

def add_spot_instability_to_table(sdata, values):
    sdata.tables['table'].obs['spot_instability_bin'] = pd.cut(values, bins=INSTABILITY_BINS, labels=INSTABILITY_LABELS, include_lowest=True, right=False, ordered=True)


# ============================================================================
# PREPARE FIGURE 4A DATA
# ============================================================================
def prepare_figure4_data(dataset, sample, method, selected_example_clusters):
    adata_path, result_path = get_paths(dataset, sample, method)
    if not adata_path.exists():
        raise FileNotFoundError(f'Figure 4 h5ad not found:\n{adata_path}')
    if not result_path.exists():
        raise FileNotFoundError(f'50-run result file not found:\n{result_path}')
    print(f'[Figure 4] dataset={dataset}, sample={sample}, method={method}')
    adata = sc.read_h5ad(adata_path)
    results = load_results(result_path, method)
    matches = load_alignment(dataset, sample, method)
    # Automatically include example-run cluster(s) in the 0.5-0.6 bin
    extra_clusters = (
        matches.loc[
            matches['example_fine_bin'].eq('[0.5,0.6)'),
            'example_cluster'
        ]
        .dropna()
        .map(cluster_string)
        .drop_duplicates()
        .tolist()
    )

    print('[0.5-0.6) example cluster(s):', extra_clusters)

    selected_example_clusters = [
        cluster_string(x) for x in selected_example_clusters
    ]

    for cluster in extra_clusters:
        if cluster not in selected_example_clusters:
            selected_example_clusters.append(cluster)

    # Preserve the original SEDR mclust spot reordering.
    if method == 'sedr_mclust':
        if sample is None:
            raise ValueError('sedr_mclust requires a sample ID.')
        idx_path = RESULT_ROOT / dataset / method / f'{sample}_idx.pkl'
        idx = np.asarray(joblib.load(idx_path))
        results = [np.asarray(labels)[idx] for labels in results]
    for run_index, labels in enumerate(results):
        if len(labels) != adata.n_obs:
            raise ValueError(f'Run {run_index} has {len(labels)} labels but {adata_path} has {adata.n_obs} spots.')
    example_runs = matches['example_run_idx'].unique()
    if len(example_runs) != 1:
        raise ValueError(f'Expected one example run, found {example_runs.tolist()}.')
    example_run = int(example_runs[0])
    if example_run != EXAMPLE_RUN_INDEX:
        raise ValueError(f'Alignment file example run does not match the Figure 4 layout setting.\nalignment file: {example_run}\nEXAMPLE_RUN_INDEX: {EXAMPLE_RUN_INDEX}')
    #selected_example_clusters = [cluster_string(cluster) for cluster in selected_example_clusters]
    if len(selected_example_clusters) != len(set(selected_example_clusters)):
        raise ValueError('SELECTED_EXAMPLE_CLUSTERS contains duplicates.')
    available_clusters = set(matches['example_cluster'].dropna())
    unavailable = set(selected_example_clusters) - available_clusters
    if unavailable:
        raise ValueError(f'Selected example clusters not found: {sorted(unavailable)}.\nAvailable clusters: {sorted(available_clusters)}')
    if len(selected_example_clusters) > len(COLOR_LIST):
        raise ValueError('Not enough colors for selected clusters.')
    if ANNOTATION_COLUMN not in adata.obs.columns or not adata.obs[ANNOTATION_COLUMN].notna().any():
        raise ValueError(f"No usable '{ANNOTATION_COLUMN}' annotation was found.")
    annotation_raw = adata.obs[ANNOTATION_COLUMN]
    annotation = annotation_raw.astype('string').fillna('Unknown').astype(str)
    if isinstance(annotation_raw.dtype, pd.CategoricalDtype):
        annotation_categories = list(annotation_raw.cat.categories.astype(str))
    else:
        annotation_categories = list(pd.unique(annotation))
    if 'Unknown' in annotation.values and 'Unknown' not in annotation_categories:
        annotation_categories.append('Unknown')
    if len(annotation_categories) > len(COLOR_LIST):
        raise ValueError('Not enough colors for reference annotation categories.')
    example_labels = np.asarray(results[example_run]).astype(str)
    overlap = pd.crosstab(pd.Series(example_labels, name='example_cluster'), pd.Series(annotation.to_numpy(), name='annotation'))
    example_to_annotation = overlap.idxmax(axis=1).to_dict()
    missing_from_example = [cluster for cluster in selected_example_clusters if cluster not in example_to_annotation]
    if missing_from_example:
        raise ValueError(f'Selected clusters absent from example run: {missing_from_example}')
    cluster_colors = dict(zip(selected_example_clusters, COLOR_LIST[:len(selected_example_clusters)]))
    # Preserve selected-cluster order so cluster and reference colors remain paired.
    selected_reference_labels = [example_to_annotation[cluster] for cluster in selected_example_clusters]
    if len(selected_reference_labels) != len(set(selected_reference_labels)):
        raise ValueError('Two selected example clusters match the same reference label. The reference label cannot be assigned two different colors.')
    selected_reference_colors = {example_to_annotation[cluster]: cluster_colors[cluster] for cluster in selected_example_clusters}
    unselected_reference_labels = [label for label in annotation_categories if label not in selected_reference_colors]
    remaining_colors = COLOR_LIST[len(selected_example_clusters):]
    if len(unselected_reference_labels) > len(remaining_colors):
        raise ValueError('Not enough remaining colors for unselected reference labels.')
    reference_palette = dict(zip(unselected_reference_labels, remaining_colors))
    reference_palette.update(selected_reference_colors)
    selected_matches = matches.loc[matches['example_cluster'].isin(selected_example_clusters)].copy()
    if selected_matches['matched_run_cluster'].isna().any():
        raise ValueError('At least one selected alignment has no matched cluster.')
    expected_pairs = {(cluster, run) for cluster in selected_example_clusters for run in range(N_RUNS)}
    observed_pairs = set(selected_matches[['example_cluster', 'run']].itertuples(index=False, name=None))
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(f'Missing alignment rows: {missing_pairs}')
    sdata, bounds = make_spatialdata(adata, dataset)
    spot_instability = load_spot_instability(adata, dataset, sample, method)
    add_spot_instability_to_table(sdata, spot_instability)
    return {'dataset': dataset, 'sample': sample, 'method': method, 'adata': adata, 'results': results, 'matches': matches, 'selected_matches': selected_matches, 'selected_example_clusters': selected_example_clusters, 'example_run': example_run, 'example_labels': example_labels, 'annotation': annotation, 'annotation_categories': annotation_categories, 'selected_reference_labels': selected_reference_labels, 'cluster_colors': cluster_colors, 'reference_palette': reference_palette, 'sdata': sdata, 'bounds': bounds}


# ============================================================================
# ALIGNED CLUSTER DISPLAY
# ============================================================================
def build_aligned_display(run_index, labels, selected_matches, selected_example_clusters, cluster_colors):
    run_matches = selected_matches.loc[selected_matches['run'].eq(run_index)].copy()
    target_to_examples = run_matches.groupby('matched_run_cluster')['example_cluster'].agg(list).to_dict()
    labels = np.asarray(labels).astype(str)
    labels_present = set(labels)
    display = np.full(len(labels), 'Other', dtype=object)
    run_collision_rows = []
    for matched_cluster, example_clusters in target_to_examples.items():
        if matched_cluster not in labels_present:
            raise ValueError(f'Matched cluster {matched_cluster} is absent from run {run_index}.')
        if len(example_clusters) == 1:
            display[labels == matched_cluster] = example_clusters[0]
        else:
            display[labels == matched_cluster] = 'Collision'
            run_collision_rows.append({'run': run_index, 'displayed_run': run_index + 1, 'matched_run_cluster': matched_cluster, 'example_clusters': ','.join(example_clusters)})
    categories = ['Other'] + list(selected_example_clusters)
    palette = {'Other': OTHER_SPOTS_COLOR, **cluster_colors}
    visible_groups = list(selected_example_clusters)
    if run_collision_rows:
        categories.append('Collision')
        palette['Collision'] = COLLISION_COLOR
        visible_groups.append('Collision')
    return (display, categories, palette, visible_groups, run_collision_rows)

def plot_aligned_run(ax, data, run_index, record_collisions=True):
    display, categories, palette, visible_groups, run_collision_rows = build_aligned_display(run_index=run_index, labels=data['results'][run_index], selected_matches=data['selected_matches'], selected_example_clusters=data['selected_example_clusters'], cluster_colors=data['cluster_colors'])
    data['sdata'].tables['table'].obs['aligned_cluster'] = pd.Categorical(display, categories=categories, ordered=True)
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='aligned_cluster', palette=palette, groups=visible_groups, fill_alpha=0.98)
    if run_index == data['example_run']:
        style_example_frame(ax)
    if record_collisions:
        return run_collision_rows
    return []

def match_axes_size_to_reference(fig, reference_ax, axes_to_resize, scale=1.2):
    """
    Resize selected axes relative to reference_ax,
    while keeping each axis centered.
    """
    fig.canvas.draw()
    ref_pos = reference_ax.get_position()
    target_width = ref_pos.width * scale
    target_height = ref_pos.height * scale
    for ax in axes_to_resize:
        pos = ax.get_position()
        center_x = (pos.x0 + pos.x1) / 2
        center_y = (pos.y0 + pos.y1) / 2
        ax.set_position([center_x - target_width / 2, center_y - target_height / 2, target_width, target_height])

def add_run_row_label(ax, text):
    ax.text(-0.12, 0.5, text, ha='right', va='center', fontsize=ROW_LABEL_FONTSIZE, fontweight='bold', linespacing=0.95, transform=ax.transAxes, clip_on=False)

def draw_example_label_cell(ax, example_run):
    """
    Dedicated rectangular grid cell.

    Nothing extends outside the cell, so there is no
    irregular callout shape in the final figure.
    """
    ax.set_axis_off()
    ax.annotate('', xy=(0.04, 0.52), xytext=(0.28, 0.52), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle='-|>', linewidth=1.1, color='black'))
    ax.text(0.58, 0.62, 'Example\nRun', ha='center', va='center', fontsize=EXAMPLE_LABEL_FONTSIZE, fontweight='bold', linespacing=0.95, transform=ax.transAxes)
    ax.text(0.58, 0.3, f'1st with\nmin. clusters\n(7 shown)', ha='center', va='center', fontsize=11, transform=ax.transAxes)


# ============================================================================
# PANEL A: 50 ALIGNED RUNS
# ============================================================================
def plot_fig4a(fig, parent_spec, data):
    """
    Figure 4A.

    Three run rows:

        runs 1-16
        runs 17-33
        runs 34-50

    All rows use 17 equal-width slots.

    In row 1:
        run indices 0-14
        dedicated Example Run label cell
        run index 15
    """
    if data['example_run'] not in RUN_ROWS[0]:
        raise ValueError('The configured example run must be in the first Figure 4A row.')
    panel_grid = parent_spec.subgridspec(nrows=3, ncols=1, hspace=ROW_HSPACE)
    run_axes = {}
    row_first_axes = []
    collision_rows = []
    example_label_ax = None
    row_items = []
    first_row = list(RUN_ROWS[0])
    insert_position = first_row.index(data['example_run']) + 1
    first_row.insert(insert_position, 'EXAMPLE_LABEL')
    if len(first_row) != RUN_GRID_COLUMNS:
        raise ValueError('First Figure 4A row does not contain 17 visual slots.')
    row_items.append(first_row)
    row_items.append(list(RUN_ROWS[1]))
    row_items.append(list(RUN_ROWS[2]))
    for row_idx, items in enumerate(row_items):
        if len(items) != RUN_GRID_COLUMNS:
            raise ValueError(f'Figure 4A row {row_idx} has {len(items)} slots instead of {RUN_GRID_COLUMNS}.')
        row_grid = panel_grid[row_idx, 0].subgridspec(nrows=1, ncols=RUN_GRID_COLUMNS, wspace=COLUMN_WSPACE)
        first_run_ax = None
        for slot_idx, item in enumerate(items):
            ax = fig.add_subplot(row_grid[0, slot_idx])
            if item == 'EXAMPLE_LABEL':
                example_label_ax = ax
                draw_example_label_cell(ax, data['example_run'])
                continue
            run_index = int(item)
            if first_run_ax is None:
                first_run_ax = ax
            run_axes[run_index] = ax
            new_collisions = plot_aligned_run(ax=ax, data=data, run_index=run_index, record_collisions=True)
            collision_rows.extend(new_collisions)
        add_run_row_label(first_run_ax, RUN_ROW_LABELS[row_idx])
        row_first_axes.append(first_run_ax)
    return (run_axes, row_first_axes, example_label_ax, collision_rows)

def plot_spot_instability_map(ax, data):
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='spot_instability_bin', palette=BIN_COLORS, groups=None, fill_alpha=1.0)
    ax.set_title('Instability Map', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', pad=8)

def add_instability_legend_5x2(ax):
    ax.set_axis_off()
    handles = [Patch(facecolor=BIN_COLORS[label], edgecolor='none', label=label) for label in INSTABILITY_LABELS]
    order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    handles = [handles[i] for i in order]
    labels = [INSTABILITY_LABELS[i] for i in order]
    legend = ax.legend(handles=handles, labels=labels, title='', loc='center left', ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.8, handleheight=0.8, handletextpad=0.5, columnspacing=1, labelspacing=0.4, borderaxespad=0.0)
    legend.get_title().set_fontweight('bold')
    try:
        legend._legend_box.align = 'left'
    except AttributeError:
        pass
    return legend

def plot_reference_annotation(ax, data):
    data['sdata'].tables['table'].obs['reference_annotation'] = pd.Categorical(data['annotation'].to_numpy(), categories=data['annotation_categories'], ordered=True)
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='reference_annotation', palette=data['reference_palette'], groups=data['selected_reference_labels'], fill_alpha=1.0)
    ax.set_title('Reference\nAnnotation', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', pad=8)

def add_reference_legend(ax, data):
    ax.set_axis_off()
    handles = [Line2D([0], [0], linestyle='none', marker='h', markersize=11, markerfacecolor=data['cluster_colors'][cluster], markeredgecolor='none', label=data['selected_reference_labels'][i]) for i, cluster in enumerate(data['selected_example_clusters'])]
    legend = ax.legend(handles=handles, loc='center left', frameon=False, fontsize=REFERENCE_LEGEND_FONTSIZE, handletextpad=0.5, labelspacing=0.55, borderaxespad=0.0, title='')
    legend.get_title().set_fontweight('bold')
    return legend

def plot_context_example_run(ax, data):
    plot_aligned_run(ax=ax, data=data, run_index=data['example_run'], record_collisions=False)
    ax.set_title('Example Run', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', linespacing=0.95, pad=8)

def add_context_bottom_title(ax, text, y=-0.11):
    """
    Bold title below a context spatial map.
    """
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', linespacing=0.95, clip_on=False)

def add_context_workflow_box(ax, text):
    """
    Dashed explanatory box on the right side
    of the context schematic.
    """
    ax.set_axis_off()
    ax.text(0.03, 0.95, text, transform=ax.transAxes, ha='left', va='top', fontsize=CONTEXT_WORKFLOW_FONTSIZE, linespacing=1.2, bbox=dict(boxstyle='round,pad=0.38', facecolor='white', edgecolor='black', linewidth=1.2, linestyle='--'))

def plot_context_binned_clusters(ax, data, fine_bin):
    """
    Show only example-run clusters whose mean
    instability belongs to one selected bin.

    No histology is plotted.
    """
    cluster_bins = data['selected_matches'][['example_cluster', 'example_fine_bin']].drop_duplicates()
    selected_clusters = cluster_bins.loc[cluster_bins['example_fine_bin'].eq(fine_bin), 'example_cluster'].astype(str).tolist()
    example_labels = np.asarray(data['example_labels']).astype(str)
    display = np.where(np.isin(example_labels, selected_clusters), example_labels, 'Other')
    categories = ['Other'] + selected_clusters
    data['sdata'].tables['table'].obs['context_binned_cluster'] = pd.Categorical(display, categories=categories, ordered=True)
    palette = {'Other': OTHER_SPOTS_COLOR}
    for cluster in selected_clusters:
        palette[cluster] = data['cluster_colors'][cluster]
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='context_binned_cluster', palette=palette, groups=selected_clusters, fill_alpha=0.98, show_histology=False)
    style_rounded_spatial_frame(ax, linewidth=CONTEXT_FRAME_LINEWIDTH, linestyle=(0, (3, 2)), rounding=0.025)
    ax.set_title('')
    handle = Patch(facecolor=BIN_COLORS[fine_bin], edgecolor='none')
    ax.legend(handles=[handle], labels=[fine_bin], loc='upper center', bbox_to_anchor=(0.5, -0.035), frameon=False, fontsize=10, handlelength=0.8, handletextpad=0.35, borderaxespad=0.0)


# ============================================================================
# UNLABELED CONTEXT REGION
# ============================================================================
# def plot_fig4_context(fig, parent_spec, data):
#     """
#     Unlabeled Figure 4 schematic.

#     Top:
#         Instability Map
#         | instability legend
#         | Example Run
#         | explanation box

#     Bottom:
#         Reference Annotation
#         | reference legend
#         | two binned-cluster plots
#         | explanation box
#     """
#     wrapper = parent_spec.subgridspec(nrows=3, ncols=3, height_ratios=[0.04, 0.92, 0.04], width_ratios=[0.03, 0.94, 0.03], hspace=0.0, wspace=0.0)
#     grid = wrapper[1, 1].subgridspec(nrows=2, ncols=4, width_ratios=[1.0, 1.7, 2.05, 2.1], height_ratios=[1.0, 1.0], wspace=0.2, hspace=0.3)
#     instability_ax = fig.add_subplot(grid[0, 0])
#     plot_spot_instability_map(instability_ax, data)
#     instability_ax.set_title('')
#     style_rounded_spatial_frame(instability_ax, linewidth=CONTEXT_FRAME_LINEWIDTH, linestyle='-', rounding=0.025)
#     add_context_bottom_title(instability_ax, 'Instability\nMap')
#     instability_legend_ax = fig.add_subplot(grid[0, 1])
#     add_instability_legend_5x2(instability_legend_ax)
#     example_ax = fig.add_subplot(grid[0, 2])
#     plot_context_example_run(example_ax, data)
#     example_ax.set_title('')
#     style_rounded_spatial_frame(example_ax, linewidth=CONTEXT_FRAME_LINEWIDTH, linestyle=(0, (3, 2)), rounding=0.025)
#     add_context_bottom_title(example_ax, 'Example\nRun')
#     workflow_top_ax = fig.add_subplot(grid[0, 3])
#     add_context_workflow_box(workflow_top_ax, 'Bin cluster in example run\nby mean instability score\n\nFind best match in ref.\nfor accuracy metrics.\n\nFind best match in\nother runs for DGEA')
#     reference_ax = fig.add_subplot(grid[1, 0])
#     plot_reference_annotation(reference_ax, data)
#     reference_ax.set_title('')
#     style_rounded_spatial_frame(reference_ax, linewidth=CONTEXT_FRAME_LINEWIDTH, linestyle='-', rounding=0.025)
#     add_context_bottom_title(reference_ax, 'Reference\nAnnotation')
#     reference_legend_ax = fig.add_subplot(grid[1, 1])
#     add_reference_legend(reference_legend_ax, data)
#     binned_grid = grid[1, 2].subgridspec(nrows=2, ncols=2, height_ratios=[1.0, 0.18], width_ratios=[1.0, 1.0], wspace=0.18, hspace=0.0)
#     binned_ax_1 = fig.add_subplot(binned_grid[0, 0])
#     plot_context_binned_clusters(binned_ax_1, data, CONTEXT_BINNED_CLUSTER_BINS[0])
#     binned_ax_2 = fig.add_subplot(binned_grid[0, 1])
#     plot_context_binned_clusters(binned_ax_2, data, CONTEXT_BINNED_CLUSTER_BINS[1])
#     binned_title_ax = fig.add_subplot(binned_grid[1, :])
#     binned_title_ax.set_axis_off()
#     binned_title_ax.text(0.5, 0.0, 'Binned Clusters', transform=binned_title_ax.transAxes, ha='center', va='bottom', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold')
#     workflow_bottom_ax = fig.add_subplot(grid[1, 3])
#     add_context_workflow_box(workflow_bottom_ax, 'Perform DGEA for all\ncluster pairs in example\nrun in same instability bin\n\nRepeat for matched pairs\nin other runs & compare\nresults to example run')
#     return {'instability_ax': instability_ax, 'instability_legend_ax': instability_legend_ax, 'example_ax': example_ax, 'reference_ax': reference_ax, 'reference_legend_ax': reference_legend_ax, 'binned_axes': [binned_ax_1, binned_ax_2], 'binned_title_ax': binned_title_ax, 'workflow_top_ax': workflow_top_ax, 'workflow_bottom_ax': workflow_bottom_ax}

def plot_fig4_context(fig, parent_spec, data):

    wrapper = parent_spec.subgridspec(
        nrows=3, ncols=3,
        height_ratios=[0.04, 0.92, 0.04],
        width_ratios=[0.03, 0.94, 0.03],
        hspace=0.0, wspace=0.0
    )

    # More horizontal room overall.
    # Top text box gets a wide region.
    # Bottom text box gets a narrower right-side region.
    grid = wrapper[1, 1].subgridspec(
        nrows=2, ncols=5,
        width_ratios=[1.0, 1.35, 1.45, 1.45, 1.35],
        height_ratios=[0.82, 1.18],
        wspace=0.18, hspace=0.28
    )

    # ========================================================
    # TOP ROW
    # ========================================================

    instability_ax = fig.add_subplot(grid[0, 0])
    plot_spot_instability_map(instability_ax, data)
    instability_ax.set_title('')
    style_rounded_spatial_frame(
        instability_ax,
        linewidth=CONTEXT_FRAME_LINEWIDTH,
        linestyle='-',
        rounding=0.025
    )
    add_context_bottom_title(instability_ax, 'Instability\nMap')

    instability_legend_ax = fig.add_subplot(grid[0, 1])
    add_instability_legend_5x2(instability_legend_ax)

    example_ax = fig.add_subplot(grid[0, 2])
    plot_context_example_run(example_ax, data)
    example_ax.set_title('')
    style_rounded_spatial_frame(
        example_ax,
        linewidth=CONTEXT_FRAME_LINEWIDTH,
        linestyle=(0, (3, 2)),
        rounding=0.025
    )
    add_context_bottom_title(example_ax, 'Example\nRun')

    # # WIDER + SHORTER top text box:
    # # spans the final two columns
    # workflow_top_ax = fig.add_subplot(grid[0, 3:5])
    # add_context_workflow_box(
    #     workflow_top_ax,
    #     'Bin cluster in example run by mean instability score\n'
    #     'Find best match in ref. for accuracy metrics\n'
    #     'Find best match in other runs for DGEA'
    # )
    # Narrower top workflow box, like the original
    top_box_grid = grid[0, 3:5].subgridspec(
        nrows=1, ncols=2,
        width_ratios=[0.45, 0.55],
        wspace=0.0
    )

    workflow_top_ax = fig.add_subplot(top_box_grid[0, 1])

    add_context_workflow_box(
        workflow_top_ax,
        'Bin cluster in example run\n'
        'by mean instability score\n\n'
        'Find best match in ref.\n'
        'for accuracy metrics.\n\n'
        'Find best match in\n'
        'other runs for DGEA'
    )


    # ========================================================
    # BOTTOM ROW
    # ========================================================

    reference_ax = fig.add_subplot(grid[1, 0])
    plot_reference_annotation(reference_ax, data)
    reference_ax.set_title('')
    style_rounded_spatial_frame(
        reference_ax,
        linewidth=CONTEXT_FRAME_LINEWIDTH,
        linestyle='-',
        rounding=0.025
    )
    add_context_bottom_title(reference_ax, 'Reference\nAnnotation')

    reference_legend_ax = fig.add_subplot(grid[1, 1])
    add_reference_legend(reference_legend_ax, data)

    # Three cluster-instability-bin plots
    binned_grid = grid[1, 2:4].subgridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 0.18],
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.14, hspace=0.0
    )

    binned_ax_1 = fig.add_subplot(binned_grid[0, 0])
    plot_context_binned_clusters(
        binned_ax_1, data, CONTEXT_BINNED_CLUSTER_BINS[0]
    )

    binned_ax_2 = fig.add_subplot(binned_grid[0, 1])
    plot_context_binned_clusters(
        binned_ax_2, data, CONTEXT_BINNED_CLUSTER_BINS[1]
    )

    binned_ax_3 = fig.add_subplot(binned_grid[0, 2])
    plot_context_binned_clusters(
        binned_ax_3, data, CONTEXT_BINNED_CLUSTER_BINS[2]
    )

    binned_title_ax = fig.add_subplot(binned_grid[1, :])
    binned_title_ax.set_axis_off()
    binned_title_ax.text(
        0.5, 0.0,
        'Binned Clusters',
        transform=binned_title_ax.transAxes,
        ha='center', va='bottom',
        fontsize=CONTEXT_TITLE_FONTSIZE,
        fontweight='bold'
    )

    # TALLER + NARROWER bottom text box:
    # only the final column
    workflow_bottom_ax = fig.add_subplot(grid[1, 4])
    add_context_workflow_box(
        workflow_bottom_ax,
        'Perform DGEA for all\n'
        'cluster pairs in example\n'
        'run in same instability bin\n\n'
        'Repeat for matched pairs\n'
        'in other runs & compare\n'
        'results to example run'
    )

    return {
        'instability_ax': instability_ax,
        'instability_legend_ax': instability_legend_ax,
        'example_ax': example_ax,
        'reference_ax': reference_ax,
        'reference_legend_ax': reference_legend_ax,
        'binned_axes': [
            binned_ax_1,
            binned_ax_2,
            binned_ax_3,
        ],
        'binned_title_ax': binned_title_ax,
        'workflow_top_ax': workflow_top_ax,
        'workflow_bottom_ax': workflow_bottom_ax,
    }
# ============================================================================
# PANEL B: CROSS-RUN CONSISTENCY
# ============================================================================
def fig4b_result_path(dataset, sample, method):
    """Alignment-summary CSV used by Figure 4B."""
    sample_part = f'_{safe_name(sample)}' if sample is not None else ''
    return ALIGNMENT_DIR / f'{safe_name(dataset)}{sample_part}_{safe_name(method)}_best_cluster_matches.csv'

def load_fig4b_method_matches(method):
    """
    Pool all usable non-example-run cluster matches for one method.

    The example run itself is excluded because its self-match has
    Jaccard = 1 and would artificially inflate cross-run consistency.
    """
    parts = []
    for dataset, sample in FIG4B_DATASET_SAMPLES:
        path = fig4b_result_path(dataset, sample, method)
        if not path.exists():
            print(f'[Figure 4B missing] {path.name}')
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f'[Figure 4B empty] {path.name}')
            continue
        required = {'example_cluster', 'run', 'example_run_idx', 'best_jaccard', 'overlap', 'example_cluster_size', 'run_cluster_size', 'example_mean_instability', 'example_fine_bin'}
        missing = required - set(df.columns)
        if missing:
            print(f'[Figure 4B invalid] {path.name}: missing {sorted(missing)}')
            continue
        for column in ['run', 'example_run_idx', 'best_jaccard', 'example_mean_instability']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        # Exclude the example run so its self-match (Jaccard=1) does not inflate consistency.
        df = df.loc[df['run'].ne(df['example_run_idx']) & df['best_jaccard'].notna() & df['example_mean_instability'].notna() & df['example_fine_bin'].astype(str).isin(INSTABILITY_LABELS)].copy()
        if df.empty:
            print(f'[Figure 4B no non-example matches] {path.name}')
            continue
        df['source_dataset'] = dataset
        df['source_sample'] = '' if sample is None else str(sample)
        df['source_method'] = method
        df['example_fine_bin'] = df['example_fine_bin'].astype(str)
        parts.append(df)
        print(f'[Figure 4B loaded] {path.name}: {len(df)} non-example matches')
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

def summarize_fig4b_clusters(matches):
    """Collapse Figure 4B to one row per example cluster."""
    id_columns = ['source_dataset', 'source_sample', 'source_method', 'example_cluster', 'example_mean_instability', 'example_fine_bin', 'example_cluster_size']
    return matches.groupby(id_columns, dropna=False, as_index=False).agg(n_nonexample_runs=('run', 'nunique'), best_jaccard_mean=('best_jaccard', 'mean'), best_jaccard_median=('best_jaccard', 'median'), best_jaccard_sd=('best_jaccard', 'std'), best_jaccard_min=('best_jaccard', 'min'), best_jaccard_q25=('best_jaccard', lambda x: x.quantile(0.25)), best_jaccard_q75=('best_jaccard', lambda x: x.quantile(0.75)), best_jaccard_max=('best_jaccard', 'max'), overlap_mean=('overlap', 'mean'), matched_run_cluster_size_mean=('run_cluster_size', 'mean'))

def fig4b_correlation_results(cluster_summary):
    valid = cluster_summary[['example_mean_instability', 'best_jaccard_mean']].dropna()
    if len(valid) < 3:
        return (np.nan, np.nan, np.nan, np.nan, len(valid))
    x = valid['example_mean_instability']
    y = valid['best_jaccard_mean']
    spearman = spearmanr(x, y)
    pearson = pearsonr(x, y)
    return (float(spearman.statistic), float(spearman.pvalue), float(pearson.statistic), float(pearson.pvalue), len(valid))

def style_fig4b_boxplot(boxplot, colors):
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor('black')
        box.set_linewidth(1.1)
    for key in ['whiskers', 'caps']:
        for line in boxplot[key]:
            line.set_color('black')
            line.set_linewidth(1.1)
    for median in boxplot['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

def plot_fig4b(fig, parent_spec, matches, cluster_summary):
    """
    Left:
        one point per example cluster, colored by instability bin;
        y is the mean best-match Jaccard over non-example runs.

    Right:
        distribution of all non-example-run best-match Jaccards
        within each instability bin.

    The second line under the left subplot reports the number of
    example clusters in each populated instability bin, in the same
    left-to-right/color order used by the dots and the distribution.
    """
    populated_bins = [fine_bin for fine_bin in INSTABILITY_LABELS if matches.loc[matches['example_fine_bin'].eq(fine_bin), 'best_jaccard'].notna().any()]
    if not populated_bins:
        raise ValueError('No populated instability bins are available for Figure 4B.')
    wrapper = parent_spec.subgridspec(nrows=2, ncols=1, height_ratios=[0.18, 0.82], hspace=0.0)
    grid = wrapper[1, 0].subgridspec(nrows=2, ncols=2, height_ratios=FIG4B_HEIGHT_RATIOS, width_ratios=[1.0, 1.0], hspace=0.0, wspace=FIG4B_INNER_WSPACE)
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    box_data = [matches.loc[matches['example_fine_bin'].eq(fine_bin), 'best_jaccard'].dropna().to_numpy(float) for fine_bin in populated_bins]
    positions = np.arange(1, len(populated_bins) + 1, dtype=float)
    box_positions = positions - 0.14
    violin_positions = positions + 0.1
    colors = [BIN_COLORS[bin_label] for bin_label in populated_bins]
    boxplot = ax_box.boxplot(box_data, positions=box_positions, widths=0.28, patch_artist=True, showfliers=False, whis=1.5, manage_ticks=False)
    style_fig4b_boxplot(boxplot, colors)
    for values, color, violin_position in zip(box_data, colors, violin_positions):
        if len(values) >= 2 and np.ptp(values) > 0:
            violin = ax_box.violinplot([values], positions=[violin_position], widths=0.55, showmeans=False, showmedians=False, showextrema=False, bw_method='scott')
            body = violin['bodies'][0]
            vertices = body.get_paths()[0].vertices
            # Keep only the right half of each violin.
            vertices[:, 0] = np.maximum(vertices[:, 0], violin_position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(1.1)
            body.set_alpha(0.6)
            body.set_zorder(1)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(populated_bins, rotation=45, ha='right', rotation_mode='anchor')
    ax_box.set_xlim(0.45, len(populated_bins) + 0.65)
    ax_box.set_ylim(0, 1)
    ax_box.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_box.set_yticklabels([])
    ax_box.tick_params(axis='both', labelsize=FIG4B_TICK_FONTSIZE)
    ax_box.set_title('(distribution)', fontweight='bold', fontsize=FIG4B_TITLE_FONTSIZE, pad=6)
    ax_box.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.4)
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    cluster_counts = []
    rng = np.random.default_rng(FIG4B_RANDOM_SEED)
    for fine_bin in populated_bins:
        sub = cluster_summary[cluster_summary['example_fine_bin'].eq(fine_bin)]
        cluster_counts.append(int(len(sub)))
        if sub.empty:
            continue
        x_values = sub['example_mean_instability'].to_numpy(float)
        y_values = sub['best_jaccard_mean'].to_numpy(float)
        jitter = rng.normal(0, 0.0025, size=len(sub))
        ax_scatter.scatter(x_values + jitter, y_values, s=FIG4B_SCATTER_SIZE, color=BIN_COLORS[fine_bin], edgecolor='black', linewidth=0.35, alpha=0.85)
    fit_data = cluster_summary[['example_mean_instability', 'best_jaccard_mean']].dropna()
    if len(fit_data) >= 2 and fit_data['example_mean_instability'].nunique() >= 2:
        x_fit = fit_data['example_mean_instability'].to_numpy(float)
        y_fit = fit_data['best_jaccard_mean'].to_numpy(float)
        slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.75, zorder=4)
    rho, spearman_p, pearson_r, pearson_p, n_clusters = fig4b_correlation_results(cluster_summary)
    print(f'[Figure 4B] Spearman rho={rho:.3f}, p={spearman_p:.3g}; Pearson r={pearson_r:.3f}, p={pearson_p:.3g}; n={n_clusters} clusters')
    ax_scatter.set_xlim(0, 0.7)
    ax_scatter.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_scatter.tick_params(axis='both', labelsize=FIG4B_TICK_FONTSIZE)
    n_text = ','.join((str(value) for value in cluster_counts))
    ax_scatter.set_xlabel(f'Cluster Instability - Example Run\n($N$ = {n_text})', fontsize=FIG4B_AXIS_LABEL_FONTSIZE, linespacing=1.05, labelpad=4)
    ax_scatter.set_ylabel('Jaccard w.r.t. Match', fontsize=FIG4B_AXIS_LABEL_FONTSIZE)
    ax_scatter.set_title('Cross-run Consistency (mean)', fontweight='bold', fontsize=FIG4B_TITLE_FONTSIZE, pad=6)
    ax_scatter.grid(linestyle=':', linewidth=0.6, alpha=0.4)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    stats = {'method': FIG4_METHOD, 'n_example_clusters': n_clusters, 'spearman_rho': rho, 'spearman_pvalue': spearman_p, 'pearson_r': pearson_r, 'pearson_pvalue': pearson_p, 'populated_bins': populated_bins, 'cluster_counts': cluster_counts}
    return (np.array([ax_scatter, ax_box], dtype=object), stats)

def save_fig4b_tables(method, matches, cluster_summary, stats):
    """Preserve the tabular outputs from the standalone Figure 4B script."""
    matches.to_csv(OUT_DIR / f'{safe_name(method)}_pooled_best_matches.csv', index=False)
    cluster_summary.to_csv(OUT_DIR / f'{safe_name(method)}_cluster_jaccard_summary.csv', index=False)
    correlation = pd.DataFrame([{'method': method, 'n_example_clusters': stats['n_example_clusters'], 'spearman_rho': stats['spearman_rho'], 'spearman_pvalue': stats['spearman_pvalue'], 'pearson_r': stats['pearson_r'], 'pearson_pvalue': stats['pearson_pvalue']}])
    correlation.to_csv(OUT_DIR / f'{safe_name(method)}_instability_jaccard_correlation.csv', index=False)


# ============================================================================
# PANEL C: DGEA CONSISTENCY
# ============================================================================
def fig4c_result_path(dataset, sample, method, suffix):
    """Return one Figure 4C DGEA/alignment CSV path."""
    sample_part = f'_{safe_name(sample)}' if sample is not None else ''
    return ALIGNMENT_DIR / f'{safe_name(dataset)}{sample_part}_{safe_name(method)}_{suffix}'

def fig4c_topk(df, k):
    """Keep the top-k unique genes by rank."""
    return df.sort_values('rank').head(int(k)).drop_duplicates('gene').copy()

def fig4c_shared_rank_metrics(dgea_long):
    """
    Recalculate the shared-gene absolute-rank-difference metric.

    For each contrast and Top-K value:
      1. compare every non-example run with the example run;
      2. retain genes present in both Top-K lists;
      3. calculate the mean absolute rank difference within that run;
      4. average those run-level means across non-example runs.
    """
    rows = []
    for comparison, sub in dgea_long.groupby('comparison', sort=False):
        example_run = int(sub['example_run_idx'].iloc[0])
        runs = sorted(sub['run'].astype(int).unique())
        if example_run not in runs:
            continue
        by_run = {run: sub.loc[sub['run'].eq(run)] for run in runs}
        for top_k in FIG4C_TOP_KS:
            example = fig4c_topk(by_run[example_run], top_k).set_index('gene')['rank']
            rank_diffs = []
            n_shared = []
            n_example_missing = []
            n_run_missing = []
            for run in runs:
                if run == example_run:
                    continue
                current = fig4c_topk(by_run[run], top_k).set_index('gene')['rank']
                shared = example.index.intersection(current.index)
                n_example_missing.append(len(example.index.difference(current.index)))
                n_run_missing.append(len(current.index.difference(example.index)))
                n_shared.append(len(shared))
                if len(shared):
                    rank_diffs.append(float(np.mean(np.abs(example.loc[shared].to_numpy(float) - current.loc[shared].to_numpy(float)))))
            rows.append({'comparison': comparison, 'top_k': int(top_k), 'rank_abs_diff_shared_mean': float(np.mean(rank_diffs)) if rank_diffs else np.nan, 'n_shared_genes_mean': float(np.mean(n_shared)) if n_shared else np.nan, 'n_example_genes_missing_in_run_mean': float(np.mean(n_example_missing)) if n_example_missing else np.nan, 'n_run_genes_missing_in_example_mean': float(np.mean(n_run_missing)) if n_run_missing else np.nan, 'n_missing_genes_mean': float(np.mean(np.asarray(n_example_missing) + np.asarray(n_run_missing))) if n_example_missing else np.nan})
    return pd.DataFrame(rows)

def load_fig4c_method(method):
    """
    Pool Figure 4C DGEA-consistency data across biological samples.

    This retains the standalone Figure 4C semantics:
      - summary CSV supplies Jaccard consistency;
      - dgea_long_top200.csv is used to recompute the shared-gene
        absolute rank difference;
      - only contrasts whose two tested clusters are in the same
        fine instability bin are retained.
    """
    pooled = []
    summary_required = {'comparison', 'top_k', 'group_a_fine_bin', 'group_b_fine_bin', 'jaccard_mean'}
    long_required = {'comparison', 'example_run_idx', 'run', 'gene', 'rank'}
    for dataset, sample in FIG4C_DATASET_SAMPLES:
        summary_path = fig4c_result_path(dataset, sample, method, 'contrast_dgea_consistency.csv')
        long_path = fig4c_result_path(dataset, sample, method, 'dgea_long_top200.csv')
        if not summary_path.exists() or not long_path.exists():
            print(f'[Figure 4C missing] {summary_path.name} or {long_path.name}')
            continue
        try:
            df = pd.read_csv(summary_path)
            dgea_long = pd.read_csv(long_path)
        except pd.errors.EmptyDataError:
            print(f'[Figure 4C empty] {summary_path.name} or {long_path.name}')
            continue
        missing = summary_required - set(df.columns)
        missing_long = long_required - set(dgea_long.columns)
        if missing or missing_long:
            print(f'[Figure 4C invalid] {summary_path.name}: summary missing {sorted(missing)}, DGEA missing {sorted(missing_long)}')
            continue
        dgea_long['run'] = pd.to_numeric(dgea_long['run'], errors='coerce')
        dgea_long['rank'] = pd.to_numeric(dgea_long['rank'], errors='coerce')
        dgea_long = dgea_long.dropna(subset=['run', 'rank', 'gene']).copy()
        dgea_long['run'] = dgea_long['run'].astype(int)
        dgea_long['gene'] = dgea_long['gene'].astype(str)
        new_rank_metrics = fig4c_shared_rank_metrics(dgea_long)
        df = df.drop(columns=['rank_abs_diff_shared_mean', 'n_shared_genes_mean', 'n_missing_genes_mean', 'n_example_genes_missing_in_run_mean', 'n_run_genes_missing_in_example_mean'], errors='ignore').merge(new_rank_metrics, on=['comparison', 'top_k'], how='left')
        # Figure 4C only compares contrasts whose two clusters share an instability bin.
        a_bin = df['group_a_fine_bin'].astype(str)
        b_bin = df['group_b_fine_bin'].astype(str)
        df = df.loc[a_bin.eq(b_bin) & a_bin.isin(INSTABILITY_LABELS)].copy()
        if df.empty:
            print(f'[Figure 4C no usable contrasts] {summary_path.name}')
            continue
        df['contrast_fine_bin'] = df['group_a_fine_bin'].astype(str)
        df['source_dataset'] = dataset
        df['source_sample'] = '' if sample is None else str(sample)
        df['source_file'] = summary_path.name
        pooled.append(df)
        print(f'[Figure 4C loaded] {summary_path.name}: {len(df)} rows')
    if not pooled:
        return pd.DataFrame()
    pooled = pd.concat(pooled, ignore_index=True)
    for column in ['top_k', 'jaccard_mean', 'rank_abs_diff_shared_mean', 'n_shared_genes_mean', 'n_missing_genes_mean', 'n_example_genes_missing_in_run_mean', 'n_run_genes_missing_in_example_mean']:
        pooled[column] = pd.to_numeric(pooled[column], errors='coerce')
    return pooled.loc[pooled['top_k'].isin(FIG4C_TOP_KS) & pooled['contrast_fine_bin'].isin(INSTABILITY_LABELS)].copy()

def fig4c_finite_values(df, fine_bin, column):
    values = df.loc[df['contrast_fine_bin'].eq(fine_bin), column].to_numpy(float)
    return values[np.isfinite(values)]

def fig4c_common_bins(pooled, top_k):
    """
    Return bins with finite data for BOTH stacked metrics.

    Using one common bin list is what makes the top Jaccard axis and
    bottom absolute-rank-difference axis share x coordinates exactly.
    """
    sub = pooled.loc[pooled['top_k'].eq(top_k)]
    return [fine_bin for fine_bin in INSTABILITY_LABELS if fig4c_finite_values(sub, fine_bin, FIG4C_METRICS['jaccard']['column']).size and fig4c_finite_values(sub, fine_bin, FIG4C_METRICS['abs_rank_diff']['column']).size]

def style_fig4c_boxplot(boxplot, colors):
    for box, color in zip(boxplot['boxes'], colors):
        box.set(facecolor=color, alpha=0.82, edgecolor='black', linewidth=1.1)
    for key in ['whiskers', 'caps']:
        for line in boxplot[key]:
            line.set(color='black', linewidth=1.1)
    for median in boxplot['medians']:
        median.set(color='black', linewidth=1.5)
    for flier in boxplot['fliers']:
        flier.set(color='black', markersize=4)

def draw_fig4c_panel(ax, pooled, top_k, metric_name, bins, show_ylabel, show_title, show_x):
    """Draw one of the eight visible Figure 4C boxplots."""
    metric = FIG4C_METRICS[metric_name]
    column = metric['column']
    sub = pooled.loc[pooled['top_k'].eq(top_k)]
    if not bins:
        ax.axis('off')
        ax.text(0.5, 0.5, f'No Top-{top_k} data', ha='center', va='center', fontsize=FIG4C_AXIS_LABEL_FONTSIZE)
        return []
    positions = np.arange(1, len(bins) + 1, dtype=float)
    box_data = [fig4c_finite_values(sub, fine_bin, column) for fine_bin in bins]
    colors = [BIN_COLORS[fine_bin] for fine_bin in bins]
    boxplot = ax.boxplot(box_data, positions=positions, widths=0.64, patch_artist=True, showfliers=True, manage_ticks=False, whis=1.5)
    style_fig4c_boxplot(boxplot, colors)
    ax.set_xlim(0.25, len(bins) + 0.75)
    if metric_name == 'jaccard':
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if not show_ylabel:
            ax.set_yticklabels([])
    elif top_k == 10:
        ax.set_ylim(0, 4)
        ax.set_yticks([0, 1, 2, 3, 4])
    elif top_k == 20:
        ax.set_ylim(0, 8)
        ax.set_yticks([0, 2, 4, 6, 8])
    elif top_k == 50:
        ax.set_ylim(0, 15)
        ax.set_yticks([0, 3, 6, 9, 12, 15])
    else:
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 15, 30, 45, 60])
    if show_x:
        ax.set_xticks(positions)
        ax.set_xticklabels(bins, rotation=45, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', labelsize=FIG4C_TICK_FONTSIZE)
    else:
        ax.set_xticks(positions)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=FIG4C_TICK_FONTSIZE)
    if show_title:
        ax.set_title(f'Top-{top_k} DEGs', fontsize=FIG4C_TITLE_FONTSIZE, fontweight='bold', pad=4)
    if show_ylabel:
        ax.set_ylabel(metric['ylabel'], fontsize=FIG4C_AXIS_LABEL_FONTSIZE)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return [len(values) for values in box_data]

def fig4c_shared_xlabel_text(pooled):
    """
    Build the one shared x-label beneath all four lower subplots.

    The standalone script used the N values left over from the final
    Top-K loop iteration, i.e. Top-200. Preserve that behavior here.
    """
    # Preserve standalone behavior: shared N values come from the final Top-K (Top-200).
    top_k = FIG4C_TOP_KS[-1]
    bins = fig4c_common_bins(pooled, top_k)
    sub = pooled.loc[pooled['top_k'].eq(top_k)]
    counts = [len(fig4c_finite_values(sub, fine_bin, FIG4C_METRICS['jaccard']['column'])) for fine_bin in bins]
    n_text = ','.join((str(count) for count in counts))
    return 'Instability Bins of Tested Clusters (' + '$N=$' + n_text + ')'

def plot_fig4c(fig, parent_spec, pooled):
    """
    Layout:

        Top-10     Top-20     Top-50     Top-200
        Jaccard    Jaccard    Jaccard    Jaccard

        Top-10     Top-20     Top-50     Top-200
        abs rank   abs rank   abs rank   abs rank

    Only the first four Top-K values are used. Each top Jaccard axis
    shares x with the absolute-rank-difference axis immediately below it.
    The top row has no x ticks or x labels. A single shared x-label is
    placed beneath the complete lower row.
    """
    wrapper = parent_spec.subgridspec(nrows=2, ncols=1, height_ratios=[0.1, 0.9], hspace=0.0)
    grid = wrapper[1, 0].subgridspec(nrows=5, ncols=len(FIG4C_TOP_KS), height_ratios=FIG4C_HEIGHT_RATIOS, hspace=FIG4C_INNER_HSPACE, wspace=FIG4C_INNER_WSPACE)
    top_axes = []
    bottom_axes = []
    for index, top_k in enumerate(FIG4C_TOP_KS):
        bins = fig4c_common_bins(pooled, top_k)
        bottom_ax = fig.add_subplot(grid[2, index])
        top_ax = fig.add_subplot(grid[0, index], sharex=bottom_ax)
        draw_fig4c_panel(ax=bottom_ax, pooled=pooled, top_k=top_k, metric_name='abs_rank_diff', bins=bins, show_ylabel=index == 0, show_title=False, show_x=True)
        draw_fig4c_panel(ax=top_ax, pooled=pooled, top_k=top_k, metric_name='jaccard', bins=bins, show_ylabel=index == 0, show_title=True, show_x=False)
        top_axes.append(top_ax)
        bottom_axes.append(bottom_ax)
    top_axes[0].yaxis.set_label_coords(-0.25, 0.5)
    bottom_axes[0].yaxis.set_label_coords(-0.25, 0.5)
    label_ax = fig.add_subplot(grid[4, :])
    label_ax.set_axis_off()
    shared_xlabel = fig4c_shared_xlabel_text(pooled)
    label_ax.text(0.5, 0.18, shared_xlabel, transform=label_ax.transAxes, ha='center', va='center', fontsize=FIG4C_SHARED_XLABEL_FONTSIZE)
    return (np.asarray([top_axes, bottom_axes], dtype=object), label_ax, shared_xlabel)

def save_fig4c_table(method, pooled):
    """Preserve the pooled Figure 4C data table from the standalone script."""
    pooled.to_csv(OUT_DIR / f'{safe_name(method)}_dgea_consistency_jaccard_data.csv', index=False)


# ============================================================================
# PANEL D: INSTABILITY VS REFERENCE QUALITY
# ============================================================================
def load_fig4d_clusterwise_metrics(method):
    """
    Load the 50-run clusterwise metrics used by Figure 4D.

    This follows the same data population and dataset/color semantics as
    the cluster-level scatter panel in Figure 5:
      - 50 runs only;
      - one method (FIG4_METHOD);
      - exclude obsolete/full-size dataset tags;
      - retain DLPFC samples plus the six singleton datasets shown here.
    """
    if not FIG4D_METRICS_CSV.exists():
        raise FileNotFoundError(f'Figure 4D clusterwise metrics file not found:\n{FIG4D_METRICS_CSV}')
    df = pd.read_csv(FIG4D_METRICS_CSV)
    required = {'dataset-sample', 'method', 'iterations', FIG4D_XCOL, *[metric_column for metric_column, _ in FIG4D_METRICS]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Figure 4D metrics table is missing columns: {sorted(missing)}')
    excluded_tags = {'ov_ffpe_full', 'coad_ffpe_full', 'Visium_HD_Human_Colon_Cancer'}
    df = df.loc[~df['dataset-sample'].isin(excluded_tags) & df['iterations'].eq(FIG4D_ITERATIONS) & df['method'].eq(method)].copy()
    sample_tags = df['dataset-sample'].astype(str)
    singleton_tags = {'mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe'}
    supported = sample_tags.str.startswith('dlpfc_', na=False) | sample_tags.isin(singleton_tags)
    df = df.loc[supported].copy()
    if df.empty:
        raise ValueError(f'No usable Figure 4D data for method {method}.')
    for column in [FIG4D_XCOL, *[metric_column for metric_column, _ in FIG4D_METRICS]]:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def fig4d_dataset_mask(df, dataset):
    """Return the rows belonging to one displayed biological dataset."""
    tags = df['dataset-sample'].astype(str)
    if dataset == 'dlpfc':
        return tags.str.startswith('dlpfc_', na=False)
    return tags.eq(dataset)

def draw_fig4d_scatter(ax, stats_ax, df, metric_column, title, show_ylabel):
    """Draw one of the three Figure 4D quality-vs-instability scatterplots."""
    x_all = df[FIG4D_XCOL].to_numpy(float)
    y_all = df[metric_column].to_numpy(float)
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    xx = x_all[valid]
    yy = y_all[valid]
    if len(xx) < 2:
        raise ValueError(f"Figure 4D metric '{metric_column}' has fewer than two finite points.")
    if np.ptp(xx) <= 0 or np.ptp(yy) <= 0:
        raise ValueError(f"Figure 4D metric '{metric_column}' does not vary enough for Pearson correlation/regression.")
    # Pearson correlation and least-squares trend use the same finite points.
    r, _ = pearsonr(xx, yy)
    r_sq = r ** 2
    slope, intercept = np.polyfit(xx, yy, 1)
    fit_x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, color='black', linestyle='--', linewidth=2, zorder=4)
    print(f'[Figure 4D] {title}: r={r:.6f}, r2={r_sq:.6f}')
    datasets_to_plot = ['dlpfc', 'mouse_brain_cerebellum', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'human_breast_cancer', 'mouse_brain', 'coad_ffpe', 'ov_ffpe']
    for dataset in datasets_to_plot:
        zdf = df.loc[fig4d_dataset_mask(df, dataset)]
        zx = zdf[FIG4D_XCOL].to_numpy(float)
        zy = zdf[metric_column].to_numpy(float)
        point_valid = np.isfinite(zx) & np.isfinite(zy)
        zx = zx[point_valid]
        zy = zy[point_valid]
        if zx.size == 0:
            continue
        color_index = FIG4D_DATASET_COLOR_INDEX[dataset]
        alpha = FIG4D_DLPFC_ALPHA if dataset == 'dlpfc' else FIG4D_NON_DLPFC_ALPHA
        ax.scatter(zx, zy, color=TABLEAU_20[color_index], s=FIG4D_SCATTER_SIZE, alpha=alpha, edgecolors='none', zorder=3)
    ax.set_title(title, fontsize=FIG4D_TITLE_FONTSIZE, fontweight='bold', pad=4)
    ax.set_xlabel('Cluster Instability', fontsize=FIG4D_AXIS_LABEL_FONTSIZE, labelpad=1)
    if show_ylabel:
        ax.set_ylabel('Quality w.r.t. Ref.', fontsize=FIG4D_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0, 0.7)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize=FIG4D_TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    stats_ax.set_axis_off()
    stats_ax.text(0.5, 0.82, f'$r = {r:.2f}$\n$r^2 = {r_sq:.2f}$', transform=stats_ax.transAxes, ha='center', va='top', fontsize=FIG4D_STATS_FONTSIZE, linespacing=1.0, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
    return {'metric': metric_column, 'title': title, 'pearson_r': float(r), 'r_squared': float(r_sq), 'n_clusters': int(len(xx))}

def add_fig4d_sample_legend(ax):
    """
    Put the Figure 4D sample legend in the fourth subplot slot.

    The visible arrangement matches the sketch: two columns and four rows,
    with the first column MB/MBC/HBC/DLPFC and the second CRC/OV/COAD.
    Marker area matches the Figure 4D scatter-point area.
    """
    ax.set_axis_off()
    labels = ['MB', 'MBC', 'HBC', 'DLPFC', 'CRC', 'OV', 'COAD']
    datasets = ['mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'dlpfc', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe']
    colors = [TABLEAU_20[FIG4D_DATASET_COLOR_INDEX[dataset]] for dataset in datasets]
    legend_marker_size = np.sqrt(140)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=legend_marker_size, markerfacecolor=color, markeredgecolor='none', label=label) for label, color in zip(labels, colors)]
    ax.text(0.03, 0.93, 'Sample', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left', va='top')
    legend = ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.0, 0.83), bbox_transform=ax.transAxes, ncol=2, frameon=False, fontsize=16, handlelength=1.4, handletextpad=0.5, columnspacing=1, labelspacing=0.4, borderaxespad=0.0)
    return legend

def plot_fig4d(fig, parent_spec, df):
    """
    Layout:

        Jaccard | Recall | Purity | Sample legend
        ------------------------------------------
        r/r^2   | r/r^2  | r/r^2  | blank

    Only the first three metric subplots from the standalone code are used.
    """
    grid = parent_spec.subgridspec(nrows=3, ncols=4, height_ratios=FIG4D_HEIGHT_RATIOS, hspace=FIG4D_INNER_HSPACE, wspace=FIG4D_INNER_WSPACE)
    scatter_axes = []
    stats_axes = []
    metric_stats = []
    for index, (metric_column, title) in enumerate(FIG4D_METRICS):
        ax = fig.add_subplot(grid[0, index])
        stats_ax = fig.add_subplot(grid[2, index])
        stat = draw_fig4d_scatter(ax=ax, stats_ax=stats_ax, df=df, metric_column=metric_column, title=title, show_ylabel=index == 0)
        scatter_axes.append(ax)
        stats_axes.append(stats_ax)
        metric_stats.append(stat)
    legend_ax = fig.add_subplot(grid[0, 3])
    legend = add_fig4d_sample_legend(legend_ax)
    legend_lower_ax = fig.add_subplot(grid[2, 3])
    legend_lower_ax.set_axis_off()
    return {'scatter_axes': np.asarray(scatter_axes, dtype=object), 'stats_axes': np.asarray(stats_axes, dtype=object), 'legend_ax': legend_ax, 'legend_lower_ax': legend_lower_ax, 'legend': legend, 'metric_stats': metric_stats}

def make_blank_panel(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_xticks([])
    ax.set_yticks([])
    if SHOW_PANEL_GUIDES:
        ax.patch.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
            spine.set_linestyle('--')
    else:
        ax.set_axis_off()
    return ax

def add_panel_letter(fig, ax, letter, x_offset=0.03, y_offset=0.015):
    fig.canvas.draw()
    pos = ax.get_position()
    fig.text(pos.x0 - x_offset, pos.y1 + y_offset, letter, fontsize=28, fontweight='bold', ha='left', va='top')


# ============================================================================
# CREATE COMPLETE FIGURE 4
# ============================================================================
def create_figure4(data, fig4b_matches, fig4b_cluster_summary, fig4c_pooled, fig4d_df):
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(nrows=5, ncols=2, height_ratios=FIG4_HEIGHT_RATIOS, hspace=FIG4_OUTER_HSPACE, wspace=FIG4_OUTER_WSPACE, left=FIG4_LEFT, right=FIG4_RIGHT, top=FIG4_TOP, bottom=FIG4_BOTTOM)
    run_axes, row_first_axes, example_label_ax, collision_rows = plot_fig4a(fig, outer[0:2, :], data)
    context_axes = plot_fig4_context(fig, outer[2:4, 0], data)
    #match_axes_size_to_reference(fig, row_first_axes[0], [context_axes['instability_ax'], context_axes['reference_ax'], context_axes['example_ax'], context_axes['binned_axes'][0], context_axes['binned_axes'][1]], scale=1.0)
    match_axes_size_to_reference(fig, row_first_axes[0], [context_axes['instability_ax'], context_axes['reference_ax'], context_axes['example_ax'], *context_axes['binned_axes']], scale=1.0)
    # Shift the unlabeled-context Example Run plot slightly to the right
    EXAMPLE_CONTEXT_XSHIFT = 0.05

    pos = context_axes['example_ax'].get_position()
    context_axes['example_ax'].set_position([pos.x0 + EXAMPLE_CONTEXT_XSHIFT, pos.y0, pos.width, pos.height])

    # Move only the second context row upward within its allocated region.
    CONTEXT_BOTTOM_ROW_SHIFT = 0.022
    #bottom_context_axes = [context_axes['reference_ax'], context_axes['reference_legend_ax'], context_axes['binned_axes'][0], context_axes['binned_axes'][1], context_axes['binned_title_ax'], context_axes['workflow_bottom_ax']]
    bottom_context_axes = [context_axes['reference_ax'], context_axes['reference_legend_ax'], *context_axes['binned_axes'], context_axes['binned_title_ax'], context_axes['workflow_bottom_ax']]
    for ax in bottom_context_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + CONTEXT_BOTTOM_ROW_SHIFT, pos.width, pos.height])

    # Align the visible RIGHT edges of the two dashed workflow boxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_text = context_axes['workflow_top_ax'].texts[0]
    bottom_text = context_axes['workflow_bottom_ax'].texts[0]
    top_box = top_text.get_window_extent(renderer=renderer)
    bottom_box = bottom_text.get_window_extent(renderer=renderer)
    # Difference between visible right edges, converted pixels -> figure coordinates
    dx = (bottom_box.x1 - top_box.x1) / fig.bbox.width
    pos = context_axes['workflow_top_ax'].get_position()
    context_axes['workflow_top_ax'].set_position([pos.x0 + dx,pos.y0,pos.width,pos.height])

    b_axes, b_stats = plot_fig4b(fig, outer[2:3, 1], fig4b_matches, fig4b_cluster_summary)
    c_axes, c_label_ax, c_shared_xlabel = plot_fig4c(fig, outer[3:5, 1], fig4c_pooled)
    d_data = plot_fig4d(fig, outer[4:5, 0], fig4d_df)
    add_panel_letter(fig, row_first_axes[0], 'A', x_offset=0.035)
    add_panel_letter(fig, b_axes[0], 'B', x_offset=0.05, y_offset=0.037)
    add_panel_letter(fig, c_axes[0, 0], 'C', x_offset=0.05, y_offset=0.037)
    add_panel_letter(fig, d_data['scatter_axes'][0], 'D', x_offset=0.035, y_offset=0.037)
    return {'fig': fig, 'outer': outer, 'run_axes': run_axes, 'row_first_axes': row_first_axes, 'example_label_ax': example_label_ax, 'context_axes': context_axes, 'b_axes': b_axes, 'b_stats': b_stats, 'c_axes': c_axes, 'c_label_ax': c_label_ax, 'c_shared_xlabel': c_shared_xlabel, 'd_data': d_data, 'collision_rows': collision_rows}


# ============================================================================
# SAVE OUTPUTS
# ============================================================================
def save_figure4(figure_data):
    fig = figure_data['fig']
    collision_rows = figure_data['collision_rows']
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_STEM.with_suffix('.png')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', pad_inches=0.03, facecolor='white')
    print(f'[saved] {png_path}')
    pdf_path = OUT_STEM.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight', pad_inches=0.03, facecolor='white')
    print(f'[saved] {pdf_path}')
    if collision_rows:
        collision_df = pd.DataFrame(collision_rows).drop_duplicates().sort_values(['run', 'matched_run_cluster'])
        collision_path = OUT_STEM.with_name(OUT_STEM.name + '_collisions').with_suffix('.csv')
        collision_df.to_csv(collision_path, index=False)
        print(f'[saved] {collision_path}')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_figure4_data(dataset='human_breast_cancer', sample=None, method=FIG4_METHOD, selected_example_clusters=SELECTED_EXAMPLE_CLUSTERS)
    fig4b_matches = load_fig4b_method_matches(FIG4_METHOD)
    if fig4b_matches.empty:
        raise ValueError('No usable alignment data were available for Figure 4B.')
    fig4b_cluster_summary = summarize_fig4b_clusters(fig4b_matches)
    fig4c_pooled = load_fig4c_method(FIG4_METHOD)
    if fig4c_pooled.empty:
        raise ValueError('No usable DGEA-consistency data were available for Figure 4C.')
    fig4d_df = load_fig4d_clusterwise_metrics(FIG4_METHOD)
    figure_data = create_figure4(data, fig4b_matches, fig4b_cluster_summary, fig4c_pooled, fig4d_df)
    save_fig4b_tables(FIG4_METHOD, fig4b_matches, fig4b_cluster_summary, figure_data['b_stats'])
    save_fig4c_table(FIG4_METHOD, fig4c_pooled)
    save_figure4(figure_data)
if __name__ == '__main__':
    main()
