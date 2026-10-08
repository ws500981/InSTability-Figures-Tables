import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from scipy import sparse
from collections import defaultdict
import copy
from functions import load_data, qc_spot_and_gene_removal
from spot_removal import rm_free_lying_pts

import joblib
from pathlib import Path
import time

import geopandas as gpd
import spatialdata as sd
import spatialdata_plot  # registers sdata.pl
from scipy.spatial import cKDTree
from shapely import box as shapely_box
from spatialdata.models import Image2DModel, ShapesModel, TableModel


# SpatialData plotting settings. SCAF4093 is Visium HD, so spots are squares.
SPATIAL_SQUARE_SIZE = 1.0
SPATIAL_IMAGE_ALPHA = 1.0
_SPATIAL_CACHE = {}


def _to_grayscale_image(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image

    rgb = image[..., :3].astype(float)
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )

    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(gray, 0, np.iinfo(image.dtype).max)
    else:
        gray = np.clip(gray, 0.0, 1.0)

    return gray.astype(image.dtype)


def _get_image_coords_scale(adata_obj, preferred_key="hires"):
    coords = np.asarray(adata_obj.obsm["spatial"], dtype=float)
    spatial = adata_obj.uns.get("spatial", {})

    if not isinstance(spatial, dict) or not spatial:
        return None, coords, 1.0

    library = spatial[next(iter(spatial))]
    images = library.get("images", {})
    scales = library.get("scalefactors", {})

    keys = (
        ("hires", "lowres")
        if preferred_key == "hires"
        else ("lowres", "hires")
    )

    for key in keys:
        if key in images:
            scale = float(scales.get(f"tissue_{key}_scalef", 1.0))
            return np.asarray(images[key]), coords * scale, scale

    return None, coords, 1.0


def _nearest_spacing(coords):
    unique_coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    if len(unique_coords) < 2:
        return 1.0

    distances = cKDTree(unique_coords).query(
        unique_coords,
        k=2,
    )[0][:, 1]
    distances = distances[
        np.isfinite(distances) & (distances > 0)
    ]

    return float(np.median(distances)) if distances.size else 1.0


def _register_spatialdata(adata_obj):
    cache_key = id(adata_obj)
    if cache_key in _SPATIAL_CACHE:
        return _SPATIAL_CACHE[cache_key]

    image, coords, scale = _get_image_coords_scale(adata_obj)
    index = pd.Index(
        adata_obj.obs_names.astype(str),
        name="instance_id",
    )

    spacing = _nearest_spacing(coords)
    half = SPATIAL_SQUARE_SIZE * spacing / 2.0

    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=shapely_box(
                coords[:, 0] - half,
                coords[:, 1] - half,
                coords[:, 0] + half,
                coords[:, 1] + half,
            ),
            index=index,
        )
    )

    table_obs = pd.DataFrame(index=index)
    table_obs["region"] = pd.Categorical(
        ["spots"] * adata_obj.n_obs
    )
    table_obs["instance_id"] = index

    table = TableModel.parse(
        ad.AnnData(obs=table_obs),
        region="spots",
        region_key="region",
        instance_key="instance_id",
    )

    images = {}
    if image is not None:
        images["histology"] = Image2DModel.parse(
            _to_grayscale_image(image)[None, ...],
            dims=("c", "y", "x"),
            c_coords=["gray"],
        )

    sdata = sd.SpatialData(
        images=images,
        shapes={"spots": shapes},
        tables={"table": table},
    )

    out = (sdata, shapes.total_bounds, scale)
    _SPATIAL_CACHE[cache_key] = out
    return out


def _spatialplot_square(
    adata_obj,
    color_key,
    ax,
    crop_coord=None,
    palette=None,
    cmap=None,
    vmin=None,
    vmax=None,
    colorbar=False,
    legend_loc=None,
    show_background=True,
    hide_shapes=False,
    title="",
    frameon=False,
):
    """Plot Visium-HD spots as square SpatialData shapes."""
    sdata, bounds, scale = _register_spatialdata(adata_obj)
    plot = sdata

    if show_background and "histology" in sdata.images:
        plot = plot.pl.render_images(
            element="histology",
            cmap="gray",
            alpha=SPATIAL_IMAGE_ALPHA,
            scale="full",
            method="matplotlib",
        )

    if not hide_shapes:
        values = adata_obj.obs[color_key]

        if isinstance(values.dtype, pd.CategoricalDtype):
            categories = list(values.cat.categories.astype(str))
            sdata.tables["table"].obs[color_key] = pd.Categorical(
                values.astype("string").to_numpy(),
                categories=categories,
                ordered=values.cat.ordered,
            )

            if palette is None:
                colors_key = f"{color_key}_colors"
                stored_colors = adata_obj.uns.get(colors_key, [])
                if len(stored_colors) >= len(categories):
                    palette = dict(
                        zip(categories, stored_colors[:len(categories)])
                    )
                else:
                    palette = dict(
                        zip(categories, color_list[:len(categories)])
                    )
            else:
                palette = {str(k): v for k, v in palette.items()}

            plot = plot.pl.render_shapes(
                element="spots",
                color=color_key,
                palette=palette,
                fill_alpha=0.98,
                outline_width=0,
                table_name="table",
                method="matplotlib",
                colorbar=False,
            )
        else:
            sdata.tables["table"].obs[color_key] = (
                values.to_numpy(dtype=float)
            )

            norm = None
            if vmin is not None or vmax is not None:
                finite = values.to_numpy(dtype=float)
                finite = finite[np.isfinite(finite)]
                default_min = float(finite.min()) if finite.size else 0.0
                default_max = float(finite.max()) if finite.size else 1.0
                norm = Normalize(
                    vmin=default_min if vmin is None else vmin,
                    vmax=default_max if vmax is None else vmax,
                )

            plot = plot.pl.render_shapes(
                element="spots",
                color=color_key,
                cmap=cmap,
                norm=norm,
                fill_alpha=0.98,
                outline_width=0,
                table_name="table",
                method="matplotlib",
                colorbar=colorbar,
            )

    plot.pl.show(
        ax=ax,
        title=title,
        frameon=frameon,
        legend_loc=legend_loc,
        colorbar=colorbar,
        show=False,
    )

    # Preserve the native histology pixels without smoothing.
    for image_artist in ax.images:
        image_artist.set_interpolation("none")
        image_artist.set_resample(False)

    if crop_coord is not None:
        xmin, xmax, ymin, ymax = crop_coord
        ax.set_xlim(xmin * scale, xmax * scale)
        ax.set_ylim(ymax * scale, ymin * scale)
    else:
        x0, y0, x1, y1 = bounds
        pad_x = 0.035 * (x1 - x0)
        pad_y = 0.035 * (y1 - y0)
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y1 + pad_y, y0 - pad_y)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.01)
    return ax


SAVE_INDIVIDUAL_SUBPLOTS = False
CLOSE_INDIVIDUAL_SUBPLOTS = False
SAVE_DGEA_RESULTS_CSV = False
PREFER_EXISTING_DGEA_CSV = True


def maybe_save_and_close_current_figure(
    output_stem,
    dpi=600,
    bbox_inches="tight",
    save_individual=SAVE_INDIVIDUAL_SUBPLOTS,
    close_individual=CLOSE_INDIVIDUAL_SUBPLOTS,
):
    """Optionally save current figure as PNG/PDF/SVG and optionally close it."""
    if save_individual:
        plt.savefig(f"{output_stem}.png", dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(f"{output_stem}.pdf", dpi=dpi, bbox_inches=bbox_inches)
        plt.savefig(f"{output_stem}.svg", dpi=dpi, bbox_inches=bbox_inches)
    if close_individual:
        plt.close(plt.gcf())


sample_names = ['SCAF4093_3229997_A1']

adataa200 = defaultdict()

cap_value = 600
umi_thres2=200
for res in ['8']:
    for sample_name in sample_names:
        adata0 = load_data(sample_name, resolution = res)
        #adataaori[sample_name] = adata0

        adata1 = rm_free_lying_pts(adata0, sample_name, res, threshold=8000, plot=False, pathh = '') # do not plot again
        adata2 = adata1.copy()

        #adata1 = qc_spot_and_gene_removal(adata1, umi_threshold=umi_thres1, gene_threshold=None) # using 100 for umi threshold and do not remove any gene yet
        #adataa100[sample_name] = adata1

        adata2 = qc_spot_and_gene_removal(adata2, umi_threshold=umi_thres2, gene_threshold=None)
        adataa200[sample_name] = adata2

adata200 = copy.deepcopy(adataa200)

# color_list = ['#3049ad', '#fe8011', '#1b7837', '#d62a2b', '#ab43fc',
#               '#8d574c', '#e187c4', '#b8bd6c', '#23bed0', '#bc510a',
#               '#0aac00', '#ff008c', '#057dff', '#a7d1e6', '#ffbb79',
#               '#99df8b', '#ff9997', '#c6b1d4', '#c49d95', '#f7bad0',
#               '#dcdb91', '#a1dbe5', '#dea884', '#84d680', '#ff80c6',
#               '#eb9495', '#91dee8', '#d3e8f2', '#ffddbc', '#ccefc5',
#               '#ffcccb', '#e2d8ea', '#e2ceca', '#fbdce8', '#eeedc8']

CLUSTER_COLORS = [
    "#3049ad", "#fe8011", "#1b7837", "#fa0000", "#ab43fc",
    "#8d574c", "#ff00d9", "#bcbd22", "#17becf",
    "#8baaf3", "#ffbb79", "#99df8b", "#fe7775", "#c6b1d4",
    "#c49d95", "#ff80c6", "#dcdb91", "#a7d1e6",
    "#393b79", "#8c6d31", "#0aac00", "#982109", "#7b4173",
    "#713230", "#ff008c", "#637939",
    "#e7cb94", "#ccefc5", "#efcece", "#f7b6d2", "#eeedc8",
]

def align_pred_palette_to_ref(ref, pred, color_pool):
    """
    Align predicted cluster colors to a reference clustering by maximum overlap.
    Mirrors the color-alignment metric used in method_uncertainty_plot.py.
    """
    ref_s = pd.Series(ref, copy=False).astype(str)
    pred_s = pd.Series(pred, index=ref_s.index, copy=False).astype(str)

    def _numeric_aware_sort(vals):
        """Sort numeric-like labels by numeric value, others lexicographically."""
        vals = list(pd.unique(pd.Series(vals).astype(str)))
        parsed = []
        for v in vals:
            try:
                parsed.append((v, float(v), True))
            except ValueError:
                parsed.append((v, None, False))
        numeric_part = sorted([t for t in parsed if t[2]], key=lambda x: (x[1], x[0]))
        text_part = sorted([t for t in parsed if not t[2]], key=lambda x: x[0])
        return [t[0] for t in numeric_part + text_part]

    ref_cats = _numeric_aware_sort(ref_s)
    if len(color_pool) < len(ref_cats):
        raise ValueError(f"Need >= {len(ref_cats)} colors, got {len(color_pool)}.")
    ref_palette = dict(zip(ref_cats, color_pool[:len(ref_cats)]))

    # pred cluster -> reference cluster by majority overlap
    pred_to_ref = pd.crosstab(pred_s, ref_s).idxmax(axis=1).to_dict()
    pred_cats = _numeric_aware_sort(pred_s)

    # Relabel pred clusters to reference cluster IDs so color and cluster number both align.
    # If several pred clusters map to the same reference cluster, keep one as that ref ID and
    # assign remaining ones fresh numeric IDs (with extra colors).
    grouped = {}
    for c in pred_cats:
        grouped.setdefault(pred_to_ref[c], []).append(c)

    ref_numeric = []
    all_ref_numeric = True
    for c in ref_cats:
        try:
            ref_numeric.append(int(float(c)))
        except ValueError:
            all_ref_numeric = False
            break

    pred_to_aligned = {}
    extra_aligned_labels = []
    if all_ref_numeric:
        used = set(ref_numeric)
        next_id = (max(used) + 1) if used else 0
        for ref_id in ref_cats:
            members = grouped.get(ref_id, [])
            if not members:
                continue
            pred_to_aligned[members[0]] = ref_id
            for member in members[1:]:
                while next_id in used:
                    next_id += 1
                new_label = str(next_id)
                used.add(next_id)
                pred_to_aligned[member] = new_label
                extra_aligned_labels.append(new_label)
                next_id += 1
    else:
        # Fallback for non-numeric reference labels.
        for ref_id in ref_cats:
            members = grouped.get(ref_id, [])
            if not members:
                continue
            pred_to_aligned[members[0]] = ref_id
            for i, member in enumerate(members[1:], start=1):
                new_label = f"{ref_id}_{i}"
                pred_to_aligned[member] = new_label
                extra_aligned_labels.append(new_label)

    # keep any unmapped categories safe (should be rare)
    for c in pred_cats:
        if c not in pred_to_aligned:
            pred_to_aligned[c] = c
            if c not in ref_cats:
                extra_aligned_labels.append(c)

    extra_aligned_labels = _numeric_aware_sort(extra_aligned_labels)
    need_total = len(ref_cats) + len(extra_aligned_labels)
    if len(color_pool) < need_total:
        raise ValueError(f"Need >= {need_total} colors for overlap alignment, got {len(color_pool)}.")

    extra_palette = dict(zip(extra_aligned_labels, color_pool[len(ref_cats):need_total]))
    pred_aligned_s = pred_s.map(pred_to_aligned)
    pred_aligned_cats = _numeric_aware_sort(pred_aligned_s)
    pred_palette = {}
    for cat in pred_aligned_cats:
        pred_palette[cat] = ref_palette[cat] if cat in ref_palette else extra_palette[cat]

    return ref_palette, pred_palette, ref_cats, pred_aligned_cats, pred_aligned_s

iterations = 50
pathh = '/home/wuw15/data_dir/my_analysis_python/data_exploratory/uncertainty/'
results = joblib.load(f'{pathh}leiden_results_{iterations}iterations.pkl')
idx=1
results1 = results[idx*50:(idx+1)*50][1]
results2 = results[idx*50:(idx+1)*50][2]
results3 = results[idx*50:(idx+1)*50][0]

adata = adata200[sample_names[0]]

results1_s = pd.Series(results1).astype(str)
results2_s = pd.Series(results2).astype(str)
results3_s = pd.Series(results3).astype(str)

ref_palette, _, ref_cats, _, _ = align_pred_palette_to_ref(results1_s, results1_s, color_list)
_, results2_palette, _, results2_cats, results2_aligned_s = align_pred_palette_to_ref(results1_s, results2_s, color_list)
_, results3_palette, _, results3_cats, results3_aligned_s = align_pred_palette_to_ref(results1_s, results3_s, color_list)

# original data and data after QC: subplot 1
from functions import load_data
adataori = load_data('SCAF4093_3229997_A1', resolution = '8')
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adataori, img_key = "hires", color=f"total_counts", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100)) # xmin(left), xmax(right), ymin(top), ymax(bottom)
    axs = _spatialplot_square(
        adataori, "total_counts", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        cmap="viridis", vmin=0, vmax=1000,
        colorbar=True, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    fig = plt.gcf() # rename colorbar label
    cbar_ax = fig.axes[-1]   # colorbar axis is usually the last one
    cbar_ax.set_ylabel("UMI count")
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/original_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# subplot 2
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata, img_key = "hires", color=f"total_counts", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100)) # xmin(left), xmax(right), ymin(top), ymax(bottom)
    axs = _spatialplot_square(
        adata, "total_counts", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        cmap="viridis", vmin=0, vmax=1000,
        colorbar=True, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    fig = plt.gcf() # rename colorbar label
    cbar_ax = fig.axes[-1]   # colorbar axis is usually the last one
    cbar_ax.set_ylabel("UMI count")
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/original_clean_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# clustering results, cropped, before filtering: subplot 3
adata.obs['results1'] = pd.Categorical(results1_s, categories=ref_cats, ordered=True)
adata.uns['results1_colors'] = [ref_palette[c] for c in ref_cats]
k = adata.obs['results1'].nunique()
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata, img_key = "hires", color=f"results1", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100))
    axs = _spatialplot_square(
        adata, "results1", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        legend_loc=None, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    #ax.text(0.98, 0.02, f"k = {k}", transform=ax.transAxes, ha="right", va="bottom", fontsize=20, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/sample1_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# subplot 4
adata.obs['results2'] = pd.Categorical(results2_aligned_s, categories=results2_cats, ordered=True)
adata.uns['results2_colors'] = [results2_palette[c] for c in results2_cats]
k = adata.obs['results2'].nunique()
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata, img_key = "hires", color=f"results2", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100))
    axs = _spatialplot_square(
        adata, "results2", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        legend_loc=None, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    #ax.text(0.98, 0.02, f"k = {k}", transform=ax.transAxes, ha="right", va="bottom", fontsize=20, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/sample2_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# subplot 5
adata.obs['results3'] = pd.Categorical(results3_aligned_s, categories=results3_cats, ordered=True)
adata.uns['results3_colors'] = [results3_palette[c] for c in results3_cats]
k = adata.obs['results3'].nunique()
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata, img_key = "hires", color=f"results3", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100))
    axs = _spatialplot_square(
        adata, "results3", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        legend_loc=None, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    #ax.text(0.98, 0.02, f"k = {k}", transform=ax.transAxes, ha="right", va="bottom", fontsize=20, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/sample3_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# plot uncertainty
from matplotlib.colors import LinearSegmentedColormap
adata_with_uncert = sc.read_h5ad(f"/home/wuw15/data_dir/my_analysis_python/data_exploratory/uncertainty/intra_inter_method/SCAF4093_3229997_A1_calculated.h5ad")

TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bins = [0.0] + TH + [1.0]
th_labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins)-2)] + [f"[{bins[-2]:.1f},{bins[-1]:.1f}]"]
thres_palette = {
    # least uncertain: blue shades
    "[0.0,0.1)": "#08306bea",
    "[0.1,0.2)": "#4292c6",
    "[0.2,0.3)": "#9ecae1",

    # medium: warm grey (slightly brown) shades
    "[0.3,0.4)": "#deebf7",

    # high: orange shades (unchanged)
    "[0.4,0.5)": "#fee391",
    "[0.5,0.6)": "#fec44f",
    "[0.6,0.7)": "#fdae6b",
    "[0.7,0.8)": "#fd8d3c",

    # very high / most uncertain: red shades
    "[0.8,0.9)": "#fb6a4a",
    "[0.9,1.0]": "#cb181d"} # turn this into a bar and for dark blue i would write highly stable and for the bottom red i would write highly unstable
# continuous cmap using the same colors in order
cmap = LinearSegmentedColormap.from_list(
    "uncertainty_custom",
    list(thres_palette.values()),
    N=256)
def _plot_counts(ax, series, order, palette, title, ylabel=True):
    vc = series.value_counts()
    counts = [int(vc.get(cat, 0)) for cat in order]
    colors = [palette[cat] for cat in order]

    ax.bar(range(len(order)), counts, color=colors)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel("#spots", fontsize=9)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)

# subplot 6
u = adata_with_uncert.obs[f"leiden_uncertainty"].to_numpy(dtype=float)
# # 1) absolute-threshold bins
# th_cat = pd.cut(
#     u,
#     bins=bins,
#     labels=th_labels,
#     include_lowest=True,
#     right=False)
# # make last bin closed on the right so u==1.0 is handled
# th_cat = th_cat.astype("object")
# th_cat[pd.isna(th_cat) & (u == 1.0)] = th_labels[-1]
# adata_with_uncert.obs[f"leiden_uncertainty_thres_bin"] = pd.Categorical(th_cat, categories=th_labels, ordered=True)
# adata_with_uncert.obs[f"leiden_uncertainty_thres_bin"] = adata_with_uncert.obs[f"leiden_uncertainty_thres_bin"].cat.reorder_categories(list(thres_palette.keys()), ordered=True)
# axs = sc.pl.spatial(adata_with_uncert, img_key="hires", color="leiden_uncertainty_thres_bin", size=1.5, bw=True, palette=thres_palette, title="", crop_coord=(7900, 9860, 5100, 7100), frameon=False, show=False)
if SAVE_INDIVIDUAL_SUBPLOTS:
    axs = _spatialplot_square(
        adata_with_uncert, "leiden_uncertainty", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        cmap=cmap, vmin=0.0, vmax=1.0,
        colorbar=True, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    # rename colorbar and set ticks at your thresholds
    fig = plt.gcf()
    cbar_ax = fig.axes[-1]
    cbar_ax.invert_yaxis() # put 0.0 at top and 1.0 at bottom
    cbar_ax.set_ylabel("Instability", rotation=270, labelpad=15)
    cbar_ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #cbar_ax.set_ylabel("Highly stable                      Highly unstable", rotation=270, labelpad=22)
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/uncert_bin_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# filtered clustering: subplot 7
u = adata_with_uncert.obs[f"leiden_uncertainty"].to_numpy(dtype=float)
keep_fn = lambda uu, t: (uu <= t)
keep = keep_fn(u, 0.3)
adata_plot = adata[keep].copy()
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata_plot, img_key = "hires", color=f"results1", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100))
    axs = _spatialplot_square(
        adata_plot, "results1", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        legend_loc=None, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/sample1_filtered_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# subplot 8
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata_plot, img_key = "hires", color=f"results2", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100))
    axs = _spatialplot_square(
        adata_plot, "results2", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        legend_loc=None, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/sample2_filtered_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# subplot 9
if SAVE_INDIVIDUAL_SUBPLOTS:
    #sc.pl.spatial(adata_plot, img_key = "hires", color=f"results3", size=1.5, bw = True, title="", crop_coord=(7900, 9860, 5100, 7100))
    axs = _spatialplot_square(
        adata_plot, "results3", plt.gca(),
        crop_coord=(7900, 9860, 5100, 7100),
        legend_loc=None, frameon=False,
    )
    ax = axs[0] if isinstance(axs, list) else axs
    ax.set_xlabel("")# remove "spatial 1" / "spatial 2"
    ax.set_ylabel("")
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/sample3_filtered_zoom",
        dpi=600,
        bbox_inches="tight",
    )

# toy knn graph: subplot 10
# plot an arbitrary kNN graph for a methods overview figure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
# toy clustered points
base_seed = 42
rng = np.random.RandomState(base_seed)
X, y = make_blobs(n_samples=200, centers=[(-2.2, -0.5), (-0.4, 1.8), (1.8, 0.2)], cluster_std=[0.7, 0.7, 0.7], random_state=base_seed)
A = kneighbors_graph(X, n_neighbors=12, mode="connectivity", include_self=False) # build kNN graph
A = A.maximum(A.T).tocsr()  # symmetrize like a typical clustering graph
n_show = 200 # choose a subset of spots
idx = np.random.RandomState(0).choice(X.shape[0], size=n_show, replace=False)
idx = np.sort(idx)
idx_set = set(idx)
if SAVE_INDIVIDUAL_SUBPLOTS:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    # edges
    for i in idx:
        nbrs = A[i].indices
        for j in nbrs:
            if j in idx_set and i < j:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="black", lw=1.0, alpha=0.25, zorder=1)
    # nodes
    ax.scatter(X[idx, 0], X[idx, 1], s=24, color="white", edgecolor="black", linewidth=1.0, zorder=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    plt.tight_layout()
    maybe_save_and_close_current_figure(
        "/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/knn_graph_subset_toy",
        dpi=600,
        bbox_inches="tight",
    )

#-----------------------------
# for these three runs, do DGEA
def preprocess_adata_for_de(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.raw = None  # explicitly disable raw
    return adata

def pct_expr_for_genes(adata, obs_col, group_label, genes):
    """Fraction of cells with non-zero expression for each gene in one group."""
    genes = pd.Index(pd.Series(genes, dtype="string").astype(str))
    if len(genes) == 0:
        return pd.Series(dtype=float)

    mask = adata.obs[obs_col].astype(str).eq(str(group_label)).to_numpy()
    if mask.sum() == 0:
        return pd.Series(np.nan, index=genes, dtype=float)

    idx = adata.var_names.get_indexer(genes)
    valid = idx >= 0
    out = np.full(len(genes), np.nan, dtype=float)
    if valid.any():
        Xg = adata[mask, idx[valid]].X
        pct = np.asarray((Xg > 0).mean(axis=0)).ravel()
        out[valid] = pct
    return pd.Series(out, index=genes, dtype=float)

def DGEA(
    adata,
    comp_name,
    pathh,
    layer='hier_consensus',
    n_genes=40,
    pval_adj_thres=0.05,
    min_pct_expr=0.1,
    min_abs_logfc=0.5,
    file_name=None,
    anonymize_gene_labels=False,
    gene_alias_map=None,
    save_individual_plots=SAVE_INDIVIDUAL_SUBPLOTS,
    close_individual_plots=CLOSE_INDIVIDUAL_SUBPLOTS,
):
    if gene_alias_map is None:
        gene_alias_map = {}
    dgea_rows = []
    adata = adata.copy()  # make a copy so the original anndata is not modified

    vc = adata.obs[layer].astype(str).value_counts()
    keep_groups = vc[vc >= 10].index
    if len(keep_groups) < 2:
        empty_df = pd.DataFrame(columns=[
            "gene", "logfc", "pval_adj", "score", "pct_expr", "abs_score", "abs_logfc",
            "group", "reference", "layer", "gene_original", "gene_plot_label"
        ])
        return gene_alias_map, empty_df

    adata = adata[adata.obs[layer].astype(str).isin(keep_groups)].copy()
    adata.obs[layer] = adata.obs[layer].astype("category")
    if adata.obs[layer].nunique() < 2:
        empty_df = pd.DataFrame(columns=[
            "gene", "logfc", "pval_adj", "score", "pct_expr", "abs_score", "abs_logfc",
            "group", "reference", "layer", "gene_original", "gene_plot_label"
        ])
        return gene_alias_map, empty_df

    for name in comp_name:
        group, ref = name.split('vs', 1)
        if (group not in adata.obs[layer].cat.categories) or (ref not in adata.obs[layer].cat.categories):
            continue
        sc.tl.rank_genes_groups(
            adata,
            groupby=layer,
            method="wilcoxon",
            key_added=name,
            groups=[group],
            reference=ref,
            rankby_abs=True,
            tie_correct=True,
            use_raw=False,
        )
        #sc.pl.rank_genes_groups(adata, n_genes=40, sharey=False, show_gene_labels=True)
        #plt.savefig(f"{pathh}differential_8um_leiden_{group}vs{ref}_abs.png", dpi = 300, bbox_inches='tight')
        result = adata.uns[name]

        df = pd.DataFrame({'gene': result['names'][group],
                        'logfc': result['logfoldchanges'][group],
                        'pval_adj': result['pvals_adj'][group],
                        'score': result['scores'][group]}).dropna()

        # Step 1: Filter and sort genes consistently with all_dgea.py
        df["pct_expr"] = pct_expr_for_genes(
            adata,
            layer,
            group,
            df["gene"],
        ).to_numpy()
        df["abs_score"] = df["score"].abs()
        df["abs_logfc"] = df["logfc"].abs()
        df = df[df["pval_adj"] < pval_adj_thres]
        df = df[df["pct_expr"] > float(min_pct_expr)]
        df = df[df["abs_logfc"] >= float(min_abs_logfc)]
        top_df = df.sort_values("abs_score", ascending=False).head(n_genes).reset_index(drop=True)

        top_df["group"] = group
        top_df["reference"] = ref
        top_df["layer"] = layer

        # Step 2: Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(top_df))
        y = top_df["score"]
        genes = top_df["gene"]

        if anonymize_gene_labels:
            plot_gene_labels = []
            for gene in genes:
                g = str(gene)
                if g not in gene_alias_map:
                    gene_alias_map[g] = f"Gene {len(gene_alias_map) + 1}"
                plot_gene_labels.append(gene_alias_map[g])
        else:
            plot_gene_labels = genes

        top_df["gene_original"] = top_df["gene"].astype(str)
        if anonymize_gene_labels:
            top_df["gene_plot_label"] = top_df["gene_original"].map(gene_alias_map)
        else:
            top_df["gene_plot_label"] = top_df["gene_original"]
        dgea_rows.append(top_df)

        # Apply custom colors
        bar_colors = ['#D65A5A' if score > 0 else '#6EAED7' for score in y]

        ax.bar(x, y, color=bar_colors)

        # X-axis setup
        ax.set_xticks(x)
        #ax.set_xticklabels(plot_gene_labels, rotation=90, fontsize=8)

        # Labels and style
        #ax.set_ylabel("Wilcoxon score")
        ax.set_ylabel("Scores")
        ax.set_xlabel("Genes")
        ax.tick_params(axis='x', which='both', labelbottom=False)
        ax.tick_params(axis='y', which='both', labelleft=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        maybe_save_and_close_current_figure(
            f"{pathh}differential_8um_leiden_{group}vs{ref}_top40_by_score{file_name}",
            dpi=600,
            bbox_inches="tight",
            save_individual=save_individual_plots,
            close_individual=close_individual_plots,
        )

    if dgea_rows:
        dgea_df = pd.concat(dgea_rows, ignore_index=True)
    else:
        dgea_df = pd.DataFrame(columns=[
            "gene", "logfc", "pval_adj", "score", "pct_expr", "abs_score", "abs_logfc",
            "group", "reference", "layer", "gene_original", "gene_plot_label"
        ])

    return gene_alias_map, dgea_df

comp_name = ['4vs8']
pathh = '/home/wuw15/data_dir/my_analysis_python/uncertainty_public_data/outs/organized_plots/fig1/'

sample_name = 'SCAF4093_3229997_A1'
load_pathh = '/home/wuw15/data_dir/my_analysis_python/data_exploratory/uncertainty/intra_inter_method/'
adata1 = sc.read_h5ad(f'{load_pathh}{sample_name}_calculated.h5ad')

# load the uncertainty values
adata.obs['leiden_uncertainty'] = adata1.obs['leiden_uncertainty'].copy()

USE_ANON_GENE_LABELS = True
MIN_PCT_EXPR = 0.0
MIN_ABS_LOGFC = 0.0
dgea_csv_paths = {
    "results1": Path(pathh) / "dgea_results1_all_thresholds.csv",
    "results2": Path(pathh) / "dgea_results2_all_thresholds.csv",
    "results3": Path(pathh) / "dgea_results3_all_thresholds.csv",
}

if PREFER_EXISTING_DGEA_CSV and all(p.exists() for p in dgea_csv_paths.values()):
    dgea_results1_df = pd.read_csv(dgea_csv_paths["results1"])
    dgea_results2_df = pd.read_csv(dgea_csv_paths["results2"])
    dgea_results3_df = pd.read_csv(dgea_csv_paths["results3"])
else:
    gene_alias_map = {}
    dgea_by_run = {"results1": [], "results2": [], "results3": []}

    for uncertainty_threshold in [0.2, 0.8]:
        adata_filtered = adata[adata.obs['leiden_uncertainty'] < uncertainty_threshold].copy()
        adata_filtered = preprocess_adata_for_de(adata_filtered)
        gene_alias_map, dgea_df = DGEA(
            adata_filtered,
            comp_name,
            pathh,
            layer='results1',
            n_genes=40,
            pval_adj_thres=0.05,
            min_pct_expr=MIN_PCT_EXPR,
            min_abs_logfc=MIN_ABS_LOGFC,
            file_name=f"_{uncertainty_threshold}_result1",
            anonymize_gene_labels=USE_ANON_GENE_LABELS,
            gene_alias_map=gene_alias_map,
        )
        dgea_df["uncertainty_threshold"] = uncertainty_threshold
        dgea_by_run["results1"].append(dgea_df)

        # subplot 11 12 13
        gene_alias_map, dgea_df = DGEA(
            adata_filtered,
            comp_name,
            pathh,
            layer='results2',
            n_genes=40,
            pval_adj_thres=0.05,
            min_pct_expr=MIN_PCT_EXPR,
            min_abs_logfc=MIN_ABS_LOGFC,
            file_name=f"_{uncertainty_threshold}_result2",
            anonymize_gene_labels=USE_ANON_GENE_LABELS,
            gene_alias_map=gene_alias_map,
        )
        dgea_df["uncertainty_threshold"] = uncertainty_threshold
        dgea_by_run["results2"].append(dgea_df)

        # subplot 14 15 16
        gene_alias_map, dgea_df = DGEA(
            adata_filtered,
            comp_name,
            pathh,
            layer='results3',
            n_genes=40,
            pval_adj_thres=0.05,
            min_pct_expr=MIN_PCT_EXPR,
            min_abs_logfc=MIN_ABS_LOGFC,
            file_name=f"_{uncertainty_threshold}_result3",
            anonymize_gene_labels=USE_ANON_GENE_LABELS,
            gene_alias_map=gene_alias_map,
        )
        dgea_df["uncertainty_threshold"] = uncertainty_threshold
        dgea_by_run["results3"].append(dgea_df)

    dgea_results1_df = pd.concat(dgea_by_run["results1"], ignore_index=True) if dgea_by_run["results1"] else pd.DataFrame()
    dgea_results2_df = pd.concat(dgea_by_run["results2"], ignore_index=True) if dgea_by_run["results2"] else pd.DataFrame()
    dgea_results3_df = pd.concat(dgea_by_run["results3"], ignore_index=True) if dgea_by_run["results3"] else pd.DataFrame()

    if SAVE_DGEA_RESULTS_CSV:
        dgea_results1_df.to_csv(dgea_csv_paths["results1"], index=False)
        dgea_results2_df.to_csv(dgea_csv_paths["results2"], index=False)
        dgea_results3_df.to_csv(dgea_csv_paths["results3"], index=False)


# # calculate jaccard
# import pandas as pd
# from itertools import combinations

# def gene_set_for_threshold(df, th, gene_col="gene_original"):
#     x = df[df["uncertainty_threshold"] == th].copy()
#     if gene_col not in x.columns:
#         gene_col = "gene"
#     return set(x[gene_col].astype(str).dropna())

# def jaccard(a, b):
#     union = a | b
#     if len(union) == 0:
#         return float("nan")
#     return len(a & b) / len(union)

# run_dfs = {
#     "results1": dgea_results1_df,
#     "results2": dgea_results2_df,
#     "results3": dgea_results3_df,
# }

# rows = []
# for th in [0.2, 0.8]:
#     gene_sets = {run: gene_set_for_threshold(df, th) for run, df in run_dfs.items()}
#     for r1, r2 in combinations(run_dfs.keys(), 2):
#         g1, g2 = gene_sets[r1], gene_sets[r2]
#         rows.append({
#             "uncertainty_threshold": th,
#             "run_pair": f"{r1} vs {r2}",
#             "n_genes_1": len(g1),
#             "n_genes_2": len(g2),
#             "intersection": len(g1 & g2),
#             "union": len(g1 | g2),
#             "jaccard": jaccard(g1, g2),
#         })

# jaccard_df = pd.DataFrame(rows)

# -----------------------------
# combine subplot 1-16 into one big VECTOR figure (Illustrator-editable)
def _clean_axis(ax):
    ax.set_axis_on()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)


def _plot_spatial_panel(ax, adata_obj, color_key, title_text, **kwargs):
    crop_coord = kwargs.pop("crop_coord", None)
    cmap_here = kwargs.pop("color_map", kwargs.pop("cmap", None))
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    legend_loc = kwargs.pop("legend_loc", None)
    colorbar_loc = kwargs.pop("colorbar_loc", None)

    _spatialplot_square(
        adata_obj,
        color_key,
        ax,
        crop_coord=crop_coord,
        cmap=cmap_here,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar_loc is not None,
        legend_loc=legend_loc,
        frameon=False,
    )
    _clean_axis(ax)
    ax.set_title(title_text, fontsize=10)


def _plot_toy_knn_panel(ax, title_text):
    # Reuse toy graph objects built above (X, A, idx, idx_set).
    for i in idx:
        nbrs = A[i].indices
        for j in nbrs:
            if j in idx_set and i < j:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="black", lw=1.0, alpha=0.25, zorder=1)
    ax.scatter(X[idx, 0], X[idx, 1], s=24, color="white", edgecolor="black", linewidth=1.0, zorder=2)
    _clean_axis(ax)
    ax.set_title(title_text, fontsize=10)


def _plot_dgea_panel(ax, dgea_df, run_name, thres, title_text, n_genes=40):
    x = dgea_df[
        (dgea_df["uncertainty_threshold"] == thres) &
        (dgea_df["layer"].astype(str) == run_name)
    ].copy()

    if x.empty:
        ax.text(0.5, 0.5, f"No data\n{run_name}, th={thres}", ha="center", va="center", fontsize=9)
        _clean_axis(ax)
        ax.set_title(title_text, fontsize=10)
        return

    x["abs_score"] = x["score"].abs()
    x = x.sort_values("abs_score", ascending=False).head(n_genes)
    y = x["score"].to_numpy()
    bar_colors = ["#D65A5A" if v > 0 else "#6EAED7" for v in y]
    ax.bar(np.arange(len(y)), y, color=bar_colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    _clean_axis(ax)
    ax.set_title(title_text, fontsize=10)

# coords = (7900, 9860, 5100, 7100)
coords=(7900, 9860, 5400, 6820)

def save_combined_overall_figure_vector(output_dir, dpi=600):
    output_dir = Path(output_dir)

    fig = plt.figure(figsize=(44, 22))
    outer = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[45, 27.5, 27.5],  # 3x4 small block | DGEA th=0.2 | DGEA th=0.8
        height_ratios=[1, 1, 1, 1],
        wspace=0.06,
        hspace=0.20,
    )
    left_gs = outer[:, 0].subgridspec(4, 4, wspace=0.04, hspace=0.30)      # subplot 1-10 (+2 legend slots) + 4 clusterwise
    mid_gs = outer[:3, 1].subgridspec(3, 1, wspace=0.00, hspace=0.20)      # subplot 11-13 (th=0.2), top 3 rows only
    right_gs = outer[:3, 2].subgridspec(3, 1, wspace=0.00, hspace=0.20)    # subplot 14-16 (th=0.8), top 3 rows only

    # subplot 1-10
    panel_specs = [
        ("subplot 1", lambda ax: _plot_spatial_panel(ax, adataori, "total_counts", "subplot 1", crop_coord=coords, vmin=0, vmax=1000, colorbar_loc=None)),
        ("subplot 2", lambda ax: _plot_spatial_panel(ax, adata, "total_counts", "subplot 2", crop_coord=coords, vmin=0, vmax=1000, colorbar_loc=None)),
        ("subplot 3", lambda ax: _plot_spatial_panel(ax, adata, "results1", "subplot 3", crop_coord=coords, legend_loc=None)),
        ("subplot 4", lambda ax: _plot_spatial_panel(ax, adata, "results2", "subplot 4", crop_coord=coords, legend_loc=None)),
        ("subplot 5", lambda ax: _plot_spatial_panel(ax, adata, "results3", "subplot 5", crop_coord=coords, legend_loc=None)),
        ("subplot 6", lambda ax: _plot_spatial_panel(ax, adata_with_uncert, "leiden_uncertainty", "subplot 6", crop_coord=coords, color_map=cmap, vmin=0.0, vmax=1.0, colorbar_loc=None)),
        ("subplot 7", lambda ax: _plot_spatial_panel(ax, adata_plot, "results1", "subplot 7", crop_coord=coords, legend_loc=None)),
        ("subplot 8", lambda ax: _plot_spatial_panel(ax, adata_plot, "results2", "subplot 8", crop_coord=coords, legend_loc=None)),
        ("subplot 9", lambda ax: _plot_spatial_panel(ax, adata_plot, "results3", "subplot 9", crop_coord=coords, legend_loc=None)),
        ("subplot 10", lambda ax: _plot_toy_knn_panel(ax, "subplot 10")),
    ]

    for i, (_, draw_fn) in enumerate(panel_specs):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(left_gs[r, c])
        draw_fn(ax)

    # Put thin, legend-like colorbars in the two empty slots (3rd row, col 3 and 4).
    # (2,3): shared total-counts colorbar for subplot 1/2
    cbar_host_12 = fig.add_subplot(left_gs[2, 2])
    cbar_host_12.axis("off")
    cbar_ax = cbar_host_12.inset_axes([0.44, 0.08, 0.12, 0.84])
    umi_norm = Normalize(vmin=0, vmax=1000)
    umi_sm = ScalarMappable(norm=umi_norm, cmap=plt.get_cmap("viridis"))
    umi_sm.set_array([])
    cbar = fig.colorbar(umi_sm, cax=cbar_ax)
    cbar_ax.set_xticks([])
    cbar_ax.tick_params(length=0, labelleft=True)
    for spine in cbar_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)

    # (2,4): subplot 6 uncertainty as discrete 10-bin legend.
    cbar_host_6 = fig.add_subplot(left_gs[2, 3])
    cbar_host_6.axis("off")
    cbar_host_6.text(0.50, 0.97, "Instability", ha="center", va="top", fontsize=9)
    n_bins = 10
    y_top = 0.91
    row_h = 0.082
    x_color = 0.08
    w_color = 0.16
    for i in range(n_bins):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        y = y_top - i * row_h
        rect = Rectangle(
            (x_color, y - row_h * 0.70),
            w_color,
            row_h * 0.62,
            transform=cbar_host_6.transAxes,
            facecolor=cmap((lo + hi) / 2.0),
            edgecolor="black",
            linewidth=0.6,
        )
        cbar_host_6.add_patch(rect)
        cbar_host_6.text(
            x_color + w_color + 0.06,
            y - row_h * 0.40,
            f"[{lo:.1f}, {hi:.1f}]" if i == n_bins - 1 else f"[{lo:.1f}, {hi:.1f})",
            transform=cbar_host_6.transAxes,
            ha="left",
            va="center",
            fontsize=8,
            color="black",
        )

    # subplot 11-16
    # middle column: th=0.2 for results1/2/3; right column: th=0.8 for results1/2/3
    dgea_map = [
        ("results1", 0.2, "subplot 11", "mid"),
        ("results2", 0.2, "subplot 12", "mid"),
        ("results3", 0.2, "subplot 13", "mid"),
        ("results1", 0.8, "subplot 14", "right"),
        ("results2", 0.8, "subplot 15", "right"),
        ("results3", 0.8, "subplot 16", "right"),
    ]
    dgea_all = pd.concat([dgea_results1_df, dgea_results2_df, dgea_results3_df], ignore_index=True)
    row_by_run = {"results1": 0, "results2": 1, "results3": 2}
    for run_name, thres, ttxt, col_id in dgea_map:
        r = row_by_run[run_name]
        ax = fig.add_subplot(mid_gs[r, 0] if col_id == "mid" else right_gs[r, 0])
        _plot_dgea_panel(ax, dgea_all, run_name, thres, ttxt, n_genes=40)

    # Additional row: 4 clusterwise spatial panels from precomputed CSV.
    cluster_bin_specs = [
        ("Highly unstable clusters", 0),
        ("Unstable clusters", 1),
        ("Modestly Stable clusters", 2),
        ("Stable clusters", 3),
    ]
    cluster_bin_csv = output_dir / "results1_cluster_mean_instability_by_bin.csv"
    cluster_bins_df = pd.read_csv(cluster_bin_csv) if cluster_bin_csv.exists() else pd.DataFrame()

    run_col = "results1"
    run_categories = adata.obs[run_col].cat.categories.astype(str).tolist()
    run_colors = adata.uns.get(f"{run_col}_colors", [])
    cluster_color_map = {str(cat): color for cat, color in zip(run_categories, run_colors)}

    for bin_label, c in cluster_bin_specs:
        ax = fig.add_subplot(left_gs[3, c])
        if not cluster_bins_df.empty and "bin_label" in cluster_bins_df.columns:
            rows = cluster_bins_df[cluster_bins_df["bin_label"] == bin_label].copy()
        else:
            rows = pd.DataFrame(columns=["cluster", "mean_instability"])

        if not rows.empty:
            rows["cluster"] = rows["cluster"].astype(str)
            rows = rows.sort_values("mean_instability", ascending=False)
            clusters_in_bin = rows["cluster"].tolist()
            adata_bin = adata[adata.obs[run_col].astype(str).isin(clusters_in_bin)].copy()
            _spatialplot_square(
                adata_bin,
                run_col,
                ax,
                crop_coord=coords,
                legend_loc=None,
                frameon=False,
            )
        else:
            clusters_in_bin = []
            _spatialplot_square(
                adata,
                None,
                ax,
                crop_coord=coords,
                hide_shapes=True,
                legend_loc=None,
                frameon=False,
            )
        _clean_axis(ax)
        ax.set_title(f"{bin_label}\n{len(clusters_in_bin)} clusters", fontsize=10)

        if len(clusters_in_bin) == 0:
            ax.text(
                0.5, -0.11, "None",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="black"
            )
            continue

        ax.text(
            0.5, -0.08, "Mean instability per cluster",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="black"
        )
        n_items_per_line = 5
        x_start = 0.02
        x_end = 0.98
        slot_w = (x_end - x_start) / n_items_per_line
        y_start = -0.16
        line_step = 0.065
        square_w = 0.018
        square_h = 0.028
        gap_after_square = 0.008

        for i, (_, row) in enumerate(rows.iterrows()):
            cluster_id = str(row["cluster"])
            token = f"{cluster_id}: {float(row['mean_instability']):.3f}"
            col_idx = i % n_items_per_line
            row_idx = i // n_items_per_line
            x_pos = x_start + col_idx * slot_w
            y_pos = y_start - row_idx * line_step
            color_box = Rectangle(
                (x_pos, y_pos - square_h + 0.002),
                square_w,
                square_h,
                transform=ax.transAxes,
                facecolor=cluster_color_map.get(cluster_id, "gray"),
                edgecolor="black",
                linewidth=0.3,
                clip_on=False,
            )
            ax.add_patch(color_box)
            txt = ax.text(
                x_pos + square_w + gap_after_square,
                y_pos,
                token,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="black",
            )

    combined_stem = output_dir / "overall_16_subplots_combined_vector"
    fig.savefig(f"{combined_stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{combined_stem}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{combined_stem}.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


save_combined_overall_figure_vector(pathh, dpi=600)