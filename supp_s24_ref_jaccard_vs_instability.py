import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
tableau_20 = plt.cm.tab20.colors

# instability data
df = pd.read_csv("./data/fig5/clusterwise_metrics_by_number_of_runs_new.csv")

metric = 'clusterwise jaccard'

# Restrict to some number of runs
nruns = 50
df = df[df['iterations'] == nruns]
df = df[df['dataset-sample'] != 'ov_ffpe_full']
df = df[df['dataset-sample'] != 'coad_ffpe_full']
df = df[df['dataset-sample'] != 'Visium_HD_Human_Colon_Cancer']

mthds = ["leiden", "louvain", "bayesspace", "graphst", 
         "stagate", "spicemix", "sedr", "sedr_mclust"]
mnames = ["Leiden", "Louvain",  "BayesSpace", "GraphST",
          "STAGATE", "SpiceMix",  "SEDR", "SEDR (mclust)"]

datas = [
    "dlpfc",
    "mouse_brain",
    "mouse_brain_cerebellum",
    "human_breast_cancer",
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "ov_ffpe",
    "coad_ffpe",
]

dnames = [
    "DLPFC",
    "MB",
    "MBC",
    "HBC",
    "CRC",
    "OV",
    "COAD",
]

colors = [
    tableau_20[14],  # DLPFC
    tableau_20[8],   # MB
    tableau_20[0],   # MBC
    tableau_20[12],  # HBC
    tableau_20[2],   # CRC
    tableau_20[18],  # OV
    tableau_20[16],  # COAD
]

# Now make exciting figure
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

axes_flat = axes.flatten()


for axi, ax in enumerate(axes_flat):
	print(axi)
	print(mthds[axi])
	print(mnames[axi])
	if axi == 8: 
		break
	else:
		mthd = mthds[axi]
		mname = mnames[axi]
		xdf = df[df['method'] == mthd]

		ax.set_title(mnames[axi], fontsize=14, fontweight='bold')	
		if axi > 3:
			ax.set_xlabel("Cluster Instability", fontsize=12)
		if axi == 0 or axi==4:
			ax.set_ylabel("Jaccard w.r.t. Ref.", fontsize=12)

		ax.set_ylim(0, 1.0)
		ax.set_xlim(0, 0.7)
		ax.set_xticks([0, 0.2, 0.4, 0.6])
		#ax.set_yticks([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])

		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)


		# --- CREATE THE DASHED CURVED-EDGE BOX ---

		# Define the anchor (bottom-left corner) and dimensions
		# You specify these in your actual data coordinates.
		# box_x = 0.32      # Left edge coordinate
		box_x = 0.22        # Left edge coordinate
		box_y = 0.0         # Bottom edge coordinate
		box_width = 30      # Width in data units
		box_height = 0.48   # Height in data units

		# Create the FancyBboxPatch
		# 'boxstyle="round,pad=0.02"' creates the curved corners.
		highlight_box = patches.FancyBboxPatch(
			(box_x, box_y),            # (x, y) anchor
			box_width,                 # width
			box_height,                # height
			boxstyle="round,pad=0.02", # Key: "round" corner style
			ec="black",                # Edge Color
			fc=(0.5, 0.5, 0.5, 0.2),   # Face Color (Grey, translucent alpha=0.2)
			linestyle=':',             # Key: Dashed border
			linewidth=2,               # Border thickness
			zorder=10                  # Ensure it sits *on top* of the data
		)

		# Add the patch to the axis
		ax.add_patch(highlight_box)

		# --- CREATE THE DASHED CURVED-EDGE BOX ---

		# Define the anchor (bottom-left corner) and dimensions
		# You specify these in your actual data coordinates.
		#box_x = 0.32      # Left edge coordinate
		box_x = 0.42       # Left edge coordinate
		box_y = 0.0        # Bottom edge coordinate
		box_width = 30     # Width in data units
		box_height = 0.48  # Height in data units

		# Create the FancyBboxPatch
		# 'boxstyle="round,pad=0.02"' creates the curved corners.
		highlight_box = patches.FancyBboxPatch(
			(box_x, box_y),            # (x, y) anchor
			box_width,                 # width
			box_height,                # height
			boxstyle="round,pad=0.02", # Key: "round" corner style
			ec="black",                # Edge Color
			fc=(0.5, 0.5, 0.5, 0.2),   # Face Color (Grey, translucent alpha=0.2)
			linestyle='--',            # Key: Dashed border
			linewidth=2,               # Border thickness
			zorder=10                  # Ensure it sits *on top* of the data
		)

		# Add the patch to the axis
		ax.add_patch(highlight_box)

		# Add fit line
		xs = xdf['clusterwise mean instability'].to_numpy(dtype=float)
		ys = xdf[metric].to_numpy(dtype=float)

		valid = (
			np.isfinite(xs)
			& np.isfinite(ys)
		)

		xx = xs[valid]
		yy = ys[valid]

		can_correlate = (
			len(xx) >= 2
			and np.ptp(xx) > 0
			and np.ptp(yy) > 0
		)

		if can_correlate:
			r, p = stats.pearsonr(xx, yy)
			r_sq = r ** 2

			print(
				" %s : r=%f r2=%f, p=%f"
				% (mthds[axi], r, r_sq, p)
			)

			slope, intercept = np.polyfit(xx, yy, 1)

			fit_x = np.linspace(0, 0.7, 100)
			fit_y = slope * fit_x + intercept

			ax.plot(
				fit_x,
				fit_y,
				color='black',
				linestyle='--',
				linewidth=2,
			)

			if p < 0.001:
				info = (
					f"$r = {r:.2f}$\n"
					f"$r^2 = {r_sq:.2f}$\n"
					f"$p < 0.001$"
				)
			else:
				info = (
					f"$r = {r:.2f}$\n"
					f"$r^2 = {r_sq:.2f}$\n"
					f"$p = {p:.3f}$"
				)

			ax.text(
				1.0,
				0.95,
				info,
				transform=ax.transAxes,
				ha='right',
				va='top',
				fontsize=10,
				bbox=dict(
					boxstyle="round,pad=0.3",
					fc="white",
					alpha=0.7,
					ec="gray",
				),
			)

		else:
			print(
				f"{mthds[axi]}: insufficient/non-varying "
				"finite data for Pearson correlation"
			)

		# Plot actual data, colored by sample

		for j, dataset in enumerate(datas):
			if dataset == "dlpfc":
				ydf = xdf[
					xdf["dataset-sample"]
					.astype(str)
					.str.startswith("dlpfc_", na=False)
				]
			else:
				ydf = xdf[
					xdf["dataset-sample"] == dataset
				]

			xx = ydf['clusterwise mean instability'].to_numpy(dtype=float)
			yy = ydf[metric].to_numpy(dtype=float)

			valid = (
				np.isfinite(xx)
				& np.isfinite(yy)
			)

			xx = xx[valid]
			yy = yy[valid]

			if len(xx) > 0:
				if np.max(xx) > 0.7:
					print("oh no")

			if axi == 6:
				ax.scatter(xx, yy, color=colors[j], 
							s=20, label=dnames[j], alpha=0.8)
			else:
				ax.scatter(xx, yy, color=colors[j], 
							s=20, alpha=0.8)

		if axi == 6:
			print("adding legend")
			fig.legend(loc='upper center',
    			   bbox_to_anchor=(0.5, 0.08),
    			   bbox_transform=fig.transFigure,
    			   fontsize=14,
    			   ncol=7,
    			   markerscale=2.5,
    			   frameon=False,)

plt.tight_layout()
fig.subplots_adjust(
    bottom=0.15
)

plt.savefig('./data/supp/s24_ref_jaccard_vs_instability/s24_cluster_jaccard_wrt_ref_vs_instability.pdf', dpi=600)
plt.savefig('./data/supp/s24_ref_jaccard_vs_instability/s24_cluster_jaccard_wrt_ref_vs_instability.png', dpi=600)


