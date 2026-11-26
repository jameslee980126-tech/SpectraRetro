"""Redraw Figure 2b–e or Figure S1 from the distributed observations."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTORS = ROOT / "datasets/supplementary_data"
FIGURE_DATA = ROOT / "results/supporting_information/source_data_figures"


def plot_properties():
    metrics = [("LogP", "LogP", "#337c9b"), ("TPSA", "TPSA (Å²)", "#378678"),
               ("FractionCSP3", "Fraction of sp³ carbon", "#8264a2")]
    reactions = pd.read_csv(DESCRIPTORS / "figure3_reaction_weighted_descriptors.csv",
                            usecols=[name for name, _, _ in metrics])
    products = pd.read_csv(DESCRIPTORS / "figure3_unique_product_descriptors.csv",
                           usecols=["heavy_atoms", "rotatable_bonds", "fully_specified"])
    products = products.loc[products["fully_specified"]]
    if len(reactions) != 92791 or len(products) != 73521:
        raise ValueError("The Figure 2 descriptor populations differ from the published populations.")
    summary = pd.read_csv(FIGURE_DATA / "figure3_property_summary.csv").set_index("descriptor")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), layout="constrained")
    rng = np.random.default_rng(20261011)
    for ax, (metric, label, color) in zip(axes[:3], metrics):
        values = reactions[metric].to_numpy()
        density, edges = np.histogram(values, bins=200, density=True)
        density = gaussian_filter1d(density, 1.1, mode="nearest")
        density *= 0.53 / density.max()
        centers = (edges[1:] + edges[:-1]) / 2
        ax.fill_between(centers, density, color=color, alpha=0.24)
        ax.plot(centers, density, color=color, lw=1.2)
        q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        np.testing.assert_allclose(q, summary.loc[metric, ["q05", "q25", "median", "q75", "q95"]])
        ax.hlines(-0.16, q[0], q[4], color=color, lw=1)
        ax.hlines(-0.16, q[1], q[3], color=color, lw=6)
        ax.scatter(q[2], -0.16, color="white", edgecolor=color, s=20, zorder=3)
        sample = rng.choice(len(values), size=350, replace=False)
        ax.scatter(values[sample], rng.uniform(-0.34, -0.23, len(sample)), s=3, alpha=0.2,
                   linewidths=0, color=color)
        ax.set(xlim=(edges[0], edges[-1]), ylim=(-0.39, 0.63), yticks=[], xlabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.spines[["top", "right", "left"]].set_visible(False)
    ax = axes[-1]
    hexes = ax.hexbin(products.heavy_atoms, products.rotatable_bonds, gridsize=(37, 18),
                      extent=(0, 85, 0, 40), mincnt=1, cmap="viridis", norm=LogNorm(vmin=1),
                      linewidths=0)
    if int(hexes.get_array().sum()) != len(products):
        raise ValueError("A product fell outside the published hexbin extent.")
    ax.set(xlim=(0, 85), ylim=(0, 40), xlabel="Heavy atoms", ylabel="Rotatable bonds")
    fig.colorbar(hexes, ax=ax, label="Products per bin")
    for panel, axis in zip("bcde", axes):
        axis.set_title(panel, loc="left", fontweight="bold")
    return fig


def plot_center_gains():
    group = ["class", "similarity_group"]
    paired = pd.read_csv(FIGURE_DATA / "figure6_paired_reaction_results.csv",
                         usecols=[*group, "dataset_index", "fspec_rank", "no_spectral_rank"])
    if len(paired) != 9250 or paired.dataset_index.nunique() != 9250:
        raise ValueError("Figure S1 requires 9,250 distinct test reactions.")
    paired["difference"] = (paired.fspec_rank.eq(1).astype(int)
                            - paired.no_spectral_rank.eq(1).astype(int))
    cells = paired.groupby(group)["difference"].agg(["mean", "size"])
    summary = pd.read_csv(FIGURE_DATA / "figure6_family_similarity_summary.csv").set_index(group)
    cells = cells.loc[summary.index]
    np.testing.assert_array_equal(cells["size"], summary["n"])
    np.testing.assert_allclose(100 * cells["mean"], summary["gain_percentage_points"])
    columns = ["low", "medium", "high"]
    gains = 100 * cells["mean"].unstack()[columns]
    counts = cells["size"].unstack()[columns]
    labels = summary["family"].groupby(level="class").first().loc[gains.index]
    fig, ax = plt.subplots(figsize=(7.6, 5.6), layout="constrained")
    image = ax.imshow(gains, vmin=-55, vmax=55, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(3), [name.capitalize() for name in columns])
    ax.set_yticks(range(len(labels)), [name.removeprefix("C-H ") for name in labels])
    ax.set_xlabel("Local similarity")
    for (i, j), gain in np.ndenumerate(gains.to_numpy()):
        ax.text(j, i, f"{gain:+.2f}\nn = {counts.iat[i, j]:,}", ha="center", va="center",
                color="white" if abs(gain) >= 32 else "black")
    fig.colorbar(image, ax=ax, label="Change in center Top-1 accuracy (percentage points)")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["2", "S1"], required=True,
                        help="2 for the descriptor panels (b–e) or S1 for the reaction-center gains.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    fig = plot_properties() if args.figure == "2" else plot_center_gains()
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / f"figure_{args.figure}.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
