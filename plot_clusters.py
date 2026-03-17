from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize clustered segments with PCA and feature-pair plots."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method\segment_features_with_clusters.csv"
        ),
        help="CSV produced by cluster_segments.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method"
        ),
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=[
            "brightness_mean_z",
            "rg_contrast_mean_z",
            "texture_mean_z",
            "ndvi_mean_z",
            "segment_area_m2_z",
        ],
        help="Feature columns to visualize.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Max points to plot (random sample if larger).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def sample_df(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed).reset_index(drop=True)


def validate_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    required = ["cluster_id"] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def plot_pca(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    x = df[feature_cols].to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=df["cluster_id"].to_numpy(),
        s=6,
        alpha=0.7,
        cmap="tab20",
        linewidths=0,
    )
    ax.set_title("Clusters in PCA Space")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cluster_id")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_pairs(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    pairs = list(combinations(feature_cols, 2))
    n = len(pairs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (xcol, ycol) in zip(axes, pairs):
        ax.scatter(
            df[xcol].to_numpy(),
            df[ycol].to_numpy(),
            c=df["cluster_id"].to_numpy(),
            s=5,
            alpha=0.6,
            cmap="tab20",
            linewidths=0,
        )
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{xcol} vs {ycol}")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cluster_sizes(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index.astype(int), counts.values, color="#4C78A8")
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("n_segments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    validate_columns(df, args.feature_cols)
    df_plot = sample_df(df, args.max_points, args.seed)

    pca_png = args.output_dir / "cluster_pca_scatter.png"
    pairs_png = args.output_dir / "cluster_feature_pairs.png"
    sizes_png = args.output_dir / "cluster_sizes.png"

    plot_pca(df_plot, args.feature_cols, pca_png)
    plot_feature_pairs(df_plot, args.feature_cols, pairs_png)
    plot_cluster_sizes(df, sizes_png)

    print(f"Rows in source: {len(df)}")
    print(f"Rows plotted: {len(df_plot)}")
    print(f"Wrote: {pca_png.resolve()}")
    print(f"Wrote: {pairs_png.resolve()}")
    print(f"Wrote: {sizes_png.resolve()}")


if __name__ == "__main__":
    main()
