from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a feature-space-diverse set of segments for manual annotation "
            "using greedy k-center sampling."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method\segment_features_for_clustering.csv"
        ),
        help="CSV created by extract_segment_features.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method"
        ),
        help="Directory for outputs.",
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
        help="Normalized feature columns used for diversity sampling.",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=100,
        help="Number of segments to select.",
    )
    parser.add_argument(
        "--output-vector",
        type=Path,
        default=Path("diverse_segments.shp"),
        help=(
            "Optional vector output for selected segments. Relative paths are resolved "
            "under --output-dir. Use .shp or .gpkg."
        ),
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    required = ["segment_id", "source_shp"] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = df[feature_cols].to_numpy(dtype=float)
    if np.isnan(x).any():
        medians = np.nanmedian(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = medians[inds[1]]
    return x


def infer_source_row_index(row: pd.Series) -> int | None:
    if "source_row_index" in row.index and pd.notna(row["source_row_index"]):
        return int(row["source_row_index"])

    segment_id = str(row.get("segment_id", ""))
    tail = segment_id.rsplit("_", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


def choose_diverse_subset(x: np.ndarray, n_select: int) -> tuple[list[int], list[float]]:
    n_rows = len(x)
    if n_rows == 0:
        return [], []

    n_select = min(n_select, n_rows)
    centroid = np.nanmean(x, axis=0)
    dists_to_centroid = np.sqrt(np.sum((x - centroid) ** 2, axis=1))
    first_idx = int(np.argmin(dists_to_centroid))

    selected = [first_idx]
    selection_distances = [0.0]
    min_dists = np.sqrt(np.sum((x - x[first_idx]) ** 2, axis=1))
    min_dists[first_idx] = -np.inf

    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] == -np.inf:
            break
        selected.append(next_idx)
        selection_distances.append(float(min_dists[next_idx]))
        candidate_dists = np.sqrt(np.sum((x - x[next_idx]) ** 2, axis=1))
        min_dists = np.minimum(min_dists, candidate_dists)
        min_dists[selected] = -np.inf

    return selected, selection_distances


def resolve_output_vector_path(output_dir: Path, output_vector: Path) -> Path:
    if output_vector.is_absolute():
        return output_vector
    return output_dir / output_vector


def export_selected_geometries(selected_df: pd.DataFrame, out_path: Path) -> Path | None:
    if selected_df.empty:
        return None

    selected = selected_df.copy()
    selected["source_row_index_resolved"] = selected.apply(infer_source_row_index, axis=1)
    if selected["source_row_index_resolved"].isna().any():
        missing_count = int(selected["source_row_index_resolved"].isna().sum())
        raise ValueError(
            "Cannot export vector output because some selected rows cannot be mapped "
            f"back to source polygons ({missing_count} row(s) missing source_row_index)."
        )

    vector_frames: list[gpd.GeoDataFrame] = []
    for source_shp, group in selected.groupby("source_shp", sort=False):
        source_gdf = gpd.read_file(source_shp).reset_index(drop=True)
        row_indexes = group["source_row_index_resolved"].astype(int).tolist()
        source_subset = source_gdf.iloc[row_indexes].copy().reset_index(drop=True)
        attrs = group.drop(columns=["source_row_index_resolved"]).reset_index(drop=True)
        merged = pd.concat([attrs, source_subset[["geometry"]]], axis=1)
        vector_frames.append(gpd.GeoDataFrame(merged, geometry="geometry", crs=source_gdf.crs))

    result = pd.concat(vector_frames, ignore_index=True)
    result_gdf = gpd.GeoDataFrame(result, geometry="geometry", crs=vector_frames[0].crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_gdf.to_file(out_path)
    return out_path


def plot_selected_feature_space(
    x: np.ndarray,
    selected_idx: list[int],
    selection_order: np.ndarray,
    out_path: Path,
) -> None:
    if len(x) == 0:
        return

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(x)
    selected_pcs = pcs[selected_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        pcs[:, 0],
        pcs[:, 1],
        s=8,
        alpha=0.18,
        color="#9AA5B1",
        linewidths=0,
        label="All segments",
    )
    scatter = ax.scatter(
        selected_pcs[:, 0],
        selected_pcs[:, 1],
        c=selection_order,
        s=42,
        alpha=0.95,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
        label="Selected segments",
    )

    for i, (x0, y0) in enumerate(selected_pcs):
        ax.text(
            x0,
            y0,
            str(int(selection_order[i])),
            fontsize=6,
            ha="center",
            va="center",
            color="black",
        )

    ax.set_title("Selected Segments in Feature Space (PCA)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Selection order")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    validate_columns(df, args.feature_cols)
    x = build_feature_matrix(df, args.feature_cols)

    selected_idx, selection_distances = choose_diverse_subset(x, args.n_select)
    selected_df = df.iloc[selected_idx].copy().reset_index(drop=True)
    selected_df["selection_order"] = np.arange(1, len(selected_df) + 1)
    selected_df["distance_to_nearest_selected_when_chosen"] = selection_distances

    out_csv = args.output_dir / "diverse_segments_for_annotation.csv"
    selected_df.to_csv(out_csv, index=False)

    vector_path = export_selected_geometries(
        selected_df, resolve_output_vector_path(args.output_dir, args.output_vector)
    )
    figure_path = args.output_dir / "diverse_segments_feature_space.png"
    plot_selected_feature_space(
        x,
        selected_idx,
        selected_df["selection_order"].to_numpy(dtype=int),
        figure_path,
    )

    coverage_radius = (
        float(max(selection_distances[1:])) if len(selection_distances) > 1 else 0.0
    )
    summary_lines = [
        f"input_csv: {args.input_csv}",
        f"total_segments_available: {len(df)}",
        f"segments_selected: {len(selected_df)}",
        f"feature_columns: {', '.join(args.feature_cols)}",
        "selection_method: greedy_k_center_feature_space",
        "starting_point: global_medoid_by_feature_centroid_distance",
        f"coverage_radius_feature_space: {coverage_radius:.6f}",
    ]
    summary_path = args.output_dir / "diverse_segments_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Total segments available: {len(df)}")
    print(f"Segments selected: {len(selected_df)}")
    print("Selection method: greedy_k_center_feature_space")
    print(f"Wrote: {out_csv.resolve()}")
    if vector_path is not None:
        print(f"Wrote: {vector_path.resolve()}")
    print(f"Wrote: {figure_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
