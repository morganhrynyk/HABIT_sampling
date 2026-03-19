from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEFAULT_CONFIG_PATH = Path(__file__).with_name("select_diverse_segments_config.txt")


def parse_config_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid config line {line_no} in {path}: {raw_line}")
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def parse_feature_cols(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_defaults(config: dict[str, str]) -> dict[str, object]:
    mode = config.get("mode", "hybrid")
    n_diverse = int(config.get("n_diverse", "50"))
    n_dense_value = config.get("n_dense")
    if n_dense_value is not None:
        n_dense = int(n_dense_value)
        n_select = n_diverse + n_dense if mode == "hybrid" else n_diverse
    else:
        n_select = int(config.get("n_select", "100"))
        n_dense = max(0, n_select - n_diverse) if mode == "hybrid" else 0

    return {
        "input_csv": Path(
            config.get(
                "input_csv",
                r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method\segment_features_for_clustering.csv",
            )
        ),
        "output_dir": Path(
            config.get(
                "output_dir",
                r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\diverse_method",
            )
        ),
        "feature_cols": parse_feature_cols(
            config.get(
                "feature_cols",
                "brightness_mean_z,rg_contrast_mean_z,texture_mean_z,ndvi_mean_z,ndwi_mean_z",
            )
        ),
        "n_select": n_select,
        "mode": mode,
        "n_diverse": n_diverse,
        "n_dense": n_dense,
        "dense_k": int(config.get("dense_k", "10")),
        "dense_candidate_percentile": float(config.get("dense_candidate_percentile", "0.85")),
        "dense_min_distance_quantile": float(config.get("dense_min_distance_quantile", "0.5")),
        "output_vector": Path(config.get("output_vector", "diverse_segments.shp")),
    }


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to key=value config file for select_diverse_segments.py.",
    )
    bootstrap_args, remaining_argv = bootstrap.parse_known_args()
    defaults = build_defaults(parse_config_file(bootstrap_args.config))

    parser = argparse.ArgumentParser(
        description=(
            "Select segments for manual annotation using feature-space coverage, "
            "density-aware sampling, or a hybrid of both."
        ),
        parents=[bootstrap],
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=defaults["input_csv"],
        help="CSV created by extract_segment_features.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output_dir"],
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=defaults["feature_cols"],
        help="Normalized feature columns used for diversity sampling.",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=defaults["n_select"],
        help="Total number of segments to select.",
    )
    parser.add_argument(
        "--mode",
        choices=["diverse", "hybrid"],
        default=defaults["mode"],
        help="Selection strategy. 'hybrid' combines coverage and dense-core sampling.",
    )
    parser.add_argument(
        "--n-diverse",
        type=int,
        default=defaults["n_diverse"],
        help="Number of coverage-focused selections when --mode hybrid.",
    )
    parser.add_argument(
        "--n-dense",
        type=int,
        default=defaults["n_dense"],
        help="Number of dense-core selections when --mode hybrid.",
    )
    parser.add_argument(
        "--dense-k",
        type=int,
        default=defaults["dense_k"],
        help="Number of neighbors used to estimate local feature-space density.",
    )
    parser.add_argument(
        "--dense-candidate-percentile",
        type=float,
        default=defaults["dense_candidate_percentile"],
        help=(
            "Only consider the top density fraction for dense-core sampling, "
            "expressed as a percentile between 0 and 1."
        ),
    )
    parser.add_argument(
        "--dense-min-distance-quantile",
        type=float,
        default=defaults["dense_min_distance_quantile"],
        help=(
            "Minimum spacing for dense selections, as a quantile of the population's "
            "nearest-neighbor distance distribution."
        ),
    )
    parser.add_argument(
        "--output-vector",
        type=Path,
        default=defaults["output_vector"],
        help=(
            "Optional vector output for selected segments. Relative paths are resolved "
            "under --output-dir. Use .shp or .gpkg."
        ),
    )
    return parser.parse_args(remaining_argv)


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


def choose_diverse_subset(
    x: np.ndarray, n_select: int, excluded_idx: set[int] | None = None
) -> tuple[list[int], list[float]]:
    n_rows = len(x)
    if n_rows == 0:
        return [], []

    excluded_idx = excluded_idx or set()
    n_select = min(n_select, n_rows)
    centroid = np.nanmean(x, axis=0)
    dists_to_centroid = np.sqrt(np.sum((x - centroid) ** 2, axis=1))
    order_to_centroid = np.argsort(dists_to_centroid)
    first_idx = None
    for idx in order_to_centroid:
        idx_i = int(idx)
        if idx_i not in excluded_idx:
            first_idx = idx_i
            break
    if first_idx is None:
        return [], []

    selected = [first_idx]
    selection_distances = [0.0]
    min_dists = np.sqrt(np.sum((x - x[first_idx]) ** 2, axis=1))
    min_dists[first_idx] = -np.inf
    if excluded_idx:
        min_dists[list(excluded_idx)] = -np.inf

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


def estimate_local_density(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(x)
    if n_rows <= 1:
        return np.ones(n_rows, dtype=float), np.zeros(n_rows, dtype=float)

    k_eff = max(1, min(k, n_rows - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(x)
    distances, _ = nn.kneighbors(x)
    mean_knn_dist = np.mean(distances[:, 1 : k_eff + 1], axis=1)
    density = 1.0 / np.maximum(mean_knn_dist, 1e-12)
    nearest_neighbor = distances[:, 1]
    return density, nearest_neighbor


def choose_dense_subset(
    x: np.ndarray,
    density: np.ndarray,
    n_select: int,
    excluded_idx: set[int],
    candidate_percentile: float,
    min_spacing: float,
) -> tuple[list[int], list[float]]:
    n_rows = len(x)
    if n_rows == 0 or n_select <= 0:
        return [], []

    percentile = float(np.clip(candidate_percentile, 0.0, 1.0))
    density_threshold = float(np.quantile(density, percentile))
    candidate_mask = density >= density_threshold
    if excluded_idx:
        candidate_mask[list(excluded_idx)] = False

    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) == 0:
        candidate_idx = np.array(
            [idx for idx in np.argsort(-density) if idx not in excluded_idx], dtype=int
        )
    if len(candidate_idx) == 0:
        return [], []

    candidate_x = x[candidate_idx]
    candidate_density = density[candidate_idx]
    start_local = int(np.argmax(candidate_density))
    start_idx = int(candidate_idx[start_local])

    selected: list[int] = []
    spacing_at_selection: list[float] = []
    selected.append(start_idx)
    reference_idx = list(excluded_idx) + [start_idx]
    if excluded_idx:
        reference_points = x[list(excluded_idx)]
        dists = np.sqrt(np.sum((reference_points - x[start_idx]) ** 2, axis=1))
        spacing_at_selection.append(float(np.min(dists)))
    else:
        spacing_at_selection.append(0.0)

    min_dists = np.sqrt(np.sum((candidate_x - x[start_idx]) ** 2, axis=1))
    min_dists[start_local] = -np.inf

    if excluded_idx:
        excluded_points = x[list(excluded_idx)]
        dists_to_excluded = np.sqrt(
            np.sum((candidate_x[:, None, :] - excluded_points[None, :, :]) ** 2, axis=2)
        )
        min_dists = np.minimum(min_dists, np.min(dists_to_excluded, axis=1))
        min_dists[start_local] = -np.inf

    while len(selected) < n_select:
        if min_spacing > 0:
            valid_mask = min_dists >= min_spacing
        else:
            valid_mask = min_dists > -np.inf
        if not np.any(valid_mask):
            break

        next_local = int(np.argmax(np.where(valid_mask, min_dists, -np.inf)))
        next_idx = int(candidate_idx[next_local])
        selected.append(next_idx)
        spacing_at_selection.append(float(min_dists[next_local]))

        candidate_dists = np.sqrt(np.sum((candidate_x - x[next_idx]) ** 2, axis=1))
        min_dists = np.minimum(min_dists, candidate_dists)
        min_dists[next_local] = -np.inf

    return selected, spacing_at_selection


def choose_hybrid_subset(
    x: np.ndarray,
    n_select: int,
    n_diverse: int,
    dense_k: int,
    dense_candidate_percentile: float,
    dense_min_distance_quantile: float,
) -> pd.DataFrame:
    n_rows = len(x)
    if n_rows == 0:
        return pd.DataFrame(
            columns=[
                "source_index",
                "selection_order",
                "selection_component",
                "selection_metric",
            ]
        )

    n_select = min(n_select, n_rows)
    if n_select <= 0:
        return pd.DataFrame(
            columns=[
                "source_index",
                "selection_order",
                "selection_component",
                "selection_metric",
            ]
        )

    if n_select == n_rows:
        records = []
        for i in range(n_rows):
            records.append(
                {
                    "source_index": i,
                    "selection_order": i + 1,
                    "selection_component": "all",
                    "selection_metric": 0.0,
                }
            )
        return pd.DataFrame.from_records(records)

    if n_select == 1 or n_diverse >= n_select:
        diverse_idx, diverse_metric = choose_diverse_subset(x, n_select=n_select)
        records = []
        for order, (idx, metric) in enumerate(zip(diverse_idx, diverse_metric), start=1):
            records.append(
                {
                    "source_index": idx,
                    "selection_order": order,
                    "selection_component": "coverage",
                    "selection_metric": float(metric),
                }
            )
        return pd.DataFrame.from_records(records)

    n_diverse = max(1, min(n_diverse, n_select))
    n_dense = n_select - n_diverse

    diverse_idx, diverse_metric = choose_diverse_subset(x, n_select=n_diverse)
    excluded = set(diverse_idx)

    density, nearest_neighbor = estimate_local_density(x, dense_k)
    quantile = float(np.clip(dense_min_distance_quantile, 0.0, 1.0))
    min_spacing = float(np.quantile(nearest_neighbor, quantile)) if len(nearest_neighbor) else 0.0
    dense_idx, dense_metric = choose_dense_subset(
        x=x,
        density=density,
        n_select=n_dense,
        excluded_idx=excluded,
        candidate_percentile=dense_candidate_percentile,
        min_spacing=min_spacing,
    )

    records = []
    order = 1
    for idx, metric in zip(diverse_idx, diverse_metric):
        records.append(
            {
                "source_index": idx,
                "selection_order": order,
                "selection_component": "coverage",
                "selection_metric": float(metric),
            }
        )
        order += 1
    for idx, metric in zip(dense_idx, dense_metric):
        records.append(
            {
                "source_index": idx,
                "selection_order": order,
                "selection_component": "dense_core",
                "selection_metric": float(metric),
            }
        )
        order += 1

    return pd.DataFrame.from_records(records)


def resolve_output_vector_path(output_dir: Path, output_vector: Path) -> Path:
    if output_vector.is_absolute():
        return output_vector
    return output_dir / output_vector


def export_selected_geometries(selected_df: pd.DataFrame, out_path: Path) -> Path | None:
    if selected_df.empty:
        return None
    import geopandas as gpd

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
    selected_idx: np.ndarray,
    selection_order: np.ndarray,
    selection_component: np.ndarray,
    out_path: Path,
) -> None:
    if len(x) == 0:
        return
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

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

    component_styles = {
        "coverage": {"color": "#D95F02", "label": "Coverage selections"},
        "dense_core": {"color": "#1B9E77", "label": "Dense-core selections"},
        "all": {"color": "#7570B3", "label": "Selected segments"},
    }
    for component in pd.unique(selection_component):
        style = component_styles.get(
            str(component),
            {"color": "#4C78A8", "label": str(component)},
        )
        mask = selection_component == component
        ax.scatter(
            selected_pcs[mask, 0],
            selected_pcs[mask, 1],
            s=46,
            alpha=0.95,
            color=style["color"],
            edgecolors="black",
            linewidths=0.35,
            label=style["label"],
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
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "hybrid":
        args.n_select = args.n_diverse + args.n_dense
    else:
        args.n_select = args.n_diverse if args.n_diverse > 0 else args.n_select

    df = pd.read_csv(args.input_csv)
    validate_columns(df, args.feature_cols)
    x = build_feature_matrix(df, args.feature_cols)

    if args.mode == "diverse":
        selection_table = choose_hybrid_subset(
            x=x,
            n_select=args.n_select,
            n_diverse=args.n_select,
            dense_k=args.dense_k,
            dense_candidate_percentile=args.dense_candidate_percentile,
            dense_min_distance_quantile=args.dense_min_distance_quantile,
        )
    else:
        selection_table = choose_hybrid_subset(
            x=x,
            n_select=args.n_select,
            n_diverse=args.n_diverse,
            dense_k=args.dense_k,
            dense_candidate_percentile=args.dense_candidate_percentile,
            dense_min_distance_quantile=args.dense_min_distance_quantile,
        )

    selected_idx = selection_table["source_index"].to_numpy(dtype=int)
    selected_df = df.iloc[selected_idx].copy().reset_index(drop=True)
    selected_df["selection_order"] = selection_table["selection_order"].to_numpy(dtype=int)
    selected_df["selection_component"] = selection_table["selection_component"].to_numpy()
    selected_df["selection_metric"] = selection_table["selection_metric"].to_numpy(dtype=float)

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
        selected_df["selection_component"].to_numpy(),
        figure_path,
    )

    component_counts = selected_df["selection_component"].value_counts()
    coverage_count = int(component_counts.get("coverage", 0))
    dense_count = int(component_counts.get("dense_core", 0))
    summary_lines = [
        f"input_csv: {args.input_csv}",
        f"total_segments_available: {len(df)}",
        f"segments_selected: {len(selected_df)}",
        f"feature_columns: {', '.join(args.feature_cols)}",
        f"selection_method: {args.mode}",
        f"coverage_segments_selected: {coverage_count}",
        f"dense_core_segments_selected: {dense_count}",
        f"n_diverse_requested: {args.n_diverse}",
        f"n_dense_requested: {args.n_dense}",
        f"dense_k: {args.dense_k}",
        f"dense_candidate_percentile: {args.dense_candidate_percentile}",
        f"dense_min_distance_quantile: {args.dense_min_distance_quantile}",
    ]
    summary_path = args.output_dir / "diverse_segments_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Total segments available: {len(df)}")
    print(f"Segments selected: {len(selected_df)}")
    print(f"Selection method: {args.mode}")
    print(f"Coverage selections: {coverage_count}")
    print(f"Dense-core selections: {dense_count}")
    print(f"Wrote: {out_csv.resolve()}")
    if vector_path is not None:
        print(f"Wrote: {vector_path.resolve()}")
    print(f"Wrote: {figure_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
