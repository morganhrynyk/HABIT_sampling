from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
    raise SystemExit(
        "Missing dependency: "
        f"{missing_pkg}. Install required packages first, for example:\n"
        "pip install geopandas scikit-learn pandas numpy"
    ) from exc


EXCLUDE_COLUMNS = {
    "geometry",
    "fid",
    "id",
    "objectid",
    "shape_leng",
    "shape_area",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster segmented polygons by numeric attributes and sample a stratified "
            "set for manual annotation."
        )
    )
    parser.add_argument(
        "--polygons-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\partial\felzenszwalb_with_masks_polygons_250m_class_cols"
        ),
        help="Directory containing segmentation shapefiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--target-sample-size",
        type=int,
        default=500,
        help="Total number of segments to sample for annotation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering and sampling.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=8,
        help="Minimum clusters to evaluate during k selection.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=40,
        help="Maximum clusters to evaluate during k selection.",
    )
    parser.add_argument(
        "--max-silhouette-samples",
        type=int,
        default=20000,
        help="Cap rows used for silhouette scoring to keep runtime manageable.",
    )
    parser.add_argument(
        "--feature-prefixes",
        nargs="*",
        default=[],
        help=(
            "Optional list of prefixes to restrict feature columns "
            "(e.g. --feature-prefixes ndvi band tex)."
        ),
    )
    return parser.parse_args()


def list_shapefiles(polygons_dir: Path) -> list[Path]:
    shp_files = sorted(polygons_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No shapefiles found in {polygons_dir}")
    return shp_files


def load_polygons(shp_files: Iterable[Path]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        gdf["chip_name"] = shp.stem
        gdf["source_shp"] = str(shp)
        gdf["segment_uid"] = gdf["chip_name"] + "_" + gdf.index.astype(str)
        frames.append(gdf)
    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)


def select_numeric_features(
    gdf: gpd.GeoDataFrame, feature_prefixes: list[str]
) -> list[str]:
    numeric_cols = []
    for col in gdf.columns:
        if col.lower() in EXCLUDE_COLUMNS:
            continue
        if col in {"chip_name", "source_shp", "segment_uid"}:
            continue
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            continue
        if feature_prefixes and not any(col.lower().startswith(p.lower()) for p in feature_prefixes):
            continue
        non_null_ratio = gdf[col].notna().mean()
        nunique = gdf[col].nunique(dropna=True)
        if non_null_ratio < 0.2 or nunique < 2:
            continue
        numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError(
            "No usable numeric feature columns were found in shapefile attributes. "
            "Check your inputs or use --feature-prefixes to target valid feature fields."
        )
    return numeric_cols


def build_feature_matrix(gdf: gpd.GeoDataFrame, feature_cols: list[str]) -> np.ndarray:
    return gdf[feature_cols].to_numpy(dtype=float)


def preprocess_features(features: np.ndarray, seed: int) -> np.ndarray:
    n_features = features.shape[1]
    pca_components = min(n_features, 50)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=seed)),
        ]
    )
    transformed = pipeline.fit_transform(features)

    explained = np.cumsum(pipeline.named_steps["pca"].explained_variance_ratio_)
    keep = int(np.searchsorted(explained, 0.95) + 1)
    keep = max(2, min(keep, transformed.shape[1]))
    return transformed[:, :keep]


def candidate_k_values(n_rows: int, k_min: int, k_max: int) -> list[int]:
    k_upper_by_data = max(2, int(math.sqrt(n_rows)))
    upper = min(k_max, k_upper_by_data, n_rows - 1)
    lower = min(k_min, upper)
    values = sorted(set([lower, upper] + list(range(lower, upper + 1, 4))))
    return [k for k in values if 2 <= k < n_rows]


def choose_k(
    x: np.ndarray, k_min: int, k_max: int, max_silhouette_samples: int, seed: int
) -> int:
    ks = candidate_k_values(len(x), k_min, k_max)
    if not ks:
        return 2

    if len(x) > max_silhouette_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_silhouette_samples, replace=False)
        x_eval = x[idx]
    else:
        x_eval = x

    best_k = ks[0]
    best_score = -1.0
    for k in ks:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=4096,
            n_init="auto",
        )
        labels = model.fit_predict(x_eval)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(x_eval, labels, metric="euclidean")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_segments(x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
    )
    return model.fit_predict(x)


def allocate_sample_sizes(cluster_sizes: pd.Series, target: int) -> pd.Series:
    if target >= int(cluster_sizes.sum()):
        return cluster_sizes.copy()

    proportional = cluster_sizes / cluster_sizes.sum() * target
    base = np.floor(proportional).astype(int)
    base = np.minimum(base, cluster_sizes.astype(int))

    remainder = target - int(base.sum())
    fractions = proportional - base
    available = (cluster_sizes - base).astype(int)
    order = fractions.sort_values(ascending=False).index.tolist()

    while remainder > 0:
        allocated = False
        for cluster_id in order:
            if available.loc[cluster_id] <= 0:
                continue
            base.loc[cluster_id] += 1
            available.loc[cluster_id] -= 1
            remainder -= 1
            allocated = True
            if remainder == 0:
                break
        if not allocated:
            break

    return base


def sample_by_cluster(
    clustered: gpd.GeoDataFrame, sample_sizes: pd.Series, seed: int
) -> gpd.GeoDataFrame:
    sampled_parts: list[gpd.GeoDataFrame] = []
    for cluster_id, n in sample_sizes.items():
        if n <= 0:
            continue
        group = clustered.loc[clustered["cluster_id"] == cluster_id]
        sampled = group.sample(n=int(n), random_state=seed)
        sampled_parts.append(sampled)
    sampled_all = pd.concat(sampled_parts, ignore_index=True)
    return gpd.GeoDataFrame(sampled_all, geometry="geometry", crs=clustered.crs)


def write_outputs(
    clustered: gpd.GeoDataFrame,
    sampled: gpd.GeoDataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_clusters_path = output_dir / "segments_with_clusters.gpkg"
    sample_path = output_dir / "sampled_segments_500.gpkg"
    sampled_csv_path = output_dir / "sampled_segments_500.csv"
    summary_path = output_dir / "cluster_summary.csv"
    features_path = output_dir / "used_features.txt"

    clustered.to_file(all_clusters_path, driver="GPKG")
    sampled.to_file(sample_path, driver="GPKG")
    sampled.drop(columns="geometry").to_csv(sampled_csv_path, index=False)

    summary = (
        clustered.groupby("cluster_id")
        .agg(
            n_segments=("segment_uid", "count"),
            n_chips=("chip_name", "nunique"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    summary.to_csv(summary_path, index=False)
    features_path.write_text("\n".join(feature_cols), encoding="utf-8")


def main() -> None:
    args = parse_args()
    shp_files = list_shapefiles(args.polygons_dir)
    polygons = load_polygons(shp_files)

    feature_cols = select_numeric_features(polygons, args.feature_prefixes)
    x_raw = build_feature_matrix(polygons, feature_cols)
    x = preprocess_features(x_raw, args.seed)

    chosen_k = choose_k(
        x,
        k_min=args.k_min,
        k_max=args.k_max,
        max_silhouette_samples=args.max_silhouette_samples,
        seed=args.seed,
    )
    labels = cluster_segments(x, n_clusters=chosen_k, seed=args.seed)

    clustered = polygons.copy()
    clustered["cluster_id"] = labels

    cluster_sizes = clustered["cluster_id"].value_counts().sort_index()
    sample_sizes = allocate_sample_sizes(cluster_sizes, args.target_sample_size)
    sampled = sample_by_cluster(clustered, sample_sizes, args.seed)

    write_outputs(clustered, sampled, feature_cols, args.output_dir)

    print(f"Loaded shapefiles: {len(shp_files)}")
    print(f"Segments available: {len(clustered)}")
    print(f"Feature columns used: {len(feature_cols)}")
    print(f"Chosen clusters (k): {chosen_k}")
    print(f"Requested sample size: {args.target_sample_size}")
    print(f"Actual sampled segments: {len(sampled)}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
