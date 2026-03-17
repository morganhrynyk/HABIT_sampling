from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select 3 representative segments per cluster: "
            "middle (closest to centroid), edge (farthest), and median-distance."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method\segment_features_with_clusters.csv"
        ),
        help="Clustered CSV from cluster_segments.py.",
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
        help="Feature columns used to compute centroid distances.",
    )
    parser.add_argument(
        "--output-vector",
        type=Path,
        default=Path("cluster_representative_segments.shp"),
        help=(
            "Optional vector output for selected segments. Relative paths are resolved "
            "under --output-dir. Use .shp or .gpkg."
        ),
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    required = ["cluster_id", "segment_id"] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def infer_source_row_index(row: pd.Series) -> int | None:
    if "source_row_index" in row.index and pd.notna(row["source_row_index"]):
        return int(row["source_row_index"])

    segment_id = str(row.get("segment_id", ""))
    tail = segment_id.rsplit("_", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


def euclidean_distances(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    diffs = x - centroid
    return np.sqrt(np.sum(diffs * diffs, axis=1))


def pick_positions(distances: np.ndarray) -> dict[str, int]:
    order = np.argsort(distances)
    middle_idx = int(order[0])
    edge_idx = int(order[-1])

    median_distance = float(np.median(distances))
    median_order = np.argsort(np.abs(distances - median_distance))
    median_idx = int(median_order[0])

    selected = {"middle": middle_idx, "edge": edge_idx, "median": median_idx}

    used = set()
    for label in ["middle", "edge", "median"]:
        idx = selected[label]
        if idx not in used:
            used.add(idx)
            continue
        if label == "median":
            for alt in median_order:
                alt_i = int(alt)
                if alt_i not in used:
                    selected[label] = alt_i
                    used.add(alt_i)
                    break
        elif label == "edge":
            for alt in order[::-1]:
                alt_i = int(alt)
                if alt_i not in used:
                    selected[label] = alt_i
                    used.add(alt_i)
                    break
        else:
            for alt in order:
                alt_i = int(alt)
                if alt_i not in used:
                    selected[label] = alt_i
                    used.add(alt_i)
                    break
    return selected


def select_cluster_rows(group: pd.DataFrame, feature_cols: list[str]) -> list[pd.Series]:
    x = group[feature_cols].to_numpy(dtype=float)
    centroid = np.nanmean(x, axis=0)
    distances = euclidean_distances(x, centroid)
    local_positions = pick_positions(distances)

    rows: list[pd.Series] = []
    for role in ["middle", "edge", "median"]:
        pos = local_positions[role]
        selected = group.iloc[pos].copy()
        selected["selection_role"] = role
        selected["distance_to_centroid"] = float(distances[pos])
        rows.append(selected)
    return rows


def resolve_output_vector_path(output_dir: Path, output_vector: Path) -> Path:
    if output_vector.is_absolute():
        return output_vector
    return output_dir / output_vector


def export_selected_geometries(selected_df: pd.DataFrame, out_path: Path) -> Path | None:
    if selected_df.empty or "source_shp" not in selected_df.columns:
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    validate_columns(df, args.feature_cols)

    selected_rows: list[pd.Series] = []
    notes: list[str] = []

    for cluster_id, group in df.groupby("cluster_id", sort=True):
        group = group.reset_index(drop=True)
        if len(group) == 0:
            continue
        if len(group) < 3:
            notes.append(f"cluster {cluster_id}: only {len(group)} segment(s), selected all")
            x = group[args.feature_cols].to_numpy(dtype=float)
            centroid = np.nanmean(x, axis=0)
            distances = euclidean_distances(x, centroid)
            role_names = ["middle", "edge", "median"][: len(group)]
            for i, role in enumerate(role_names):
                row = group.iloc[i].copy()
                row["selection_role"] = role
                row["distance_to_centroid"] = float(distances[i])
                selected_rows.append(row)
            continue

        selected_rows.extend(select_cluster_rows(group, args.feature_cols))

    selected_df = pd.DataFrame(selected_rows)
    selected_df = selected_df.sort_values(["cluster_id", "selection_role"]).reset_index(drop=True)

    out_csv = args.output_dir / "cluster_representative_segments.csv"
    selected_df.to_csv(out_csv, index=False)
    vector_path = export_selected_geometries(
        selected_df, resolve_output_vector_path(args.output_dir, args.output_vector)
    )

    summary_lines = [
        f"input_csv: {args.input_csv}",
        f"total_segments_available: {len(df)}",
        f"clusters_found: {df['cluster_id'].nunique()}",
        f"segments_selected: {len(selected_df)}",
        "expected_if_all_clusters_have_3: clusters_found * 3",
    ]
    notes_path = args.output_dir / "cluster_representative_selection_notes.txt"
    notes_path.write_text("\n".join(notes) if notes else "No issues.", encoding="utf-8")

    summary_path = args.output_dir / "cluster_representative_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Total segments available: {len(df)}")
    print(f"Clusters found: {df['cluster_id'].nunique()}")
    print(f"Segments selected: {len(selected_df)}")
    print(f"Wrote: {out_csv.resolve()}")
    if vector_path is not None:
        print(f"Wrote: {vector_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    print(f"Wrote: {notes_path.resolve()}")


if __name__ == "__main__":
    main()
