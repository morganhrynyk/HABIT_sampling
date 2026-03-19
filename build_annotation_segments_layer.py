from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the full segmentation shapefiles into one annotation layer and "
            "flag segments selected by select_diverse_segments.py."
        )
    )
    parser.add_argument(
        "--full-segments-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\full\felzenszwalb_with_masks_polygons_250m_class_cols"
        ),
        help="Directory containing the full segment shapefiles to merge.",
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\diverse_method\diverse_segments_for_annotation.csv"
        ),
        help="CSV output from select_diverse_segments.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\annotations"
        ),
        help="Directory for the merged annotation layer.",
    )
    parser.add_argument(
        "--output-name",
        default="full_segments_annotation_status.shp",
        help="Output shapefile name.",
    )
    parser.add_argument(
        "--annotation-field",
        default="ann_sel",
        help="Binary field name for annotation status. Keep it <=10 chars for shapefile.",
    )
    parser.add_argument(
        "--all-segments-name",
        default="all_segments.shp",
        help="Output shapefile name for the plain merged segments layer.",
    )
    return parser.parse_args()


def infer_source_row_index(row: pd.Series) -> int | None:
    if "source_row_index" in row.index and pd.notna(row["source_row_index"]):
        return int(row["source_row_index"])

    segment_id = str(row.get("segment_id", ""))
    tail = segment_id.rsplit("_", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


def list_shapefiles(full_segments_dir: Path) -> list[Path]:
    shp_files = sorted(full_segments_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No shapefiles found in {full_segments_dir}")
    return shp_files


def build_selected_lookup(selected_df: pd.DataFrame) -> set[tuple[str, int]]:
    if "source_shp" not in selected_df.columns:
        raise ValueError("Selected CSV is missing required column: source_shp")

    selected = selected_df.copy()
    selected["source_row_index_resolved"] = selected.apply(infer_source_row_index, axis=1)
    if selected["source_row_index_resolved"].isna().any():
        missing_count = int(selected["source_row_index_resolved"].isna().sum())
        raise ValueError(
            "Selected CSV cannot be mapped back to source polygons because "
            f"{missing_count} row(s) are missing source_row_index information."
        )

    return {
        (str(source_shp), int(source_row_index))
        for source_shp, source_row_index in zip(
            selected["source_shp"], selected["source_row_index_resolved"]
        )
    }


def merge_segments(
    shp_files: list[Path],
    selected_lookup: set[tuple[str, int]],
    annotation_field: str,
):
    import geopandas as gpd

    frames: list[gpd.GeoDataFrame] = []
    for shp_path in shp_files:
        gdf = gpd.read_file(shp_path).reset_index(drop=True)
        gdf["chip_name"] = shp_path.stem
        gdf["source_shp"] = str(shp_path)
        gdf["src_row"] = gdf.index.astype(int)
        gdf["segment_id"] = gdf["chip_name"] + "_" + gdf["src_row"].astype(str)
        keys = list(zip(gdf["source_shp"], gdf["src_row"]))
        gdf[annotation_field] = [1 if key in selected_lookup else 0 for key in keys]
        frames.append(gdf)

    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)


def main() -> None:
    args = parse_args()
    if len(args.annotation_field) > 10:
        raise ValueError(
            f"annotation-field '{args.annotation_field}' is too long for shapefile output."
        )

    shp_files = list_shapefiles(args.full_segments_dir)
    selected_df = pd.read_csv(args.selected_csv)
    selected_lookup = build_selected_lookup(selected_df)

    merged_gdf = merge_segments(shp_files, selected_lookup, args.annotation_field)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    all_segments_path = args.output_dir / args.all_segments_name
    merged_gdf.to_file(out_path)
    merged_gdf.drop(columns=[args.annotation_field]).to_file(all_segments_path)

    selected_count = int(merged_gdf[args.annotation_field].sum())
    print(f"Shapefiles merged: {len(shp_files)}")
    print(f"Segments in merged layer: {len(merged_gdf)}")
    print(f"Segments flagged for annotation: {selected_count}")
    print(f"Wrote: {out_path.resolve()}")
    print(f"Wrote: {all_segments_path.resolve()}")


if __name__ == "__main__":
    main()
