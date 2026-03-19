from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


CHIP_ROOT_PATTERN = re.compile(r"(chip_r\d+_c\d+_\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-segment brightness/texture/NDVI/NDWI/rg_contrast means "
            "from nested rasters and write a table ready for clustering."
        )
    )
    parser.add_argument(
        "--polygons-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\full\felzenszwalb_with_masks_polygons_250m_class_cols"
        ),
        help="Directory containing segment shapefiles.",
    )
    parser.add_argument(
        "--feature-stacks-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\02_intermediate_data"
        ),
        help="Root directory containing nested feature rasters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method"
        ),
        help="Output directory for extracted segment features.",
    )
    parser.add_argument(
        "--segment-id-col",
        default=None,
        help="Optional existing segment ID column in shapefiles.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=8,
        help="Minimum in-segment pixels required to compute features.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=12000,
        help="Maximum rows to use in the feature comparison figure.",
    )
    return parser.parse_args()


def discover_shapefiles(polygons_dir: Path) -> list[Path]:
    files = sorted(polygons_dir.glob("*.shp"))
    if not files:
        raise FileNotFoundError(f"No shapefiles found in: {polygons_dir}")
    return files


def discover_rasters(feature_stacks_dir: Path) -> list[Path]:
    rasters = sorted(feature_stacks_dir.rglob("*.tif")) + sorted(
        feature_stacks_dir.rglob("*.tiff")
    )
    if not rasters:
        raise FileNotFoundError(f"No rasters found under: {feature_stacks_dir}")
    return rasters


def extract_chip_root(name: str) -> str | None:
    match = CHIP_ROOT_PATTERN.search(name)
    if not match:
        return None
    return match.group(1).lower()


def identify_feature_type(path: Path) -> str | None:
    text = f"{path.parent.name} {path.stem}".lower()
    if "brightness" in text:
        return "brightness"
    if "rg_contrast" in text:
        return "rg_contrast"
    if "texture" in text:
        return "texture"
    if "ndwi" in text:
        return "ndwi"
    if "ndvi" in text:
        return "ndvi"
    return None


def build_feature_index(rasters: list[Path]) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for raster in rasters:
        chip_root = extract_chip_root(raster.stem)
        feature = identify_feature_type(raster)
        if chip_root is None or feature is None:
            continue

        key = (chip_root, feature)
        if key not in index:
            index[key] = raster
            continue

        # Prefer shorter relative paths if duplicates appear.
        existing = index[key]
        if len(str(raster)) < len(str(existing)):
            index[key] = raster
    return index


def read_first_band(path: Path) -> tuple[np.ndarray, float, tuple[int, int], rasterio.Affine]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        pixel_area_m2 = float(abs(src.transform.a * src.transform.e))
        shape = (src.height, src.width)
        transform = src.transform
    return arr, pixel_area_m2, shape, transform


def process_chip(
    shp_path: Path,
    brightness_path: Path,
    rg_contrast_path: Path,
    texture_path: Path,
    ndvi_path: Path,
    ndwi_path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return pd.DataFrame()

    brightness, pixel_area_m2, shape, transform = read_first_band(brightness_path)
    rg_contrast, _, rg_contrast_shape, rg_contrast_transform = read_first_band(
        rg_contrast_path
    )
    texture, _, texture_shape, texture_transform = read_first_band(texture_path)
    ndvi, _, ndvi_shape, ndvi_transform = read_first_band(ndvi_path)
    ndwi, _, ndwi_shape, ndwi_transform = read_first_band(ndwi_path)

    if (
        rg_contrast_shape != shape
        or texture_shape != shape
        or ndvi_shape != shape
        or ndwi_shape != shape
    ):
        raise ValueError(
            f"Raster size mismatch for {shp_path.stem}: "
            f"brightness={shape}, rg_contrast={rg_contrast_shape}, "
            f"texture={texture_shape}, ndvi={ndvi_shape}, ndwi={ndwi_shape}"
        )
    if (
        rg_contrast_transform != transform
        or texture_transform != transform
        or ndvi_transform != transform
        or ndwi_transform != transform
    ):
        raise ValueError(
            f"Raster transform mismatch for {shp_path.stem}. "
            "Rasters must be aligned."
        )

    records: list[dict] = []
    chip_root = extract_chip_root(shp_path.stem) or shp_path.stem.lower()
    for row_idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        seg_mask = geometry_mask(
            [geom],
            out_shape=shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )
        pixels = int(seg_mask.sum())
        if pixels < args.min_pixels:
            continue

        b_vals = brightness[seg_mask]
        r_vals = rg_contrast[seg_mask]
        t_vals = texture[seg_mask]
        n_vals = ndvi[seg_mask]
        w_vals = ndwi[seg_mask]
        if (
            np.all(np.isnan(b_vals))
            or np.all(np.isnan(r_vals))
            or np.all(np.isnan(t_vals))
            or np.all(np.isnan(n_vals))
            or np.all(np.isnan(w_vals))
        ):
            continue

        if args.segment_id_col and args.segment_id_col in gdf.columns:
            segment_id = str(row[args.segment_id_col])
        else:
            segment_id = f"{chip_root}_{row_idx}"

        records.append(
            {
                "chip_name": chip_root,
                "segment_id": segment_id,
                "source_row_index": int(row_idx),
                "source_shp": str(shp_path),
                "brightness_raster": str(brightness_path),
                "rg_contrast_raster": str(rg_contrast_path),
                "texture_raster": str(texture_path),
                "ndvi_raster": str(ndvi_path),
                "ndwi_raster": str(ndwi_path),
                "segment_area_m2": float(pixels * pixel_area_m2),
                "brightness_mean": float(np.nanmean(b_vals)),
                "rg_contrast_mean": float(np.nanmean(r_vals)),
                "texture_mean": float(np.nanmean(t_vals)),
                "ndvi_mean": float(np.nanmean(n_vals)),
                "ndwi_mean": float(np.nanmean(w_vals)),
            }
        )
    return pd.DataFrame.from_records(records)


def add_normalized_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        std = float(out[col].std(ddof=0))
        if std <= 0 or np.isnan(std):
            out[f"{col}_z"] = 0.0
        else:
            out[f"{col}_z"] = (out[col] - float(out[col].mean())) / std
    return out


def sample_for_plot(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42).reset_index(drop=True)


def plot_feature_comparison(
    df: pd.DataFrame, feature_cols: list[str], out_path: Path, max_points: int
) -> None:
    import matplotlib.pyplot as plt

    plot_df = sample_for_plot(df, max_points=max_points)
    n = len(feature_cols)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(3.2 * n, 3.2 * n))

    for row_idx, ycol in enumerate(feature_cols):
        for col_idx, xcol in enumerate(feature_cols):
            ax = axes[row_idx, col_idx]
            if row_idx == col_idx:
                ax.hist(plot_df[xcol].to_numpy(dtype=float), bins=40, color="#4C78A8", alpha=0.85)
            elif row_idx > col_idx:
                ax.scatter(
                    plot_df[xcol].to_numpy(dtype=float),
                    plot_df[ycol].to_numpy(dtype=float),
                    s=5,
                    alpha=0.2,
                    color="#1B9E77",
                    linewidths=0,
                )
            else:
                corr = plot_df[[xcol, ycol]].corr().iloc[0, 1]
                ax.text(
                    0.5,
                    0.5,
                    f"r = {corr:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#333333",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if row_idx == n - 1:
                ax.set_xlabel(xcol)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(ycol)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Feature Comparison Matrix", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shp_files = discover_shapefiles(args.polygons_dir)
    raster_files = discover_rasters(args.feature_stacks_dir)
    feature_index = build_feature_index(raster_files)

    all_frames: list[pd.DataFrame] = []
    missing_features: list[str] = []

    for shp in shp_files:
        chip_root = extract_chip_root(shp.stem)
        if chip_root is None:
            missing_features.append(f"{shp.stem}: could not parse chip root")
            continue

        brightness_path = feature_index.get((chip_root, "brightness"))
        rg_contrast_path = feature_index.get((chip_root, "rg_contrast"))
        texture_path = feature_index.get((chip_root, "texture"))
        ndvi_path = feature_index.get((chip_root, "ndvi"))
        ndwi_path = feature_index.get((chip_root, "ndwi"))

        missing = []
        if brightness_path is None:
            missing.append("brightness")
        if rg_contrast_path is None:
            missing.append("rg_contrast")
        if texture_path is None:
            missing.append("texture")
        if ndvi_path is None:
            missing.append("ndvi")
        if ndwi_path is None:
            missing.append("ndwi")
        if missing:
            missing_features.append(f"{chip_root}: missing {', '.join(missing)}")
            continue

        frame = process_chip(
            shp,
            brightness_path,
            rg_contrast_path,
            texture_path,
            ndvi_path,
            ndwi_path,
            args,
        )
        if not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        raise RuntimeError(
            "No features were extracted. Check chip root parsing and raster naming "
            "(expects chip_r###_c###_###### + brightness/rg_contrast/texture/ndvi/ndwi "
            "in file or folder names)."
        )

    features_df = pd.concat(all_frames, ignore_index=True)
    feature_value_cols = [
        "brightness_mean",
        "rg_contrast_mean",
        "texture_mean",
        "ndvi_mean",
        "ndwi_mean",
    ]
    features_df = add_normalized_columns(
        features_df,
        feature_value_cols + ["segment_area_m2"],
    )

    out_csv = args.output_dir / "segment_features_for_clustering.csv"
    features_df.to_csv(out_csv, index=False)
    comparison_path = args.output_dir / "feature_comparison_matrix.png"
    plot_feature_comparison(
        features_df,
        feature_value_cols,
        comparison_path,
        max_points=args.max_plot_points,
    )

    missing_path = args.output_dir / "missing_features_by_chip.txt"
    missing_path.write_text("\n".join(missing_features), encoding="utf-8")

    print(f"Shapefiles discovered: {len(shp_files)}")
    print(f"Rasters discovered (recursive): {len(raster_files)}")
    print(f"Segments with extracted features: {len(features_df)}")
    print(f"Wrote: {out_csv.resolve()}")
    print(f"Wrote feature comparison figure: {comparison_path.resolve()}")
    print(f"Wrote missing-feature report: {missing_path.resolve()}")


if __name__ == "__main__":
    main()
