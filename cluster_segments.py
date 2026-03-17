from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster segment feature table and report cluster count."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method\segment_features_for_clustering.csv"
        ),
        help="CSV created by make_clusters.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"\\Lab\Groups\Wetlands\Working\ProjectWork\HABIT\TestCases\HABIT_segmentation\03_outputs\sampling\cluster_method"
        ),
        help="Directory for clustered outputs.",
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
        help="Columns used for clustering.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of clusters (default: 30).",
    )
    parser.add_argument("--k-min", type=int, default=4, help="Min k for auto-select.")
    parser.add_argument("--k-max", type=int, default=20, help="Max k for auto-select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-silhouette-samples",
        type=int,
        default=20000,
        help="Max rows used for silhouette scoring.",
    )
    return parser.parse_args()


def validate_feature_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. "
            "Run make_clusters.py first or pass valid --feature-cols."
        )
    return cols


def get_eval_matrix(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def select_k(x: np.ndarray, k_min: int, k_max: int, max_rows: int, seed: int) -> tuple[int, dict[int, float]]:
    if len(x) < 3:
        raise ValueError("Need at least 3 rows for clustering.")

    k_min = max(2, k_min)
    k_max = min(k_max, len(x) - 1)
    if k_min > k_max:
        k_min = k_max

    x_eval = get_eval_matrix(x, max_rows, seed)
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(x_eval)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(x_eval, labels)
        scores[k] = float(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    feature_cols = validate_feature_columns(df, args.feature_cols)
    x = df[feature_cols].to_numpy(dtype=float)

    if args.k is not None:
        chosen_k = int(args.k)
        score_map: dict[int, float] = {}
    else:
        chosen_k, score_map = select_k(
            x,
            k_min=args.k_min,
            k_max=args.k_max,
            max_rows=args.max_silhouette_samples,
            seed=args.seed,
        )

    model = KMeans(n_clusters=chosen_k, random_state=args.seed, n_init=20)
    labels = model.fit_predict(x)
    df["cluster_id"] = labels

    out_csv = args.output_dir / "segment_features_with_clusters.csv"
    df.to_csv(out_csv, index=False)

    cluster_counts = (
        df["cluster_id"].value_counts().sort_index().rename_axis("cluster_id").reset_index(name="n_segments")
    )
    counts_csv = args.output_dir / "cluster_counts.csv"
    cluster_counts.to_csv(counts_csv, index=False)

    summary_lines = [
        f"input_csv: {args.input_csv}",
        f"rows_clustered: {len(df)}",
        f"feature_columns: {', '.join(feature_cols)}",
        f"chosen_k: {chosen_k}",
        f"clusters_produced: {df['cluster_id'].nunique()}",
    ]
    if score_map:
        summary_lines.append("silhouette_scores_by_k:")
        for k in sorted(score_map):
            summary_lines.append(f"  k={k}: {score_map[k]:.4f}")

    summary_txt = args.output_dir / "clustering_summary.txt"
    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Rows clustered: {len(df)}")
    print(f"Chosen k: {chosen_k}")
    print(f"Clusters produced: {df['cluster_id'].nunique()}")
    print(f"Wrote: {out_csv.resolve()}")
    print(f"Wrote: {counts_csv.resolve()}")
    print(f"Wrote: {summary_txt.resolve()}")


if __name__ == "__main__":
    main()
