#!/usr/bin/env python
"""
Minimal unified analysis script.

Replaces:
  - analyze_dimensionality.py
  - analyze_results.py
  - generate_detailed_analysis.py
  - summarize_results.py

Usage:
  python analysis.py

Assumptions:
  Result files are CSVs at: reports/results_*.csv
  Each CSV has columns:
    embedding, dimensionality, classifier,
    cv_accuracy_mean, cv_accuracy_std,
    cv_precision_mean, cv_precision_std,
    cv_recall_mean, cv_recall_std,
    cv_f1_mean, cv_f1_std,
    cv_fit_time_mean, cv_fit_time_std
"""

import glob
import math
import pandas as pd
import sys

PATTERN = "reports/results_*.csv"


def load_results():
    files = glob.glob(PATTERN)
    if not files:
        print(f"[ERROR] No result files found matching {PATTERN}")
        sys.exit(1)
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    if not frames:
        print("[ERROR] No readable CSV result files.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    needed = {
        "embedding",
        "dimensionality",
        "classifier",
        "cv_f1_mean",
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_fit_time_mean",
    }
    missing = needed - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)
    return df


def efficiency_score(row):
    # Simple latency-aware score: higher F1 and lower time produce higher score.
    return row["cv_f1_mean"] / math.log1p(row["cv_fit_time_mean"])


def main():
    df = load_results()
    total = len(df)
    df_sorted = df.sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)
    df["efficiency_score"] = df.apply(efficiency_score, axis=1)
    eff_best = df.sort_values("efficiency_score", ascending=False).iloc[0]

    print("=" * 70)
    print(f"TOTAL EXPERIMENTS: {total}")
    print("=" * 70)

    best_overall = df_sorted.iloc[0]
    print("\nBEST OVERALL MODEL:")
    print(
        f"  {best_overall['embedding']}-{int(best_overall['dimensionality'])}-{best_overall['classifier'].upper()}"
    )
    print(
        f"  F1={best_overall['cv_f1_mean']:.4f} Acc={best_overall['cv_accuracy_mean']:.4f} "
        f"Precision={best_overall['cv_precision_mean']:.4f} Recall={best_overall['cv_recall_mean']:.4f} "
        f"Time={best_overall['cv_fit_time_mean']:.2f}s"
    )

    print("\nTOP 10 BY F1:")
    for i, row in enumerate(df_sorted.head(10).itertuples(), 1):
        print(
            f" #{i}: {row.embedding}-{int(row.dimensionality)}-{row.classifier.upper()} "
            f"F1={row.cv_f1_mean:.4f} Time={row.cv_fit_time_mean:.2f}s"
        )

    print("\nBEST BY EMBEDDING:")
    for emb in sorted(df.embedding.unique()):
        sub = df[df.embedding == emb].sort_values("cv_f1_mean", ascending=False).iloc[0]
        print(
            f"  {emb.upper()}: {sub['embedding']}-{int(sub['dimensionality'])}-{sub['classifier'].upper()} "
            f"F1={sub['cv_f1_mean']:.4f}"
        )

    print("\nBEST BY CLASSIFIER:")
    for clf in sorted(df.classifier.unique()):
        sub = (
            df[df.classifier == clf].sort_values("cv_f1_mean", ascending=False).iloc[0]
        )
        print(
            f"  {clf.upper()}: {sub['embedding']}-{int(sub['dimensionality'])}-{sub['classifier'].upper()} "
            f"F1={sub['cv_f1_mean']:.4f} Time={sub['cv_fit_time_mean']:.2f}s"
        )

    print("\nBEST BY DIMENSIONALITY:")
    for dim in sorted(df.dimensionality.unique()):
        sub = (
            df[df.dimensionality == dim]
            .sort_values("cv_f1_mean", ascending=False)
            .iloc[0]
        )
        print(
            f"  {int(dim)}D: {sub['embedding']}-{int(sub['dimensionality'])}-{sub['classifier'].upper()} "
            f"F1={sub['cv_f1_mean']:.4f}"
        )

    print("\nDIMENSIONALITY SUMMARY (mean over all models with that dimension):")
    for dim in sorted(df.dimensionality.unique()):
        subset = df[df.dimensionality == dim]
        print(
            f"  {int(dim)}D: count={len(subset)} "
            f"F1_mean={subset.cv_f1_mean.mean():.4f} "
            f"Time_mean={subset.cv_fit_time_mean.mean():.2f}s"
        )

    print("\nDIMENSIONALITY TRADE-OFFS (consecutive dims per embedding+classifier):")
    grouped = df.groupby(["embedding", "classifier"])
    for (emb, clf), sub in grouped:
        sub = sub.sort_values("dimensionality")
        rows = sub.to_dict(orient="records")
        for prev, curr in zip(rows, rows[1:]):
            f1_gain_pct = (
                (curr["cv_f1_mean"] - prev["cv_f1_mean"]) / prev["cv_f1_mean"] * 100
                if prev["cv_f1_mean"]
                else float("nan")
            )
            time_increase_pct = (
                (curr["cv_fit_time_mean"] - prev["cv_fit_time_mean"])
                / prev["cv_fit_time_mean"]
                * 100
                if prev["cv_fit_time_mean"]
                else float("nan")
            )
            print(
                f"  {emb}-{clf.upper()} {int(prev['dimensionality'])}D→{int(curr['dimensionality'])}D "
                f"F1_gain={f1_gain_pct:.2f}% Time_increase={time_increase_pct:.2f}%"
            )

    print("\nEFFICIENCY WINNER (F1/log1p(time)):")
    print(
        f"  {eff_best['embedding']}-{int(eff_best['dimensionality'])}-{eff_best['classifier'].upper()} "
        f"Score={eff_best['efficiency_score']:.4f} F1={eff_best['cv_f1_mean']:.4f} "
        f"Time={eff_best['cv_fit_time_mean']:.2f}s"
    )

    # Missing experiment detection
    expected_grid = []
    embeddings_dims = {
        "word2vec": [100, 200, 300],
        "fasttext": [100, 200, 300],
        "bert": [100, 200, 300, 768],
    }
    classifiers = ["lr", "svm", "rf"]
    for emb, dims in embeddings_dims.items():
        for d in dims:
            for c in classifiers:
                expected_grid.append((emb, d, c))

    have = set(
        (r.embedding, int(r.dimensionality), r.classifier) for r in df.itertuples()
    )
    missing = [trip for trip in expected_grid if trip not in have]

    print("\nMISSING EXPERIMENTS:")
    if missing:
        for emb, d, c in missing:
            print(f"  {emb}-{d}-{c}")
    else:
        print("  None")

    print("\nDONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
