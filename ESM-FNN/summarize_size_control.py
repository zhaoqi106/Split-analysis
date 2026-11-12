"""Summarize training-size control results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

_DEFAULT_PROJECT_ROOT = Path("/data")


def _ensure_protocol(project_root: Path):
    adapt_dir = project_root / "adapt"
    if str(adapt_dir) not in sys.path:
        sys.path.insert(0, str(adapt_dir))
    import protocol

    protocol.set_roots(project_root)
    return protocol


_ensure_protocol(_DEFAULT_PROJECT_ROOT)
from protocol import regression_metrics


def collect(output_root: Path) -> pd.DataFrame:
    rows = []
    runs_dir = output_root / "runs"
    if not runs_dir.is_dir():
        return pd.DataFrame()
    for task_dir in sorted(runs_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        manifest_path = task_dir / "run_manifest.json"
        oof_path = task_dir / "oof_predictions.csv"
        fold_metrics_path = task_dir / "fold_metrics.csv"
        if not (manifest_path.is_file() and oof_path.is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            oof = pd.read_csv(oof_path)
        except Exception:
            continue
        fold_row = {}
        if fold_metrics_path.is_file():
            try:
                fm = pd.read_csv(fold_metrics_path)
                if len(fm):
                    fold_row = fm.iloc[0].to_dict()
            except Exception:
                pass
        # per-fold metrics from oof
        y_true = pd.to_numeric(oof["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(oof["y_pred"], errors="coerce").to_numpy(dtype=float)
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "task_key": task_dir.name,
                "dataset": manifest.get("dataset"),
                "split_seed": manifest.get("split_seed"),
                "outer_fold": manifest.get("outer_fold"),
                "n_fit": manifest.get("n_fit", manifest.get("n_inner_train")),
                "n_inner_train_available": manifest.get("n_inner_train_available"),
                "n_outer_train": manifest.get("n_outer_train"),
                "n_inner_val": manifest.get("n_inner_val"),
                "n_outer_val": manifest.get("n_outer_val"),
                "selected_epoch": manifest.get("selected_epoch"),
                "best_inner_val_metric": manifest.get("best_inner_val_metric"),
                "sampling_seed": manifest.get("sampling_seed"),
                "model_seed": manifest.get("model_seed"),
                "pcc": m["pcc"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "spearman": m["spearman"],
                "n_predicted": m["n"],
                "target_type": "full" if manifest.get("n_fit", 0) == 0 or manifest.get("n_fit") == manifest.get("n_inner_train_available") else "reduced",
                "scale_label": "full" if (manifest.get("n_fit", 0) == 0 or manifest.get("n_fit") == manifest.get("n_inner_train_available")) else f"nfit{manifest.get('n_fit')}",
                "result_source": "size_control_run",
                "oof_path": str(oof_path),
            }
        )
    return pd.DataFrame(rows)


def pooled_oof(output_root: Path, per_fold: pd.DataFrame) -> pd.DataFrame:
    runs_dir = output_root / "runs"
    records = []
    # group by dataset, seed, n_fit
    groups = per_fold.groupby(["dataset", "split_seed", "scale_label"], dropna=False)
    for (dataset, seed, scale_label), g in groups:
        ys_true = []
        ys_pred = []
        for _, r in g.iterrows():
            oof = pd.read_csv(r["oof_path"])
            ys_true.append(pd.to_numeric(oof["y_true"], errors="coerce").to_numpy(dtype=float))
            ys_pred.append(pd.to_numeric(oof["y_pred"], errors="coerce").to_numpy(dtype=float))
        if not ys_true:
            continue
        yt = np.concatenate(ys_true)
        yp = np.concatenate(ys_pred)
        m = regression_metrics(yt, yp)
        records.append(
            {
                "dataset": dataset,
                "split_seed": seed,
                "scale_label": scale_label,
                "n_fit": g["n_fit"].mean(),
                "n_folds": len(g),
                "n_records": len(yt),
                "pcc": m["pcc"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "spearman": m["spearman"],
            }
        )
    return pd.DataFrame(records)


def collect_formal_full(output_root: Path, project_root: Path) -> pd.DataFrame:
    """Full-scale reference points reuse the existing full runs."""
    import csv

    plan_path = output_root / "size_control_plan.csv"
    if not plan_path.is_file():
        return pd.DataFrame()
    plan = list(csv.DictReader(plan_path.open(encoding="utf-8")))
    formal_root = project_root / "adapt" / "runs" / "adapt_code_v1"
    rows = []
    for t in plan:
        if t.get("target_type") != "full":
            continue
        dataset = t["dataset"]
        seed = int(t["split_seed"])
        fold = int(t["outer_fold"])
        task_key = f"ESM-FNN__{dataset}__RCSV__seed{seed}__fold{fold}"
        d = formal_root / task_key
        mpath = d / "run_manifest.json"
        opath = d / "oof_predictions.csv"
        if not (mpath.is_file() and opath.is_file()):
            continue
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        oof = pd.read_csv(opath)
        y_true = pd.to_numeric(oof["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(oof["y_pred"], errors="coerce").to_numpy(dtype=float)
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "task_key": t["task_key"],
                "dataset": dataset,
                "split_seed": seed,
                "outer_fold": fold,
                "n_fit": manifest.get("n_inner_train"),
                "n_inner_train_available": manifest.get("n_inner_train"),
                "n_outer_train": manifest.get("n_outer_train"),
                "n_inner_val": manifest.get("n_inner_val"),
                "n_outer_val": manifest.get("n_outer_val"),
                "selected_epoch": manifest.get("selected_epoch"),
                "best_inner_val_metric": manifest.get("best_inner_val_metric"),
                "sampling_seed": "",
                "model_seed": manifest.get("model_seed"),
                "pcc": m["pcc"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "spearman": m["spearman"],
                "n_predicted": m["n"],
                "target_type": "full",
                "scale_label": "full",
                "result_source": "original_formal",
                "oof_path": str(opath),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="汇总 size control 结果")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default=str(_DEFAULT_PROJECT_ROOT))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    project_root = Path(args.project_root)
    per_fold = collect(output_root)
    formal_full = collect_formal_full(output_root, project_root)
    if len(formal_full):
        per_fold = pd.concat([per_fold, formal_full], ignore_index=True)
    out_dir = output_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)

    pooled = pooled_oof(output_root, per_fold)
    pooled.to_csv(out_dir / "pooled_oof_by_seed.csv", index=False)

    # scale summary: aggregate per (dataset, n_fit) across seeds/folds
    if len(per_fold):
        scale = (
            per_fold.groupby(["dataset", "scale_label"], dropna=False)
            .agg(
                n_tasks=("task_key", "count"),
                pcc_mean=("pcc", "mean"),
                pcc_std=("pcc", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                n_fit_mean=("n_fit", "mean"),
                n_fit_min=("n_fit", "min"),
                n_fit_max=("n_fit", "max"),
            )
            .reset_index()
        )
        scale.to_csv(out_dir / "scale_summary_per_fold.csv", index=False)

    if len(pooled):
        scale_pooled = (
            pooled.groupby(["dataset", "scale_label"], dropna=False)
            .agg(
                n_seeds=("split_seed", "nunique"),
                pcc_mean=("pcc", "mean"),
                pcc_std=("pcc", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
            )
            .reset_index()
        )
        scale_pooled.to_csv(out_dir / "scale_summary_pooled.csv", index=False)

    # Per-fold difference between full and reduced scale on the same RCSV records.
    if len(per_fold):
        delta_rows = []
        full = per_fold[per_fold["target_type"] == "full"]
        reduced = per_fold[per_fold["target_type"] == "reduced"]
        full_idx = full.set_index(["dataset", "split_seed", "outer_fold"])
        for _, r in reduced.iterrows():
            key = (r["dataset"], r["split_seed"], r["outer_fold"])
            if key not in full_idx.index:
                continue
            f = full_idx.loc[key]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            delta_rows.append(
                {
                    "dataset": r["dataset"],
                    "split_seed": r["split_seed"],
                    "outer_fold": r["outer_fold"],
                    "n_fit_reduced": r["n_fit"],
                    "n_fit_full": f["n_fit"],
                    "pcc_reduced": r["pcc"],
                    "pcc_full": f["pcc"],
                    "delta_pcc": r["pcc"] - f["pcc"],
                    "rmse_reduced": r["rmse"],
                    "rmse_full": f["rmse"],
                    "delta_rmse": r["rmse"] - f["rmse"],
                    "mae_reduced": r["mae"],
                    "mae_full": f["mae"],
                    "delta_mae": r["mae"] - f["mae"],
                }
            )
        if delta_rows:
            pd.DataFrame(delta_rows).to_csv(out_dir / "full_vs_reduced_delta.csv", index=False)

    # copy plan
    plan = output_root / "size_control_plan.csv"
    if plan.is_file():
        pd.read_csv(plan).to_csv(out_dir / "size_control_plan.csv", index=False)
    plan_summary = output_root / "size_control_plan_summary.csv"
    if plan_summary.is_file():
        pd.read_csv(plan_summary).to_csv(out_dir / "size_control_plan_summary.csv", index=False)

    print(f"per_fold rows: {len(per_fold)}")
    print(f"pooled rows: {len(pooled)}")
    print(f"summary dir: {out_dir}")


if __name__ == "__main__":
    main()
