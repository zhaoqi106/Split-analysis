"""Shared per-fold output tables for ESM-FNN."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from protocol import regression_metrics, write_json, write_table, y_true


def mutation_values(frame: pd.DataFrame) -> list[str]:
    if "Mutation" in frame.columns:
        return frame["Mutation"].astype(str).tolist()
    return [""] * len(frame)


def enrich_predictions(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    precision_mode: str,
    train_mode: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["mutation"] = mutation_values(source)
    out["precision_mode"] = precision_mode
    out["train_mode"] = train_mode
    return out


def inner_split_table(
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    outer_val: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    for role, frame in (
        ("inner_train", inner_train),
        ("inner_val", inner_val),
        ("outer_test", outer_val),
    ):
        piece = pd.DataFrame(
            {
                "record_uid": frame["record_uid"].astype(str),
                "ID": frame["ID"].astype(str),
                "PDB": frame["PDB"].astype(str),
                "mutation": mutation_values(frame),
                "role": role,
            }
        )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def write_fold_bundle(
    output_dir: Path,
    *,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    outer_val: pd.DataFrame,
    selection: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(inner_split_table(inner_train, inner_val, outer_val), output_dir / "inner_split_manifest.csv")
    write_table(history, output_dir / "training_history.csv")
    if len(history) and "phase" in history.columns:
        selected = history.loc[history["phase"].astype(str) == "select"]
        write_table(selected if len(selected) else history, output_dir / "inner_history.csv")
    else:
        write_table(history, output_dir / "inner_history.csv")
    write_json(dict(selection), output_dir / "model_selection.json")
    write_json(dict(manifest), output_dir / "run_manifest.json")
    write_json(dict(manifest), output_dir / "run_config.json")
    write_table(predictions, output_dir / "oof_predictions.csv")

    scored = predictions.loc[pd.to_numeric(predictions["y_pred"], errors="coerce").notna()].copy()
    metrics = regression_metrics(
        pd.to_numeric(scored["y_true"], errors="coerce").to_numpy(dtype=float) if len(scored) else y_true(outer_val)[:0],
        pd.to_numeric(scored["y_pred"], errors="coerce").to_numpy(dtype=float) if len(scored) else y_true(outer_val)[:0],
    )
    coverage = {
        "n_rows": int(len(predictions)),
        "n_predicted": int(len(scored)),
        "n_failed": int((predictions["prediction_status"].astype(str) == "failed").sum()) if "prediction_status" in predictions else 0,
        "coverage": (len(scored) / len(predictions)) if len(predictions) else 0.0,
        "duplicate_id_groups": _duplicate_id_groups(predictions),
        "record_uid_unique": bool(predictions["record_uid"].nunique() == len(predictions)) if len(predictions) else True,
    }
    fold_row = {
        "model": manifest.get("model", ""),
        "dataset": manifest.get("dataset", ""),
        "split_type": manifest.get("split_type", ""),
        "split_seed": manifest.get("split_seed", ""),
        "outer_fold": manifest.get("outer_fold", ""),
        **metrics,
        **coverage,
        "selected_epoch": selection.get("selected_epoch", ""),
        "precision_mode": manifest.get("precision_mode", ""),
        "train_mode": manifest.get("train_mode", ""),
        "n_outer_train": manifest.get("n_outer_train", ""),
        "n_inner_train": manifest.get("n_inner_train", ""),
        "n_inner_val": manifest.get("n_inner_val", ""),
        "n_outer_val": manifest.get("n_outer_val", ""),
    }
    write_table(pd.DataFrame([fold_row]), output_dir / "fold_metrics.csv")
    coverage_row = dict(coverage)
    coverage_row["model"] = manifest.get("model", "")
    coverage_row["outer_fold"] = manifest.get("outer_fold", "")
    write_table(pd.DataFrame([coverage_row]), output_dir / "prediction_coverage.csv")
    _write_baseline(output_dir, predictions, inner_train, inner_val, manifest)
    return {**metrics, **coverage}


def _duplicate_id_groups(frame: pd.DataFrame) -> int:
    if "ID" not in frame.columns or "record_uid" not in frame.columns or len(frame) == 0:
        return 0
    grouped = frame.groupby(frame["ID"].astype(str))["record_uid"].nunique()
    return int((grouped > 1).sum())


def _write_baseline(
    output_dir: Path,
    predictions: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    manifest: Mapping[str, object],
) -> None:
    labels = pd.concat([y_frame(inner_train), y_frame(inner_val)], ignore_index=True)
    values = pd.to_numeric(labels["ddG"], errors="coerce").dropna()
    baseline_value = float(values.mean()) if len(values) else float("nan")
    baseline = predictions.copy()
    baseline["model"] = f"{manifest.get('model', '')}__train_mean"
    baseline["y_pred"] = baseline_value
    baseline["prediction_status"] = "baseline"
    baseline["failure_reason"] = "外层训练集标签均值，只作对照，不是模型预测"
    baseline["train_mode"] = "train_mean_baseline"
    write_table(baseline, output_dir / "baseline_predictions.csv")


def y_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["ddG"]]


def history_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))
