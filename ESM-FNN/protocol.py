"""Outer-fold protocol with an inner validation split.

Does not rewrite the master CSV or create the RCSV, PCSV, or ICSV files.
record_uid is the physical row number in the master CSV, so duplicate M1101 IDs stay separate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DATA_ROOT = Path("/data")
RAW_DIR = DATA_ROOT / "data_split" / "raw"
SPLIT_ROOT = DATA_ROOT / "data_split" / "splits"


def set_roots(data_root, split_root=None):
    global DATA_ROOT, RAW_DIR, SPLIT_ROOT
    DATA_ROOT = Path(data_root)
    RAW_DIR = DATA_ROOT / "data_split" / "raw"
    SPLIT_ROOT = Path(split_root) if split_root else DATA_ROOT / "data_split" / "splits"


DATASETS = ("S1131", "M1101")
SEEDS = (0, 42, 1234, 2026, 1024)
SPLIT_TYPES = ("RCSV", "PCSV", "ICSV")
OUTER_FOLDS = (0, 1, 2, 3, 4)
INNER_FRACTION = 0.10

WT_CHAIN_COLS = ("R1", "R2", "R3", "R4", "R5", "R6")
MUT_CHAIN_COLS = ("R1m", "R2m", "R3m", "R4m", "R5m", "R6m")
RIGHT_WT_COLS = ("L1", "L2", "L3")
RIGHT_MUT_COLS = ("L1m", "L2m", "L3m")

OOF_COLUMNS = (
    "dataset",
    "model",
    "split_type",
    "split_seed",
    "outer_fold",
    "inner_split_seed",
    "model_seed",
    "record_uid",
    "ID",
    "PDB",
    "y_true",
    "y_pred",
    "n_outer_train",
    "n_inner_train",
    "n_inner_val",
    "n_outer_val",
    "selected_epoch",
    "selection_metric",
    "best_inner_val_metric",
    "prediction_status",
    "failure_reason",
)


class ProtocolError(RuntimeError):
    """The outer split or the row alignment cannot be trusted."""


class ICSVGroupMapMissing(ProtocolError):
    """No reusable ICSV connected-component map is on disk."""


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def record_uid(dataset: str, row_index: int) -> str:
    return f"{dataset}:{int(row_index):06d}"


def load_master(dataset: str) -> pd.DataFrame:
    if dataset not in DATASETS:
        raise ProtocolError(f"未知数据集: {dataset}")
    path = RAW_DIR / f"{dataset}.csv"
    frame = read_csv(path)
    if "ID" not in frame.columns or "PDB" not in frame.columns or "ddG" not in frame.columns:
        raise ProtocolError(f"{path} 缺少 ID、PDB 或 ddG")
    frame = frame.copy()
    frame.insert(0, "record_uid", [record_uid(dataset, i) for i in range(len(frame))])
    frame.insert(1, "master_row", np.arange(len(frame), dtype=int))
    return frame


def _fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    subset = frame.loc[:, list(columns)].astype(str)
    joined = subset.apply(lambda row: "\x1f".join(row.tolist()), axis=1)
    return joined.map(lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest())


def attach_record_uid(fold: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Align each outer-split row to the master CSV by full-row content.

    The master CSV has no two identical rows. Zero or multiple matches stop the run.
    """
    columns = [column for column in master.columns if column not in {"record_uid", "master_row"}]
    missing = [column for column in columns if column not in fold.columns]
    extra = [column for column in fold.columns if column not in columns]
    if missing or extra:
        raise ProtocolError(f"划分列与正式 CSV 不一致，缺少 {missing}，多出 {extra}")

    master_key = _fingerprint(master, columns)
    fold_key = _fingerprint(fold, columns)
    buckets: dict[str, list[str]] = {}
    for key, uid in zip(master_key, master["record_uid"]):
        buckets.setdefault(key, []).append(uid)

    uids: list[str] = []
    ambiguous: list[int] = []
    unknown: list[int] = []
    for position, key in enumerate(fold_key):
        matches = buckets.get(key, [])
        if len(matches) == 1:
            uids.append(matches[0])
        elif len(matches) == 0:
            unknown.append(position)
            uids.append("")
        else:
            ambiguous.append(position)
            uids.append("")

    if unknown or ambiguous:
        raise ProtocolError(
            f"划分行无法唯一对齐到正式 CSV: unknown={unknown[:10]} ambiguous={ambiguous[:10]}"
        )

    aligned = fold.copy()
    aligned.insert(0, "record_uid", uids)
    return aligned


def outer_fold_paths(dataset: str, seed: int, split_type: str, fold: int) -> tuple[Path, Path]:
    directory = SPLIT_ROOT / dataset / str(seed) / split_type
    if split_type == "AGCV":
        return directory / f"fold_{fold}_outer_train.csv", directory / f"fold_{fold}_outer_eval.csv"
    return directory / f"train_fold_{fold}.csv", directory / f"val_fold_{fold}.csv"


def _align_saved_fold(frame: pd.DataFrame, master: pd.DataFrame, path: Path) -> pd.DataFrame:
    """AGCV files already store record_uid. The aligned row number must match it."""
    if "record_uid" not in frame.columns:
        raise ProtocolError(f"{path} 缺少 record_uid")
    stored = frame["record_uid"].astype(str).tolist()
    body = frame.drop(columns=["record_uid"])
    aligned = attach_record_uid(body, master)
    if aligned["record_uid"].astype(str).tolist() != stored:
        raise ProtocolError(f"{path} 的 record_uid 与正式 CSV 行号不一致")
    return aligned


def _read_aligned_fold(path: Path, master: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise ProtocolError(f"外层划分文件不存在: {path}")
    return _align_saved_fold(read_csv(path), master, path)


def load_agcv_fold(dataset: str, seed: int, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return outer_train, outer_eval, inner_train, and inner_val from the files on disk."""
    if dataset not in DATASETS:
        raise ProtocolError(f"未知数据集: {dataset}")
    if seed not in SEEDS or fold not in OUTER_FOLDS:
        raise ProtocolError(f"种子或折号不在既定协议内: seed={seed} fold={fold}")
    directory = SPLIT_ROOT / dataset / str(seed) / "AGCV"
    master = load_master(dataset)
    inner_train = _read_aligned_fold(directory / f"fold_{fold}_inner_train.csv", master)
    inner_val = _read_aligned_fold(directory / f"fold_{fold}_inner_val.csv", master)
    outer_train = _read_aligned_fold(directory / f"fold_{fold}_outer_train.csv", master)
    outer_eval = _read_aligned_fold(directory / f"fold_{fold}_outer_eval.csv", master)
    train_ids = set(inner_train["record_uid"]) | set(inner_val["record_uid"])
    if train_ids != set(outer_train["record_uid"]):
        raise ProtocolError(f"{directory} fold {fold} 的 inner 并集不是 outer_train")
    if set(inner_train["record_uid"]) & set(inner_val["record_uid"]):
        raise ProtocolError(f"{directory} fold {fold} 的 inner_train 与 inner_val 相交")
    if set(outer_train["record_uid"]) & set(outer_eval["record_uid"]):
        raise ProtocolError(f"{directory} fold {fold} 的 outer_train 与 outer_eval 相交")
    return outer_train, outer_eval, inner_train, inner_val


def load_outer_fold(dataset: str, seed: int, split_type: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_type == "AGCV":
        outer_train, outer_eval, _inner_train, _inner_val = load_agcv_fold(dataset, seed, fold)
        return outer_train, outer_eval
    if split_type not in SPLIT_TYPES:
        raise ProtocolError(f"未知划分类型: {split_type}")
    if seed not in SEEDS or fold not in OUTER_FOLDS:
        raise ProtocolError(f"种子或折号不在既定协议内: seed={seed} fold={fold}")
    master = load_master(dataset)
    train_path, val_path = outer_fold_paths(dataset, seed, split_type, fold)
    if not train_path.is_file() or not val_path.is_file():
        raise ProtocolError(f"外层划分文件不存在: {train_path} / {val_path}")
    train = attach_record_uid(read_csv(train_path), master)
    val = attach_record_uid(read_csv(val_path), master)
    overlap = set(train["record_uid"]) & set(val["record_uid"])
    if overlap:
        raise ProtocolError(f"外层训练集与评估集相交: {sorted(overlap)[:5]}")
    return train, val


def input_signature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"record_uid", "master_row", "ddG"}
    return [column for column in frame.columns if column not in blocked]


def conflicting_duplicate_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that share an input and differ in ddG. Report them; do not merge them."""
    columns = input_signature_columns(frame)
    signature = _fingerprint(frame, columns)
    work = frame.copy()
    work["_input_signature"] = signature
    rows: list[dict[str, object]] = []
    for key, group in work.groupby("_input_signature", sort=False):
        labels = group["ddG"].astype(str).unique().tolist()
        ids = group["ID"].astype(str).unique().tolist()
        if len(group) > 1 and (len(labels) > 1 or len(ids) > 1 or len(group) != len(ids)):
            rows.append(
                {
                    "input_signature": key,
                    "n_records": int(len(group)),
                    "IDs": "|".join(ids),
                    "record_uids": "|".join(group["record_uid"].astype(str)),
                    "ddG_values": "|".join(group["ddG"].astype(str)),
                    "labels_conflict": bool(len(labels) > 1),
                }
            )
    return pd.DataFrame(rows)


def _group_holdout(
    frame: pd.DataFrame,
    group_labels: Sequence[object],
    seed: int,
    fraction: float = INNER_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if len(frame) != len(group_labels):
        raise ProtocolError("分组标签数量与外层训练集不一致")
    work = frame.copy()
    work["_group"] = [str(label) for label in group_labels]
    sizes = work.groupby("_group", sort=False).size().to_dict()
    names = list(sizes)
    if len(names) < 2:
        raise ProtocolError(
            f"外层训练集只有 {len(names)} 个不可拆组，无法做组级内部留出"
        )

    rng = np.random.default_rng(seed)
    order = names[:]
    rng.shuffle(order)
    target = max(1, int(round(len(work) * fraction)))
    ceiling = max(target, int(round(len(work) * 0.25)))
    chosen: list[str] = []
    chosen_n = 0
    skipped_large: list[str] = []

    for name in order:
        remaining_groups = len(names) - len(chosen) - 1
        if remaining_groups < 1:
            break
        size = int(sizes[name])
        if size > ceiling:
            skipped_large.append(name)
            continue
        if chosen and chosen_n >= target:
            break
        if chosen and chosen_n + size > ceiling and chosen_n >= max(1, target // 2):
            continue
        chosen.append(name)
        chosen_n += size
        if chosen_n >= target:
            break

    if not chosen:
        smallest = min(order, key=lambda name: (sizes[name], name))
        chosen = [smallest]
        chosen_n = int(sizes[smallest])

    val_mask = work["_group"].isin(chosen)
    inner_val = work.loc[val_mask].reset_index(drop=True)
    inner_train = work.loc[~val_mask].reset_index(drop=True)
    info = {
        "target_records": target,
        "n_groups_total": len(names),
        "n_groups_inner_val": len(chosen),
        "n_groups_skipped_as_large": len(skipped_large),
        "fraction_achieved": chosen_n / len(work),
    }
    return inner_train, inner_val, info


def inner_split_rcsv(train: pd.DataFrame, seed: int, fraction: float = INNER_FRACTION):
    rng = np.random.default_rng(seed)
    order = np.arange(len(train))
    rng.shuffle(order)
    n_val = max(1, int(round(len(train) * fraction)))
    n_val = min(n_val, len(train) - 1)
    val_positions = set(order[:n_val].tolist())
    mask = np.array([i in val_positions for i in range(len(train))])
    inner_val = train.loc[mask].reset_index(drop=True)
    inner_train = train.loc[~mask].reset_index(drop=True)
    info = {
        "target_records": n_val,
        "n_groups_total": len(train),
        "n_groups_inner_val": n_val,
        "fraction_achieved": len(inner_val) / len(train),
    }
    return inner_train, inner_val, info


def inner_split_pcsv(train: pd.DataFrame, seed: int, fraction: float = INNER_FRACTION):
    return _group_holdout(train, train["PDB"].astype(str).str.upper(), seed, fraction)


def locate_icsv_group_map(dataset: str) -> Path | None:
    candidates = [
        SPLIT_ROOT / dataset / "icsv_group_map.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def inner_split_icsv(train: pd.DataFrame, dataset: str, seed: int, fraction: float = INNER_FRACTION):
    path = locate_icsv_group_map(dataset)
    if path is None:
        raise ICSVGroupMapMissing(
            f"{dataset} 没有可复用的 ICSV 连通组映射。"
            "data_split/output 只有 split_summary.csv 的组计数，"
            "不能从外层折成员关系反推完整连通组。"
            "请用 three_splits.py 在同一次聚类中生成 "
            f"{SPLIT_ROOT / dataset / 'icsv_group_map.csv'}，"
            "其中必须含 record_uid 和 group_id。"
        )
    mapping = read_csv(path)
    if not {"record_uid", "group_id"}.issubset(mapping.columns):
        raise ProtocolError(f"{path} 需要 record_uid 和 group_id 列")
    lookup = dict(zip(mapping["record_uid"], mapping["group_id"]))
    missing = [uid for uid in train["record_uid"] if uid not in lookup]
    if missing:
        raise ProtocolError(f"ICSV 组映射缺少外层训练记录，例如 {missing[:5]}")
    labels = [lookup[uid] for uid in train["record_uid"]]
    return _group_holdout(train, labels, seed, fraction)


def inner_split_agcv(train: pd.DataFrame, dataset: str, seed: int):
    """Read the inner files in this seed's directory whose record_uid set matches outer train.

    Antigen groups can put the same outer train in more than one seed. The inner split
    is generated with `outer_seed * 1000 + fold_id`, so the first seed on disk is not a substitute.
    """
    if int(seed) not in SEEDS:
        raise ProtocolError(f"种子不在既定协议内: {seed}")
    wanted = set(train["record_uid"].astype(str))
    master = load_master(dataset)
    directory = SPLIT_ROOT / dataset / str(int(seed)) / "AGCV"
    matches = []
    for fold in OUTER_FOLDS:
        inner_train_path = directory / f"fold_{fold}_inner_train.csv"
        inner_val_path = directory / f"fold_{fold}_inner_val.csv"
        if not inner_train_path.is_file() or not inner_val_path.is_file():
            continue
        inner_train = _read_aligned_fold(inner_train_path, master)
        inner_val = _read_aligned_fold(inner_val_path, master)
        found = set(inner_train["record_uid"]) | set(inner_val["record_uid"])
        if found != wanted:
            continue
        matches.append((fold, inner_train, inner_val, inner_train_path, inner_val_path))
    if len(matches) != 1:
        raise ProtocolError(
            f"{directory} 里与当前外层训练集一致的 AGCV 内部划分有 {len(matches)} 套，需要恰好 1 套"
        )
    fold, inner_train, inner_val, inner_train_path, inner_val_path = matches[0]
    info = {
        "source": "AGCV fold files for the requested seed",
        "seed": int(seed),
        "fold": int(fold),
        "inner_train_path": str(inner_train_path),
        "inner_val_path": str(inner_val_path),
        "target_records": len(inner_val),
        "n_groups_inner_val": len(inner_val),
        "fraction_achieved": (len(inner_val) / len(train)) if len(train) else float("nan"),
    }
    return inner_train, inner_val, info


def make_inner_split(train: pd.DataFrame, dataset: str, split_type: str, seed: int):
    if split_type == "RCSV":
        return inner_split_rcsv(train, seed)
    if split_type == "PCSV":
        return inner_split_pcsv(train, seed)
    if split_type == "ICSV":
        return inner_split_icsv(train, dataset, seed)
    if split_type == "AGCV":
        return inner_split_agcv(train, dataset, seed)
    raise ProtocolError(f"未知划分类型: {split_type}")


def assert_isolation(
    outer_train: pd.DataFrame,
    outer_val: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    split_type: str,
) -> None:
    outer_train_ids = set(outer_train["record_uid"])
    outer_val_ids = set(outer_val["record_uid"])
    inner_train_ids = set(inner_train["record_uid"])
    inner_val_ids = set(inner_val["record_uid"])
    if outer_train_ids & outer_val_ids:
        raise ProtocolError("外层训练与外层评估相交")
    if inner_train_ids & inner_val_ids:
        raise ProtocolError("内部训练与内部验证相交")
    if not inner_train_ids <= outer_train_ids or not inner_val_ids <= outer_train_ids:
        raise ProtocolError("内部子集超出外层训练集")
    if inner_train_ids | inner_val_ids != outer_train_ids:
        raise ProtocolError("内部子集没有覆盖完整外层训练集")
    if split_type == "PCSV":
        shared = set(inner_train["PDB"].str.upper()) & set(inner_val["PDB"].str.upper())
        if shared:
            raise ProtocolError(f"同一 PDB 同时进入内部训练和内部验证: {sorted(shared)[:5]}")
    if split_type == "ICSV":
        if "_group" in inner_train.columns and "_group" in inner_val.columns:
            shared_groups = set(inner_train["_group"]) & set(inner_val["_group"])
        else:
            shared_groups = set()
            try:
                dataset = str(outer_train["record_uid"].iloc[0]).split(":", 1)[0]
                path = locate_icsv_group_map(dataset)
                if path is not None:
                    mapping = read_csv(path)
                    lookup = dict(zip(mapping["record_uid"], mapping["group_id"]))
                    shared_groups = {
                        lookup[uid] for uid in inner_train["record_uid"] if uid in lookup
                    } & {
                        lookup[uid] for uid in inner_val["record_uid"] if uid in lookup
                    }
            except Exception:  # pragma: no cover - defensive fallback
                shared_groups = set()
        if shared_groups:
            raise ProtocolError(f"同一 ICSV 组被拆开: {sorted(shared_groups)[:5]}")


def y_true(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["ddG"], errors="coerce").to_numpy(dtype=float)


def regression_metrics(y_true_values: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true_values) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0:
        return {"rmse": np.nan, "mae": np.nan, "pcc": np.nan, "spearman": np.nan, "n": 0}
    truth = y_true_values[mask]
    pred = y_pred[mask]
    residual = pred - truth
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    pcc = _corr(truth, pred)
    spearman = _corr(pd.Series(truth).rank().to_numpy(), pd.Series(pred).rank().to_numpy())
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "spearman": spearman, "n": n}


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def empty_prediction_frame(rows: pd.DataFrame, **fields: object) -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype=object) for column in OOF_COLUMNS})
    if len(rows) == 0:
        return frame
    out = pd.DataFrame(
        {
            "record_uid": rows["record_uid"].astype(str),
            "ID": rows["ID"].astype(str),
            "PDB": rows["PDB"].astype(str),
            "y_true": y_true(rows),
            "y_pred": np.nan,
            "prediction_status": "not_run",
            "failure_reason": "",
        }
    )
    for key, value in fields.items():
        out[key] = value
    for column in OOF_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out.loc[:, list(OOF_COLUMNS)]


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class FoldCheck:
    dataset: str
    split_type: str
    split_seed: int
    outer_fold: int
    n_outer_train: int
    n_outer_val: int
    n_inner_train: int
    n_inner_val: int
    inner_status: str
    inner_detail: str


def check_partition(dataset: str, seed: int, split_type: str) -> dict[str, object]:
    master = load_master(dataset)
    seen_val: list[str] = []
    fold_sizes = []
    for fold in OUTER_FOLDS:
        train, val = load_outer_fold(dataset, seed, split_type, fold)
        if set(train["record_uid"]) | set(val["record_uid"]) != set(master["record_uid"]):
            raise ProtocolError(f"{dataset} {seed} {split_type} fold {fold} 没有覆盖正式 CSV")
        fold_sizes.append({"fold": fold, "n_train": len(train), "n_val": len(val)})
        seen_val.extend(val["record_uid"].tolist())
    if len(seen_val) != len(set(seen_val)) or set(seen_val) != set(master["record_uid"]):
        raise ProtocolError(f"{dataset} {seed} {split_type} 的五折评估集不是正式 CSV 的划分")
    return {"dataset": dataset, "seed": seed, "split_type": split_type, "folds": fold_sizes}
