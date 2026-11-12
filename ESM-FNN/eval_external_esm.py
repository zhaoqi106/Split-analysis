"""
External ESM-FNN inference with the five fold_*_best_model checkpoints.

  SARS  <- M1101/esmfnn/family/fold_*_best_model.pt
  ATLAS <- S1131/esmfnn/family/fold_*_best_model.pt

Tables use sequences only. Columns: ID, optional PDB, optional ddG,
R1-R6, L1-L3, R1m-R6m, L1m-L3m.

  python eval_external_esm.py --external SARS --weights-from M1101
  python eval_external_esm.py --external ATLAS --weights-from S1131
  python eval_external_esm.py --run-both
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import warnings

import torch

# Older ESM checkpoints need weights_only=False on PyTorch 2.6+.
try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    sig = inspect.signature(_original_torch_load)
    if "weights_only" in sig.parameters and "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import esm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
from torch import nn

warnings.filterwarnings("ignore")

DATA_ROOT = "/data/external"
ESM_MODEL_DEFAULT = "/data/ESM-FNN/esm2_t33_650M_UR50D.pt"

EXTERNAL_DATASETS = ("SARS", "ATLAS")
WEIGHT_DATASETS = ("S1131", "S645", "S4169", "M1101")
DEFAULT_PAIRS = (
    ("SARS", "M1101"),
    ("ATLAS", "S1131"),
)

_RL_CHAINS = ["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3"]
_RL_MUT = {
    "R1": "R1m",
    "R2": "R2m",
    "R3": "R3m",
    "R4": "R4m",
    "R5": "R5m",
    "R6": "R6m",
    "L1": "L1m",
    "L2": "L2m",
    "L3": "L3m",
}

CHAIN_CONFIG = {name: list(_RL_CHAINS) for name in (*EXTERNAL_DATASETS, *WEIGHT_DATASETS)}
MUT_CONFIG = {name: dict(_RL_MUT) for name in CHAIN_CONFIG}


class SimpleFFN(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser(description="ESM-FFN 外部验证：加载多折 best_model 推理")
    p.add_argument("--external", choices=EXTERNAL_DATASETS, default=None, help="外部验证集")
    p.add_argument(
        "--weights-from",
        choices=WEIGHT_DATASETS,
        default=None,
        help="权重来源训练集",
    )
    p.add_argument("--split", default="family", help="权重所在划分，默认 family")
    p.add_argument(
        "--run-both",
        action="store_true",
        help="依次跑 SARS<-M1101 与 ATLAS<-S1131",
    )
    p.add_argument("--data-root", default=DATA_ROOT, help="Root for external tables and trained weights.")
    p.add_argument(
        "--model-root",
        default=None,
        help="覆盖权重目录；默认 {data-root}/{weights}/esmfnn/{split}",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="输出目录；默认 {data-root}/{external}/esmfnn/external_from_{weights}_{split}",
    )
    p.add_argument("--esm-model", default=ESM_MODEL_DEFAULT)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=None)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument(
        "--no-bf16",
        action="store_true",
        help="特征提取关闭 FP16（默认 CUDA 上用 FP16，与训练一致）",
    )
    p.add_argument(
        "--invert-pred",
        action="store_true",
        help="对外预测取负（标签方向相反时用）",
    )
    return p.parse_args()


def resolve_jobs(args):
    if args.run_both:
        return list(DEFAULT_PAIRS)
    if args.external and args.weights_from:
        return [(args.external, args.weights_from)]
    raise SystemExit(
        "请指定 --external 与 --weights-from，或使用 --run-both\n"
        "示例: python eval_external_esm.py --external SARS --weights-from M1101"
    )


def parse_fold_list(folds_str: str):
    folds = []
    for part in str(folds_str).split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    if not folds:
        raise ValueError(f"无效 --folds: {folds_str}")
    return folds


def resolve_device(use_cpu, gpu_id):
    if use_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None and gpu_id is None:
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if gpu_id is None:
        return torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    if gpu_id >= n_gpu:
        raise ValueError(f"指定的 GPU_ID={gpu_id} 不存在。当前可见 GPU 数量: {n_gpu}")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def validate_input_columns(df: pd.DataFrame, dataset_name: str):
    wt_cols = CHAIN_CONFIG[dataset_name]
    mut_cols = [MUT_CONFIG[dataset_name][c] for c in wt_cols]
    required = wt_cols + mut_cols + ["ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{dataset_name} 主表缺少列: {missing}\n"
            f"期望 wt: {wt_cols}\n期望 mut: {mut_cols}\n另需 ID"
        )


def prepare_external_df(df: pd.DataFrame, tag: str):
    """Check sequence columns. If ddG is missing, write predictions only."""
    validate_input_columns(df, tag)
    out = df.copy()
    out["ID"] = out["ID"].astype(str).str.strip()

    has_label = "ddG" in out.columns and not out["ddG"].isna().all()
    if not has_label:
        print(f"⚠️ {tag} 无有效 ddG，仅保存预测，不计算指标")
        out["ddG"] = np.nan
    else:
        out["ddG"] = pd.to_numeric(out["ddG"], errors="coerce")
        n_bad = int(out["ddG"].isna().sum())
        if n_bad:
            raise ValueError(f"{tag} 有 {n_bad} 条 ddG 非数值/缺失")
    return out, has_label


def preprocess_df(df: pd.DataFrame, dataset_name: str):
    wt_cols = CHAIN_CONFIG[dataset_name]
    mut_map = MUT_CONFIG[dataset_name]

    wt_complex, mut_complex, mut_pos_list = [], [], []
    nan_warning_shown = False

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"预处理[{dataset_name}]"):
        wt_chains = ["" if pd.isna(row[col]) else str(row[col]).strip() for col in wt_cols]
        wt_nonempty = [c for c in wt_chains if c]
        wt_seq = "".join(wt_nonempty) if wt_nonempty else "A"
        wt_complex.append(wt_seq)

        mut_chains = [
            "" if pd.isna(row[mut_map[col]]) else str(row[mut_map[col]]).strip()
            for col in wt_cols
        ]
        mut_nonempty = [c for c in mut_chains if c]
        mut_seq = "".join(mut_nonempty) if mut_nonempty else "A"
        mut_complex.append(mut_seq)

        positions = []
        global_pos = 0
        for k in range(len(wt_nonempty)):
            wt_c = wt_nonempty[k]
            mut_c = mut_nonempty[k] if k < len(mut_nonempty) else ""
            min_len = min(len(wt_c), len(mut_c))
            for j in range(min_len):
                if wt_c[j] != mut_c[j]:
                    positions.append(global_pos + j)
            global_pos += len(wt_c)

        mut_pos_list.append(",".join(map(str, positions)) if positions else "")

        if not nan_warning_shown and any(pd.isna(row[col]) for col in wt_cols):
            print(f"⚠️ 警告: 存在 NaN wt 链，已跳过空链 (dataset={dataset_name})")
            nan_warning_shown = True

    out = df.copy()
    out["wt_complex_seq"] = wt_complex
    out["mut_complex_seq"] = mut_complex
    out["mut_positions_in_complex"] = mut_pos_list
    return out


def get_feature(esm_model, batch_converter, device, use_fp16, wt_seq, mut_seq, mut_positions_str):
    positions = [int(p) for p in mut_positions_str.split(",")] if mut_positions_str else []
    data = [("wt", wt_seq), ("mut", mut_seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_fp16 and device.type == "cuda",
        ):
            results = esm_model(batch_tokens, repr_layers=[33])
            reps = results["representations"][33]

    if not positions:
        return np.zeros(1280, dtype=np.float32)

    delta_sum = np.zeros(1280, dtype=np.float32)
    for pos in positions:
        token_idx = pos + 1
        if token_idx >= reps.shape[1]:
            print(f"⚠️ 位置越界跳过: pos={pos} (seq_len={len(wt_seq)})")
            continue
        delta = reps[1, token_idx] - reps[0, token_idx]
        delta_sum += delta.float().cpu().numpy()

    return delta_sum.astype(np.float32)


def extract_features(df, esm_model, batch_converter, device, use_fp16):
    print("提取外部验证集特征...")
    X = np.array(
        [
            get_feature(
                esm_model,
                batch_converter,
                device,
                use_fp16,
                row["wt_complex_seq"],
                row["mut_complex_seq"],
                row["mut_positions_in_complex"],
            )
            for _, row in tqdm(df.iterrows(), total=len(df), desc="ESM特征")
        ],
        dtype=np.float32,
    )
    print(
        f"  X shape={X.shape} | std={X.std():.6f} | "
        f"全零样本={int(np.sum(np.all(np.isclose(X, 0), axis=1)))}"
    )
    return X


def find_fold_checkpoints(model_root, folds):
    found, missing = [], []
    for fold in folds:
        path = os.path.join(model_root, f"fold_{fold}_best_model.pt")
        if os.path.isfile(path):
            found.append((fold, path))
        else:
            missing.append(fold)
    if missing:
        raise FileNotFoundError(
            f"缺少 fold 权重 {missing}，期望: {model_root}/fold_*_best_model.pt"
        )
    return found


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    out = {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "PCC": np.nan,
        "MSE": mse,
        "R2": float(r2_score(y_true, y_pred)),
    }
    if len(y_true) >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        try:
            pcc, _ = pearsonr(y_true, y_pred)
            if not (np.isnan(pcc) or np.isinf(pcc)):
                out["PCC"] = float(pcc)
        except Exception:
            pass
    return out


def format_metrics(m):
    """Log RMSE, MAE, then PCC."""
    def _fmt(key):
        v = m.get(key, np.nan)
        return f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "nan"

    return f"RMSE={_fmt('RMSE')} | MAE={_fmt('MAE')} | PCC={_fmt('PCC')}"


def aggregate_fold_metrics(fold_metrics):
    """Mean and sample standard deviation (ddof=1) of RMSE, MAE, and PCC across folds."""
    keys = ("RMSE", "MAE", "PCC")
    mean_out, std_out = {}, {}
    for k in keys:
        vals = np.asarray([m[k] for m in fold_metrics], dtype=np.float64)
        mean_out[k] = float(np.nanmean(vals)) if len(vals) else np.nan
        if len(vals) > 1 and np.sum(~np.isnan(vals)) > 1:
            std_out[k] = float(np.nanstd(vals, ddof=1))
        else:
            std_out[k] = 0.0 if len(vals) == 1 else np.nan
    return mean_out, std_out


def print_fold_metrics_table(fold_ids, fold_metrics, mean_m, std_m):
    print("\n" + "-" * 72)
    print(f"{'Fold':>8} | {'RMSE':>10} | {'MAE':>10} | {'PCC':>10}")
    print("-" * 72)
    for fold, m in zip(fold_ids, fold_metrics):
        print(
            f"{fold:>8} | {m['RMSE']:10.4f} | {m['MAE']:10.4f} | {m['PCC']:10.4f}"
        )
    print("-" * 72)
    print(
        f"{'mean':>8} | {mean_m['RMSE']:10.4f} | {mean_m['MAE']:10.4f} | "
        f"{mean_m['PCC']:10.4f}"
    )
    print(
        f"{'std':>8} | {std_m['RMSE']:10.4f} | {std_m['MAE']:10.4f} | "
        f"{std_m['PCC']:10.4f}"
    )
    print("-" * 72)
    print(
        f"多折汇总: RMSE={mean_m['RMSE']:.4f}±{std_m['RMSE']:.4f} | "
        f"MAE={mean_m['MAE']:.4f}±{std_m['MAE']:.4f} | "
        f"PCC={mean_m['PCC']:.4f}±{std_m['PCC']:.4f}"
    )


def load_ffn(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_ffn = SimpleFFN().to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_ffn.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_ffn.load_state_dict(checkpoint)
    model_ffn.eval()
    return model_ffn


def run_one_job(args, external, weights_from, esm_model, batch_converter, device, use_fp16):
    split = args.split
    data_root = os.path.abspath(args.data_root)
    master_csv = os.path.join(data_root, external, f"{external}.csv")
    model_root = os.path.abspath(
        args.model_root
        or os.path.join(data_root, weights_from, "esmfnn", split)
    )
    out_root = os.path.abspath(
        args.output_root
        or os.path.join(
            data_root,
            external,
            "esmfnn",
            f"external_from_{weights_from}_{split}",
        )
    )
    folds = parse_fold_list(args.folds)
    os.makedirs(out_root, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"外测: {external}  |  权重: {weights_from}/esmfnn/{split}")
    print("=" * 88)
    print(f"📋 主表: {master_csv}")
    print(f"📂 权重: {model_root}")
    print(f"📂 输出: {out_root}")

    if not os.path.isfile(master_csv):
        raise FileNotFoundError(f"主表不存在: {master_csv}")

    df = pd.read_csv(master_csv)
    df, has_label = prepare_external_df(df, tag=external)
    df = preprocess_df(df, external)
    print(
        f"  样本数={len(df)} | has_label={has_label} | "
        f"空突变位点={(df['mut_positions_in_complex'] == '').sum()}"
    )

    ckpts = find_fold_checkpoints(model_root, folds)
    for fold, path in ckpts:
        print(f"  fold_{fold}: {path}")

    X = extract_features(df, esm_model, batch_converter, device, use_fp16)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    summary_rows = []
    pred_matrix = []
    fold_ids = []
    fold_metrics = []

    for fold, ckpt_path in ckpts:
        print(f"\n--- Fold {fold} 推理 ---")
        model_ffn = load_ffn(ckpt_path, device)
        with torch.no_grad():
            y_pred = model_ffn(X_t).detach().cpu().numpy().reshape(-1)
        if args.invert_pred:
            y_pred = -y_pred

        pred_path = os.path.join(out_root, f"predicate_fold{fold}.csv")
        pred_df = pd.DataFrame(
            {
                "ID": df["ID"].astype(str).to_numpy(),
                "ddG_true": df["ddG"].to_numpy() if has_label else np.nan,
                "ddG_pre": y_pred,
            }
        )
        pred_df.to_csv(pred_path, index=False)
        print(f"  → {pred_path}")

        row = {
            "external": external,
            "weights_from": weights_from,
            "split": split,
            "fold": fold,
            "ckpt": ckpt_path,
            "num_samples": len(pred_df),
            "invert_pred": bool(args.invert_pred),
            "RMSE": np.nan,
            "MAE": np.nan,
            "PCC": np.nan,
            "MSE": np.nan,
            "R2": np.nan,
        }
        if has_label:
            m = calc_metrics(df["ddG"].to_numpy(), y_pred)
            row.update(m)
            fold_ids.append(fold)
            fold_metrics.append(m)
            print(f"  Fold {fold} 指标: {format_metrics(m)}")
        else:
            print("  无 ddG 标签，跳过本折指标")
        summary_rows.append(row)
        pred_matrix.append(y_pred)

        del model_ffn
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if has_label and fold_metrics:
        mean_m, std_m = aggregate_fold_metrics(fold_metrics)
        print_fold_metrics_table(fold_ids, fold_metrics, mean_m, std_m)
        summary_rows.append(
            {
                "external": external,
                "weights_from": weights_from,
                "split": split,
                "fold": "folds_mean",
                "ckpt": "-",
                "num_samples": len(df),
                "invert_pred": bool(args.invert_pred),
                **mean_m,
                "MSE": np.nan,
                "R2": np.nan,
            }
        )
        summary_rows.append(
            {
                "external": external,
                "weights_from": weights_from,
                "split": split,
                "fold": "folds_std",
                "ckpt": "-",
                "num_samples": len(df),
                "invert_pred": bool(args.invert_pred),
                **std_m,
                "MSE": np.nan,
                "R2": np.nan,
            }
        )

    pred_matrix = np.asarray(pred_matrix, dtype=np.float64)
    ens_mean = pred_matrix.mean(axis=0)
    ens_median = np.median(pred_matrix, axis=0)
    ens_std = pred_matrix.std(axis=0, ddof=1) if len(pred_matrix) > 1 else np.zeros_like(ens_mean)

    base = {
        "ID": df["ID"].astype(str).to_numpy(),
        "ddG_true": df["ddG"].to_numpy() if has_label else np.full(len(df), np.nan),
    }
    ens_mean_path = os.path.join(out_root, "predicate_ensemble_mean.csv")
    ens_median_path = os.path.join(out_root, "predicate_ensemble_median.csv")
    pd.DataFrame({**base, "ddG_pre": ens_mean}).to_csv(ens_mean_path, index=False)
    pd.DataFrame({**base, "ddG_pre": ens_median}).to_csv(ens_median_path, index=False)

    detail_path = os.path.join(out_root, "predicate_all_folds_detail.csv")
    detail = pd.DataFrame(base)
    for i, (fold, _) in enumerate(ckpts):
        detail[f"pred_fold_{fold}"] = pred_matrix[i]
    detail["pred_mean"] = ens_mean
    detail["pred_median"] = ens_median
    detail["pred_std"] = ens_std
    detail.to_csv(detail_path, index=False)

    print(f"\n  → {ens_mean_path}")
    print(f"  → {ens_median_path}")
    print(f"  → {detail_path}")

    if has_label:
        print("\n集成预测指标（对样本预测取 mean/median 后再算）:")
        for name, preds in (("ensemble_mean", ens_mean), ("ensemble_median", ens_median)):
            m = calc_metrics(df["ddG"].to_numpy(), preds)
            summary_rows.append(
                {
                    "external": external,
                    "weights_from": weights_from,
                    "split": split,
                    "fold": name,
                    "ckpt": "-",
                    "num_samples": len(df),
                    "invert_pred": bool(args.invert_pred),
                    **m,
                }
            )
            print(f"  {name}: {format_metrics(m)}")

    summary_df = pd.DataFrame(summary_rows)
    preferred = [
        "external",
        "weights_from",
        "split",
        "fold",
        "RMSE",
        "MAE",
        "PCC",
        "MSE",
        "R2",
        "num_samples",
        "invert_pred",
        "ckpt",
    ]
    cols = [c for c in preferred if c in summary_df.columns]
    cols += [c for c in summary_df.columns if c not in cols]
    summary_df = summary_df[cols]
    summary_path = os.path.join(out_root, "external_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  → {summary_path}")
    return summary_df


def main():
    args = parse_args()
    jobs = resolve_jobs(args)

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    if len(jobs) > 1 and (args.model_root or args.output_root):
        print("⚠️ --run-both 时忽略 --model-root / --output-root，改用默认路径")
        args.model_root = None
        args.output_root = None

    device = resolve_device(args.use_cpu, args.gpu)
    use_fp16 = (not args.no_bf16) and device.type == "cuda"
    print(f"✅ device={device} | fp16_feat={use_fp16}")
    print(f"加载 ESM-2: {args.esm_model}")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm_model)
    esm_model.eval()
    esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()

    all_summaries = []
    for external, weights_from in jobs:
        summary = run_one_job(
            args,
            external,
            weights_from,
            esm_model,
            batch_converter,
            device,
            use_fp16,
        )
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        merged = pd.concat(all_summaries, ignore_index=True)
        merged_path = os.path.join(os.path.abspath(args.data_root), "esmfnn_external_metrics_all.csv")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"\n全部完成。汇总: {merged_path}")
    else:
        print("\n全部完成。")


if __name__ == "__main__":
    main()
