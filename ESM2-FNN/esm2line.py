import os
import warnings
import argparse
import inspect
import re
from pathlib import Path

import torch

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
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
from torch import nn, optim

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "output" / "SHUJU_clean" / "ESM-FNN"
DEFAULT_ESM_MODEL = os.environ.get("ESM_MODEL")

# R/L are legacy source-data column prefixes for the two interaction sides;
# they do not imply antibody/antigen identity.
_RL_CHAINS = ["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3"]
_RL_MUT = {
    "R1": "R1m", "R2": "R2m", "R3": "R3m", "R4": "R4m",
    "R5": "R5m", "R6": "R6m", "L1": "L1m", "L2": "L2m", "L3": "L3m",
}

CHAIN_CONFIG = {
    "M1101": list(_RL_CHAINS),
    "S1131": list(_RL_CHAINS),
}

MUT_CONFIG = {
    "M1101": dict(_RL_MUT),
    "S1131": dict(_RL_MUT),
}


def parse_args():
    parser = argparse.ArgumentParser(description="ESM-2 + FFN 五折训练（BF16）")
    parser.add_argument("--dataset", required=True, choices=sorted(CHAIN_CONFIG.keys()))
    parser.add_argument("--split", required=True, choices=("RCSV", "PCSV", "ICSV"))
    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
        help="划分根目录，实际读取 {split-root}/{dataset}/{split}/",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="结果根目录，实际写出 {result-root}/{dataset}/{split}/",
    )
    parser.add_argument(
        "--esm-model",
        type=Path,
        default=DEFAULT_ESM_MODEL,
        help="ESM-2 checkpoint；也可通过 ESM_MODEL 环境变量设置",
    )
    parser.add_argument("--gpu", type=int, default=None, help="物理 GPU 编号；若已设 CUDA_VISIBLE_DEVICES 可省略")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def resolve_device(use_cpu, gpu_id):
    if use_cpu or not torch.cuda.is_available():
        return torch.device("cpu")

    # 启动脚本用 CUDA_VISIBLE_DEVICES 隔离单卡时，这里统一用 cuda:0
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


def get_feature(esm_model, batch_converter, device, use_bf16, wt_seq, mut_seq, mut_positions_str):
    positions = [int(p) for p in mut_positions_str.split(",")] if mut_positions_str else []
    data = [("wt", wt_seq), ("mut", mut_seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
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


def preprocess_df(df, wt_cols, dataset_name, mut_config):
    wt_complex, mut_complex, mut_pos_list = [], [], []
    nan_warning_shown = False

    for _, row in tqdm(df.iterrows(), total=len(df), desc="预处理序列"):
        wt_chains = ["" if pd.isna(row[col]) else str(row[col]).strip() for col in wt_cols]
        wt_nonempty = [c for c in wt_chains if c]
        wt_seq = "".join(wt_nonempty) if wt_nonempty else "A"
        wt_complex.append(wt_seq)

        mut_chains = [
            "" if pd.isna(row[mut_config[col]]) else str(row[mut_config[col]]).strip()
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
            print(f"⚠️ 警告: 数据集中存在 NaN 序列，已自动处理 (dataset={dataset_name})")
            nan_warning_shown = True

    df = df.copy()
    df["wt_complex_seq"] = wt_complex
    df["mut_complex_seq"] = mut_complex
    df["mut_positions_in_complex"] = mut_pos_list
    return df


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


def safe_pearsonr(y_true, y_pred):
    try:
        if len(y_true) < 2:
            return np.nan
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return np.nan
        pcc, _ = pearsonr(y_true, y_pred)
        return pcc
    except Exception:
        return np.nan


def save_best_model(model_ffn, result_dir, dataset_name, fold, epoch, rmse, mae, pcc):
    model_path = os.path.join(result_dir, f"fold_{fold}_best_model.pt")
    checkpoint = {
        "fold": fold,
        "best_epoch": epoch,
        "best_rmse": float(rmse) if not np.isnan(rmse) else None,
        "best_mae": float(mae) if not np.isnan(mae) else None,
        "best_pcc": None if (pcc is None or (isinstance(pcc, float) and np.isnan(pcc))) else float(pcc),
        "selection_metric": "pcc",
        "model_state_dict": model_ffn.state_dict(),
        "model_config": {"input_dim": 1280},
        "dataset_name": dataset_name,
        "precision": "bf16",
    }
    torch.save(checkpoint, model_path)
    print(f"  ✅ 已保存当前最优模型权重 -> {model_path}")
    return model_path


def build_prediction_df(val_df, best_preds):
    """写出标准预测列: ID, ddG_true, ddG_pred"""
    if "ID" not in val_df.columns:
        raise KeyError("验证集缺少 ID 列，无法写出预测文件")
    if "ddG" not in val_df.columns:
        raise KeyError("验证集缺少 ddG 列，无法写出预测文件")
    return pd.DataFrame(
        {
            "ID": val_df["ID"].astype(str).to_numpy(),
            "ddG_true": val_df["ddG"].to_numpy(),
            "ddG_pred": best_preds,
        }
    )


def load_fold_df(fold_path, dataset_name):
    fold_df = pd.read_csv(fold_path)
    wt_cols = CHAIN_CONFIG[dataset_name]
    mut_cols = list(MUT_CONFIG[dataset_name].values())
    required = wt_cols + mut_cols + ["ddG", "ID"]
    missing = [c for c in required if c not in fold_df.columns]
    if missing:
        raise KeyError(f"{fold_path} 缺少必要列: {missing}")
    return fold_df


def train_one_fold(
    train_path,
    val_path,
    fold,
    *,
    dataset_name,
    mut_config,
    esm_model,
    batch_converter,
    device,
    use_bf16,
    result_dir,
    epochs,
    lr,
    split_name,
):
    print(f"\n=== Fold {fold} 开始 ===")
    train_df = load_fold_df(train_path, dataset_name)
    val_df = load_fold_df(val_path, dataset_name)

    wt_cols = CHAIN_CONFIG[dataset_name]
    train_df = preprocess_df(train_df, wt_cols, dataset_name, mut_config)
    val_df = preprocess_df(val_df, wt_cols, dataset_name, mut_config)

    print("提取训练集特征...")
    X_train = np.array(
        [
            get_feature(
                esm_model,
                batch_converter,
                device,
                use_bf16,
                row["wt_complex_seq"],
                row["mut_complex_seq"],
                row["mut_positions_in_complex"],
            )
            for _, row in train_df.iterrows()
        ],
        dtype=np.float32,
    )
    y_train = train_df["ddG"].values.astype(np.float32)

    print("提取验证集特征...")
    X_val = np.array(
        [
            get_feature(
                esm_model,
                batch_converter,
                device,
                use_bf16,
                row["wt_complex_seq"],
                row["mut_complex_seq"],
                row["mut_positions_in_complex"],
            )
            for _, row in val_df.iterrows()
        ],
        dtype=np.float32,
    )
    y_val = val_df["ddG"].values.astype(np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    model_ffn = SimpleFFN().to(device)
    optimizer = optim.Adam(model_ffn.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_rmse = np.nan
    best_mae = np.nan
    best_pcc = -float("inf")
    best_epoch = -1
    best_preds = None
    best_model_path = None
    last_rmse, last_mae, last_pcc, last_preds = np.nan, np.nan, np.nan, None

    for epoch in range(epochs):
        model_ffn.train()
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            pred = model_ffn(X_train_t)
            loss = criterion(pred.float(), y_train_t)
        loss.backward()
        optimizer.step()

        model_ffn.eval()
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                val_pred = model_ffn(X_val_t).float().detach().cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        pcc = safe_pearsonr(y_val, val_pred)
        last_rmse, last_mae, last_pcc, last_preds = rmse, mae, pcc, val_pred.copy()

        if np.isnan(pcc):
            print(f"  Epoch {epoch:2d} | Val RMSE: {rmse:.4f} | MAE: {mae:.4f} | PCC: nan")
        else:
            print(f"  Epoch {epoch:2d} | Val RMSE: {rmse:.4f} | MAE: {mae:.4f} | PCC: {pcc:.4f}")

        if not np.isnan(pcc) and pcc > best_pcc:
            best_rmse = rmse
            best_mae = mae
            best_pcc = pcc
            best_epoch = epoch
            best_preds = val_pred.copy()
            best_model_path = save_best_model(
                model_ffn=model_ffn,
                result_dir=result_dir,
                dataset_name=dataset_name,
                fold=fold,
                epoch=best_epoch,
                rmse=best_rmse,
                mae=best_mae,
                pcc=best_pcc,
            )

    if best_preds is None:
        print("  ⚠️ 全程 PCC 无效，回退保存最后一个 epoch 的权重与预测")
        best_rmse, best_mae, best_pcc = last_rmse, last_mae, last_pcc
        best_epoch = epochs - 1
        best_preds = last_preds
        best_model_path = save_best_model(
            model_ffn=model_ffn,
            result_dir=result_dir,
            dataset_name=dataset_name,
            fold=fold,
            epoch=best_epoch,
            rmse=best_rmse,
            mae=best_mae,
            pcc=best_pcc,
        )

    if best_pcc == -float("inf"):
        best_pcc = np.nan

    val_save_df = build_prediction_df(val_df, best_preds)
    pred_path = os.path.join(
        result_dir,
        f"{dataset_name}_{split_name}_fold{fold}_pred.csv",
    )
    val_save_df.to_csv(pred_path, index=False)

    print(f"  Fold {fold} 最优 Epoch {best_epoch}（按最高 PCC）已保存 -> {pred_path}")
    print(f"  Fold {fold} 最优模型权重已保存 -> {best_model_path}")
    if np.isnan(best_pcc):
        print(f"  Best PCC=nan | Best RMSE={best_rmse:.4f} | Best MAE={best_mae:.4f}")
    else:
        print(f"  Best PCC={best_pcc:.4f} | Best RMSE={best_rmse:.4f} | Best MAE={best_mae:.4f}")

    return best_rmse, best_mae, best_pcc, best_epoch, best_model_path, val_save_df


def discover_fold_pairs(split_dir):
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"划分目录不存在: {split_dir}")

    train_folds = {}
    val_folds = {}

    for filename in os.listdir(split_dir):
        train_match = re.fullmatch(r"train_fold_(\d+)\.csv", filename)
        val_match = re.fullmatch(r"val_fold_(\d+)\.csv", filename)
        if train_match:
            train_folds[int(train_match.group(1))] = os.path.join(split_dir, filename)
        elif val_match:
            val_folds[int(val_match.group(1))] = os.path.join(split_dir, filename)

    train_ids = set(train_folds)
    val_ids = set(val_folds)
    missing_val = sorted(train_ids - val_ids)
    missing_train = sorted(val_ids - train_ids)
    if missing_val or missing_train:
        messages = []
        if missing_val:
            messages.append(f"缺少 val 文件的 fold: {missing_val}")
        if missing_train:
            messages.append(f"缺少 train 文件的 fold: {missing_train}")
        raise FileNotFoundError(f"{split_dir} 中的训练/验证文件不成对：" + "；".join(messages))

    fold_ids = sorted(train_ids & val_ids)
    if not fold_ids:
        raise FileNotFoundError(
            f"{split_dir} 中没有找到成对的 train_fold_N.csv 和 val_fold_N.csv"
        )
    return [(fold, train_folds[fold], val_folds[fold]) for fold in fold_ids]


def main():
    args = parse_args()
    dataset_name = args.dataset
    if dataset_name not in MUT_CONFIG:
        raise KeyError(f"MUT_CONFIG 未定义数据集: {dataset_name}")

    if args.cpu_threads is not None:
        torch.set_num_threads(max(1, args.cpu_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))

    if args.esm_model is None or not args.esm_model.is_file():
        raise FileNotFoundError(
            "ESM-2 checkpoint not found. Set ESM_MODEL or pass --esm-model."
        )

    split_dir = os.path.join(args.split_root, dataset_name, args.split)
    result_dir = os.path.join(args.result_root, dataset_name, args.split)
    os.makedirs(result_dir, exist_ok=True)

    device = resolve_device(args.use_cpu, args.gpu)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and not use_bf16:
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16，已停止以避免退回 FP32。")

    print(f"数据集: {dataset_name}")
    print(f"划分目录: {split_dir}")
    print(f"结果目录: {result_dir}")
    print(f"设备: {device} | 精度: BF16 | CPU threads: {torch.get_num_threads()}")

    print("加载 ESM-2 650M 模型...")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(str(args.esm_model))
    esm_model.eval()
    esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()

    fold_pairs = discover_fold_pairs(split_dir)
    num_folds = len(fold_pairs)
    run_label = "单次 train/val 划分" if num_folds == 1 else f"{num_folds}-fold"
    print(f"开始运行 {dataset_name}/{args.split} ESM-2 + FFN ({run_label})")
    print(f"检测到的 fold: {[fold for fold, _, _ in fold_pairs]}")

    rmse_list, mae_list, pcc_list = [], [], []
    fold_summary = []
    all_pred_frames = []

    for fold, train_path, val_path in fold_pairs:
        r, m, p, e, model_path, pred_df = train_one_fold(
            train_path,
            val_path,
            fold,
            dataset_name=dataset_name,
            mut_config=MUT_CONFIG[dataset_name],
            esm_model=esm_model,
            batch_converter=batch_converter,
            device=device,
            use_bf16=use_bf16,
            result_dir=result_dir,
            epochs=args.epochs,
            lr=args.lr,
            split_name=args.split,
        )
        rmse_list.append(r)
        mae_list.append(m)
        pcc_list.append(p)
        fold_summary.append(
            {
                "fold": fold,
                "best_epoch": e,
                "rmse": r,
                "mae": m,
                "pcc": p,
                "best_model_path": model_path,
            }
        )
        all_pred_frames.append(pred_df)

    avg_rmse = float(np.mean(rmse_list))
    avg_mae = float(np.mean(mae_list))
    avg_pcc = float(np.nanmean(pcc_list)) if not np.all(np.isnan(pcc_list)) else np.nan
    std_rmse = float(np.std(rmse_list, ddof=1))
    std_mae = float(np.std(mae_list, ddof=1))
    std_pcc = (
        float(np.nanstd(pcc_list, ddof=1))
        if not np.all(np.isnan(pcc_list))
        else np.nan
    )

    summary_df = pd.DataFrame(fold_summary)
    summary_csv_path = os.path.join(result_dir, "fold_metrics_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    all_pred_path = os.path.join(result_dir, "predicate_all_folds.csv")
    pd.concat(all_pred_frames, ignore_index=True).to_csv(all_pred_path, index=False)

    print("\n" + "=" * 60)
    print(f"{dataset_name}/{args.split} 最终结果（{run_label}，基于每个 fold 最优 epoch）")
    print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"MAE : {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"PCC : {avg_pcc:.4f} ± {std_pcc:.4f}" if not np.isnan(avg_pcc) else "PCC : nan")
    print("=" * 60)

    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"ESM-2 + FFN baseline on {dataset_name}\n")
        f.write(f"Split type: {args.split}\n")
        f.write("Precision: BF16\n")
        f.write("Selection metric: highest validation PCC\n")
        f.write(f"Run mode: {run_label}\n")
        f.write(f"Detected folds: {[fold for fold, _, _ in fold_pairs]}\n")
        f.write(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"MAE : {avg_mae:.4f} ± {std_mae:.4f}\n")
        if np.isnan(avg_pcc):
            f.write("PCC : nan\n")
        else:
            f.write(f"PCC : {avg_pcc:.4f} ± {std_pcc:.4f}\n")
        f.write(f"Per-fold best RMSE: {rmse_list}\n")
        f.write(f"Per-fold best MAE : {mae_list}\n")
        f.write(f"Per-fold best PCC : {pcc_list}\n")
        f.write(f"Per-fold best epoch: {[x['best_epoch'] for x in fold_summary]}\n")
        f.write(f"Per-fold best model path: {[x['best_model_path'] for x in fold_summary]}\n")
        f.write(f"Per-fold metrics csv: {summary_csv_path}\n")
        f.write(f"All-fold predictions csv: {all_pred_path}\n")
        f.write("Prediction columns: ID,ddG_true,ddG_pred\n")

    print(f"✅ 全部完成！结果保存在 {result_dir}/")
    print(
        f"✅ 验证集预测文件: {dataset_name}_{args.split}_fold*_pred.csv "
        "(列: ID,ddG_true,ddG_pred)"
    )
    print(f"✅ 五折汇总预测: {all_pred_path}")
    print("✅ 最优模型权重: fold_*_best_model.pt")
    print(f"✅ 指标汇总: {summary_csv_path}")


if __name__ == "__main__":
    main()
