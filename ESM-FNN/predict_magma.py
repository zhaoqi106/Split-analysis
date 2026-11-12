"""
MAGMA inference. Load the best ABBIND and SKEMPI models and write a summary table.

  /data/external/MAGMA/MAGMA.csv
  wt/{PDB}.pdb
  mut/{ID}.pdb
  Partners: HL_AB means antibody chains H,L then antigen chains A,B.
  Mutation: <wtAA><chain><resnum(+icode)><mutAA>, comma-separated.

  /data/external/attab/ESM/{DATASET}/result/{split}/fold_{k}_best_model.pt
  complexsplits and pdbsplits add an all row (mean and sample standard deviation).
"""

import os
import re
import warnings
import argparse
import inspect
from collections import OrderedDict

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
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

DATASETS = ["ABBIND", "SKEMPI"]
SPLITS = ["antigensplits", "antibodiesplits", "complexsplits", "pdbsplits"]
FIVE_FOLD_SPLITS = {"complexsplits", "pdbsplits"}

GPU_ID = 7
USE_CPU = False
ESM_MODEL_PATH = "/data/ESM-FNN/esm2_t33_650M_UR50D.pt"

MAGMA_DIR = "/data/external/MAGMA"
MAGMA_CSV = os.path.join(MAGMA_DIR, "MAGMA.csv")
WT_DIR = os.path.join(MAGMA_DIR, "wt")
MUT_DIR = os.path.join(MAGMA_DIR, "mut")

MODEL_ROOT = "/data/external/attab/ESM"
OUTPUT_DIR = os.path.join(MODEL_ROOT, "MAGMA")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "MAGMA_all_models_predictions_summary.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map three-letter and nonstandard residues to one letter. Unknown residues become X.
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}

MUT_RE = re.compile(r"^([A-Z])([A-Za-z])(\d+)([A-Za-z]?)([A-Z])$")


def get_device():
    if USE_CPU or not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if GPU_ID is None:
        return torch.device("cuda")
    if GPU_ID >= n_gpu:
        raise ValueError(f"GPU_ID={GPU_ID} 不存在，当前 GPU 数量: {n_gpu}")
    torch.cuda.set_device(GPU_ID)
    return torch.device(f"cuda:{GPU_ID}")


def parse_partners(partners: str):
    """HL_AB -> ['H','L','A','B']. The underscore separates the two sides."""
    partners = str(partners).strip()
    if "_" not in partners:
        raise ValueError(f"Partners 格式异常（缺少 '_'）: {partners}")
    left, right = partners.split("_", 1)
    if not left or not right:
        raise ValueError(f"Partners 左右链为空: {partners}")
    return list(left), list(right), list(left) + list(right)


def parse_mutation_token(token: str):
    token = token.strip()
    m = MUT_RE.fullmatch(token)
    if not m:
        raise ValueError(f"无法解析突变: {token}")
    wt_aa, chain, resnum, icode, mut_aa = m.groups()
    return {
        "wt_aa": wt_aa,
        "chain": chain,
        "resnum": int(resnum),
        "icode": (icode or " ").upper() if icode else " ",
        "mut_aa": mut_aa,
        "raw": token,
    }


def parse_mutations(mutation_str: str):
    if pd.isna(mutation_str) or str(mutation_str).strip() == "":
        return []
    tokens = [t.strip() for t in str(mutation_str).split(",") if t.strip()]
    return [parse_mutation_token(t) for t in tokens]


def read_pdb_chain_residues(pdb_path: str):
    """
    Read residue sequences and numbering by chain.

    chain_seq: {chain: sequence_str}
    chain_index: {chain: {(resnum, icode): seq_pos_0based}}
    """
    chain_residues = {}  # chain -> OrderedDict[(resnum, icode)] = aa1

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("CA", "CΑ"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            if not chain:
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            icode = line[26:27]
            if icode == "" or icode == "\x00":
                icode = " "
            else:
                icode = icode.upper()

            aa1 = AA3_TO_1.get(resname, "X")
            key = (resnum, icode)
            d = chain_residues.setdefault(chain, OrderedDict())
            # Keep the first copy of each residue.
            if key not in d:
                d[key] = aa1

    chain_seq = {}
    chain_index = {}
    for chain, od in chain_residues.items():
        seq = "".join(od.values())
        index = {k: i for i, k in enumerate(od.keys())}
        chain_seq[chain] = seq
        chain_index[chain] = index
    return chain_seq, chain_index


def build_complex_from_pdb(pdb_path: str, chain_order):
    """Concatenate non-empty chains in Partners order. Return (complex_seq, chain_offsets)."""
    chain_seq, chain_index = read_pdb_chain_residues(pdb_path)
    parts = []
    offsets = {}  # chain -> global start index in concatenated nonempty sequence
    global_pos = 0
    for ch in chain_order:
        seq = chain_seq.get(ch, "")
        if not seq:
            continue
        offsets[ch] = {
            "start": global_pos,
            "seq": seq,
            "index": chain_index[ch],
        }
        parts.append(seq)
        global_pos += len(seq)
    complex_seq = "".join(parts) if parts else "A"
    return complex_seq, offsets, chain_seq


def mutation_positions_from_annotation(offsets, mutations, wt_seq_len):
    """Map Mutation sites onto the concatenated sequence as 0-based positions."""
    positions = []
    warnings_local = []
    for mut in mutations:
        ch = mut["chain"]
        if ch not in offsets:
            warnings_local.append(f"链 {ch} 不在拼接序列中: {mut['raw']}")
            continue
        info = offsets[ch]
        key = (mut["resnum"], mut["icode"])
        # If the insertion code is absent, try a blank, then letter keys.
        if key not in info["index"]:
            alt_keys = [
                (mut["resnum"], " "),
                (mut["resnum"], mut["icode"].lower()),
                (mut["resnum"], mut["icode"].upper()),
            ]
            key = next((k for k in alt_keys if k in info["index"]), None)
        if key is None:
            warnings_local.append(f"残基未找到: {mut['raw']}")
            continue
        local_pos = info["index"][key]
        # Check the wild-type amino acid. X is allowed.
        observed = info["seq"][local_pos]
        if observed not in (mut["wt_aa"], "X") and mut["wt_aa"] != "X":
            warnings_local.append(
                f"wt氨基酸不一致 {mut['raw']}: pdb={observed}, mut={mut['wt_aa']}"
            )
        gpos = info["start"] + local_pos
        if gpos < 0 or gpos >= wt_seq_len:
            warnings_local.append(f"全局位置越界: {mut['raw']} -> {gpos}")
            continue
        positions.append(gpos)
    seen = set()
    uniq = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq, warnings_local


def positions_by_sequence_diff(wt_seq, mut_seq):
    positions = []
    for i, (a, b) in enumerate(zip(wt_seq, mut_seq)):
        if a != b:
            positions.append(i)
    return positions


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
        return float(pcc)
    except Exception:
        return np.nan


def format_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return "nan"
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return f"{mean:.6f} ± {std:.6f}"


def discover_fold_models(result_dir):
    """Find fold_{N}_best_model.pt and sort by fold number."""
    if not os.path.isdir(result_dir):
        return []
    folds = []
    for name in os.listdir(result_dir):
        m = re.fullmatch(r"fold_(\d+)_best_model\.pt", name)
        if m:
            folds.append((int(m.group(1)), os.path.join(result_dir, name)))
    folds.sort(key=lambda x: x[0])
    return folds


def get_feature(esm_model, batch_converter, device, wt_seq, mut_seq, mut_positions_str):
    positions = [int(p) for p in mut_positions_str.split(",")] if mut_positions_str else []
    data = [("wt", wt_seq), ("mut", mut_seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
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
        delta_sum += delta.cpu().numpy()
    return delta_sum.astype(np.float32)


def prepare_magma_dataframe(df: pd.DataFrame):
    rename = {c: c.replace("\ufeff", "") for c in df.columns}
    df = df.rename(columns=rename)

    required = ["PDB", "ID", "Partners", "Mutation", "ddG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"MAGMA.csv 缺少列: {missing}")

    wt_seqs, mut_seqs, pos_list, status = [], [], [], []
    n_warn = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="从PDB构建MAGMA序列"):
        pdb_id = str(row["PDB"]).strip()
        sample_id = str(row["ID"]).strip()
        partners = str(row["Partners"]).strip()

        wt_path = os.path.join(WT_DIR, f"{pdb_id}.pdb")
        mut_path = os.path.join(MUT_DIR, f"{sample_id}.pdb")

        if not os.path.isfile(wt_path):
            wt_seqs.append("A")
            mut_seqs.append("A")
            pos_list.append("")
            status.append(f"missing_wt:{wt_path}")
            continue
        if not os.path.isfile(mut_path):
            wt_seqs.append("A")
            mut_seqs.append("A")
            pos_list.append("")
            status.append(f"missing_mut:{mut_path}")
            continue

        try:
            _, _, chain_order = parse_partners(partners)
            wt_seq, wt_offsets, _ = build_complex_from_pdb(wt_path, chain_order)
            mut_seq, _, _ = build_complex_from_pdb(mut_path, chain_order)
            mutations = parse_mutations(row["Mutation"])

            pos_annot, warns = mutation_positions_from_annotation(
                wt_offsets, mutations, len(wt_seq)
            )
            pos_diff = positions_by_sequence_diff(wt_seq, mut_seq)

            # Use annotated Mutation sites, otherwise a sequence difference.
            if pos_annot:
                positions = pos_annot
                note = "ok_annot"
            elif pos_diff:
                positions = pos_diff
                note = "ok_seqdiff_fallback"
            else:
                positions = []
                note = "no_mutation_positions"

            if warns:
                n_warn += 1
                if n_warn <= 20:
                    print(f"⚠️ {sample_id}: {'; '.join(warns[:3])}")
                note = note + ";warn"

            wt_seqs.append(wt_seq)
            mut_seqs.append(mut_seq)
            pos_list.append(",".join(map(str, positions)))
            status.append(note)
        except Exception as e:
            wt_seqs.append("A")
            mut_seqs.append("A")
            pos_list.append("")
            status.append(f"error:{e}")

    out = df.copy()
    out["wt_complex_seq"] = wt_seqs
    out["mut_complex_seq"] = mut_seqs
    out["mut_positions_in_complex"] = pos_list
    out["prep_status"] = status
    return out


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    pcc = safe_pearsonr(y_true, y_pred)
    return rmse, mae, pcc


def load_ffn(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_ffn = SimpleFFN().to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_ffn.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_ffn.load_state_dict(checkpoint)
    model_ffn.eval()
    return model_ffn


def main():
    device = get_device()
    print(f"当前设备: {device}")
    print(f"读取 MAGMA: {MAGMA_CSV}")

    df = pd.read_csv(MAGMA_CSV, encoding="utf-8-sig")
    df = prepare_magma_dataframe(df)

    print("预处理状态统计:")
    print(df["prep_status"].value_counts(dropna=False))
    print("空突变位点样本数:", int((df["mut_positions_in_complex"] == "").sum()))
    print("总样本数:", len(df))

    print("加载 ESM-2 650M...")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_PATH)
    esm_model.eval()
    esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()

    print("提取 MAGMA 特征（全数据集只提一次）...")
    X = np.array(
        [
            get_feature(
                esm_model,
                batch_converter,
                device,
                row["wt_complex_seq"],
                row["mut_complex_seq"],
                row["mut_positions_in_complex"],
            )
            for _, row in tqdm(df.iterrows(), total=len(df), desc="ESM特征")
        ],
        dtype=np.float32,
    )
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_true = df["ddG"].values.astype(np.float32)

    print("X shape:", X.shape)
    print("全零特征样本数:", int(np.sum(np.all(np.isclose(X, 0), axis=1))))

    # Free the ESM model before the small FFN.
    del esm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_rows = []
    metric_cache = {}  # (dataset, split) -> list of (fold, rmse, mae, pcc)

    for dataset in DATASETS:
        for spli in SPLITS:
            result_dir = os.path.join(MODEL_ROOT, dataset, "result", spli)
            fold_models = discover_fold_models(result_dir)
            if not fold_models:
                print(f"❌ 未找到模型: {result_dir}/fold_*_best_model.pt ，跳过")
                continue

            print(f"\n=== {dataset} / {spli} | folds={[f for f, _ in fold_models]} ===")
            fold_metrics = []

            for fold, model_path in fold_models:
                model_name = f"{dataset}_{spli}_fold{fold}"
                print(f"加载: {model_path}")
                model_ffn = load_ffn(model_path, device)

                with torch.no_grad():
                    pred = model_ffn(X_t).detach().cpu().numpy().astype(np.float64)

                rmse, mae, pcc = compute_metrics(y_true, pred)
                fold_metrics.append((fold, rmse, mae, pcc))
                print(
                    f"  {model_name} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
                    f"PCC={pcc:.4f}" if not np.isnan(pcc) else
                    f"  {model_name} | RMSE={rmse:.4f} | MAE={mae:.4f} | PCC=nan"
                )

                for i, sample_id in enumerate(df["ID"].astype(str).tolist()):
                    all_rows.append({
                        "model": model_name,
                        "ID": sample_id,
                        "pred": float(pred[i]),
                        "true": float(y_true[i]),
                        "RMSE": rmse,
                        "MAE": mae,
                        "PCC": pcc,
                    })

            metric_cache[(dataset, spli)] = fold_metrics

            # For five-fold splits, add an all row with mean and standard deviation.
            if spli in FIVE_FOLD_SPLITS and len(fold_metrics) > 0:
                rmses = [x[1] for x in fold_metrics]
                maes = [x[2] for x in fold_metrics]
                pccs = [x[3] for x in fold_metrics]
                all_rows.append({
                    "model": f"{dataset}_{spli}_all",
                    "ID": "all",
                    "pred": np.nan,
                    "true": np.nan,
                    "RMSE": format_mean_std(rmses),
                    "MAE": format_mean_std(maes),
                    "PCC": format_mean_std(pccs),
                })
                print(
                    f"  {dataset}_{spli}_all | "
                    f"RMSE={format_mean_std(rmses)} | "
                    f"MAE={format_mean_std(maes)} | "
                    f"PCC={format_mean_std(pccs)}"
                )

    out_df = pd.DataFrame(all_rows, columns=["model", "ID", "pred", "true", "RMSE", "MAE", "PCC"])
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    summary_rows = []
    for (dataset, spli), fold_metrics in metric_cache.items():
        for fold, rmse, mae, pcc in fold_metrics:
            summary_rows.append({
                "model": f"{dataset}_{spli}_fold{fold}",
                "RMSE": rmse,
                "MAE": mae,
                "PCC": pcc,
            })
        if spli in FIVE_FOLD_SPLITS and fold_metrics:
            summary_rows.append({
                "model": f"{dataset}_{spli}_all",
                "RMSE": format_mean_std([x[1] for x in fold_metrics]),
                "MAE": format_mean_std([x[2] for x in fold_metrics]),
                "PCC": format_mean_std([x[3] for x in fold_metrics]),
            })
    summary_path = os.path.join(OUTPUT_DIR, "MAGMA_metrics_only_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("✅ MAGMA 全模型推理完成（无训练）")
    print(f"✅ 总表: {OUTPUT_CSV}")
    print(f"✅ 指标摘要: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
