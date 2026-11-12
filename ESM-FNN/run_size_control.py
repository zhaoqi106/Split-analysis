"""One training-size control task. The outer RCSV eval set stays fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_VERSION = "esmfnn_size_control_v1_20261104"

DEFAULT_PROJECT_ROOT = Path("/data")
DEFAULT_SPLIT_ROOT = Path("/data/data_split/splits")
DEFAULT_ESM_MODEL = Path("/data/ESM-FNN/esm2_t33_650M_UR50D.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESM-FNN 训练规模对照（固定 RCSV 评价集）")
    parser.add_argument("--dataset", required=True, choices=["S1131", "M1101"])
    parser.add_argument("--split-type", default="RCSV", choices=["RCSV"])
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--n-fit", type=int, default=0, help="实际参与梯度更新的样本数；0 表示使用全部可用内部训练集")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--esm-model", default=str(DEFAULT_ESM_MODEL))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def setup_paths(args) -> tuple[Path, Path, Path, Path, Path]:
    project_root = Path(args.project_root).resolve()
    split_root = Path(args.split_root).resolve()
    esm_model = Path(args.esm_model).resolve()
    esm_dir = project_root / "ESM-FNN"
    adapt_dir = project_root / "adapt"
    for p in (adapt_dir, esm_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    # patch protocol constants
    import protocol

    protocol.set_roots(project_root, split_root)
    if args.feature_cache_dir:
        cache_dir = Path(args.feature_cache_dir).resolve()
    else:
        cache_dir = project_root / "cache" / "esm_fnn_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return project_root, split_root, esm_model, esm_dir, cache_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_esm_weight(explicit: str) -> Path | None:
    path = Path(explicit)
    return path if path.is_file() else None


def _amp(device, use_fp16):
    import torch

    enabled = bool(use_fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    return enabled, scaler


def _fresh(seed, device):
    import torch
    from esm2line import SimpleFFN

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return SimpleFFN().to(device)


def select_epoch(x_train, y_train, x_select, y_select, epochs, lr, seed, device, use_fp16):
    import torch

    model = _fresh(seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    enabled, scaler = _amp(device, use_fp16)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_select_t = torch.tensor(x_select, dtype=torch.float32, device=device)
    history = []
    best_pcc = -np.inf
    best_epoch = -1
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
            pred = model(x_train_t)
            loss = criterion(pred.float(), y_train_t)
        if not torch.isfinite(loss):
            history.append({"phase": "select", "epoch": epoch, "train_loss": float("nan"), "pcc": float("nan")})
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
                selected = model(x_select_t).float().detach().cpu().numpy()
        from protocol import regression_metrics

        metrics = regression_metrics(y_select, selected)
        history.append({"phase": "select", "epoch": epoch, "train_loss": float(loss.item()), **metrics})
        pcc = metrics["pcc"]
        if np.isfinite(pcc) and pcc > best_pcc:
            best_pcc = float(pcc)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return best_epoch, best_pcc, best_state, history


def predict_from_state(state, x_val, device, use_fp16):
    import torch
    from esm2line import SimpleFFN

    model = SimpleFFN().to(device)
    model.load_state_dict(state)
    model.eval()
    enabled = bool(use_fp16 and device.type == "cuda")
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
            return model(x_val_t).float().detach().cpu().numpy()


FEATURE_CONFIG_EXTRA = {
    "repr_layer": 33,
    "aggregation": "MUT_minus_WT_at_mut_positions_summed",
    "sequence_order": "chains concatenated in CHAIN_CONFIG order",
    "code_version": CODE_VERSION,
}


def feature_meta(dataset, weight_sha256):
    import json

    from esm2line import CHAIN_CONFIG, MUT_CONFIG

    return {
        "dataset": dataset,
        "chain_config": list(CHAIN_CONFIG[dataset]),
        "mut_config": dict(MUT_CONFIG[dataset]),
        "weight_sha256": weight_sha256,
        **FEATURE_CONFIG_EXTRA,
    }


def load_or_init_cache(cache_dir: Path, dataset: str, weight_sha256: str):
    cache_path = cache_dir / f"features_{dataset}.pt"
    if cache_path.is_file():
        try:
            obj = torch_load(cache_path)
        except Exception:
            obj = None
        if isinstance(obj, dict) and obj.get("meta") == feature_meta(dataset, weight_sha256):
            return obj, cache_path
    return {"meta": feature_meta(dataset, weight_sha256), "data": {}}, cache_path


def torch_load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def torch_save(obj, path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def extract_features(frame, dataset, encoder, batch_converter, device, use_fp16, cache_obj, cache_path, weight_sha256):
    import torch
    from esm2line import CHAIN_CONFIG, MUT_CONFIG, get_feature, preprocess_df

    meta = feature_meta(dataset, weight_sha256)
    if cache_obj.get("meta") != meta:
        cache_obj["meta"] = meta
        cache_obj["data"] = {}

    data = cache_obj["data"]
    pending_mask = [uid not in data for uid in frame["record_uid"].astype(str)]
    pending = frame.loc[pending_mask].reset_index(drop=True)
    if len(pending):
        prepared = preprocess_df(pending, CHAIN_CONFIG[dataset], dataset, MUT_CONFIG[dataset])
        rows_by_uid = dict(zip(pending["record_uid"].astype(str), prepared.itertuples(index=False)))
        for uid, row in rows_by_uid.items():
            vec = get_feature(
                encoder,
                batch_converter,
                device,
                use_fp16,
                row.wt_complex_seq,
                row.mut_complex_seq,
                row.mut_positions_in_complex,
            )
            data[uid] = {
                "x": np.asarray(vec, dtype=np.float32),
                "wt_complex_seq": str(row.wt_complex_seq),
                "mut_complex_seq": str(row.mut_complex_seq),
                "mut_positions_in_complex": str(row.mut_positions_in_complex),
            }
        torch_save(cache_obj, cache_path)

    uids = frame["record_uid"].astype(str).tolist()
    missing = [u for u in uids if u not in data]
    if missing:
        raise RuntimeError(f"特征缓存缺少记录: {missing[:5]}")
    return np.stack([data[u]["x"] for u in uids], axis=0).astype(np.float32)


def identity_fields(args, full_counts, n_fit, n_inner_train_available, sampling_seed):
    return dict(
        dataset=args.dataset,
        model="ESM-FNN",
        split_type=args.split_type,
        split_seed=args.split_seed,
        outer_fold=args.outer_fold,
        inner_split_seed=args.split_seed,
        model_seed=args.model_seed,
        sampling_seed=sampling_seed,
        n_outer_train=full_counts["n_outer_train"],
        n_inner_train_available=n_inner_train_available,
        n_fit=n_fit,
        n_inner_val=full_counts["n_inner_val"],
        n_outer_val=full_counts["n_outer_val"],
    )


def main() -> None:
    args = parse_args()
    project_root, split_root, esm_model_path, esm_dir, cache_dir = setup_paths(args)

    import protocol
    from protocol import (
        load_outer_fold,
        make_inner_split,
        y_true,
        write_table,
        write_json,
        empty_prediction_frame,
        regression_metrics,
    )
    from study_outputs import enrich_predictions, history_frame, write_fold_bundle

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outer_train, outer_val = load_outer_fold(args.dataset, args.split_seed, args.split_type, args.outer_fold)
    inner_train, inner_val, inner_info = make_inner_split(outer_train, args.dataset, args.split_type, args.split_seed)

    n_inner_train_available = len(inner_train)
    n_fit = args.n_fit if args.n_fit and args.n_fit > 0 else n_inner_train_available
    if n_fit > n_inner_train_available:
        raise SystemExit(f"n_fit={n_fit} 超过可用内部训练数 {n_inner_train_available}")
    if n_fit <= 0:
        raise SystemExit("n_fit 无效")

    rng = np.random.default_rng(args.sampling_seed)
    perm = rng.permutation(len(inner_train))
    selected_positions = perm[:n_fit]
    sampled_inner_train = inner_train.iloc[selected_positions].reset_index(drop=True)

    outer_train_ids = set(outer_train["record_uid"])
    outer_val_ids = set(outer_val["record_uid"])
    inner_val_ids = set(inner_val["record_uid"])
    sampled_ids = set(sampled_inner_train["record_uid"])
    if outer_train_ids & outer_val_ids:
        raise SystemExit("外层训练与外层评估相交")
    if sampled_ids & inner_val_ids:
        raise SystemExit("抽样训练与内部验证相交")
    if sampled_ids & outer_val_ids:
        raise SystemExit("抽样训练与外层评估相交")
    if not sampled_ids <= set(inner_train["record_uid"]):
        raise SystemExit("抽样记录不属于原始内部训练集")
    if len(sampled_inner_train) != n_fit:
        raise SystemExit("实际抽样数量不等于目标数量")

    full_counts = {
        "n_outer_train": len(outer_train),
        "n_inner_train": n_fit,
        "n_inner_val": len(inner_val),
        "n_outer_val": len(outer_val),
    }

    write_table(inner_train, output_dir / "inner_train_original.csv")
    write_table(sampled_inner_train, output_dir / "inner_train_sampled.csv")
    write_table(inner_val, output_dir / "inner_val.csv")
    write_table(outer_val, output_dir / "outer_eval.csv")
    sampling_manifest = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "outer_fold": args.outer_fold,
        "sampling_seed": args.sampling_seed,
        "n_inner_train_available": n_inner_train_available,
        "n_fit": n_fit,
        "permutation": perm.tolist(),
        "selected_positions": [int(x) for x in selected_positions],
        "selected_record_uids": sampled_inner_train["record_uid"].astype(str).tolist(),
    }
    write_json(sampling_manifest, output_dir / "sampling_manifest.json")

    weight = resolve_esm_weight(str(esm_model_path))
    fields = identity_fields(args, full_counts, n_fit, n_inner_train_available, args.sampling_seed)
    manifest = {
        **fields,
        "inner_info": inner_info,
        "esm_weight": str(weight) if weight else "",
        "precision_mode": "fp16",
        "train_mode": "best_inner_checkpoint",
        "eval_protocol": "best_inner_checkpoint",
        "frozen_modules": ["esm2_t33_650M"],
        "trained_modules": ["SimpleFFN"],
        "epochs_cap": args.epochs,
        "lr": args.lr,
        "code_version": CODE_VERSION,
        "n_inner_train_available": n_inner_train_available,
        "n_fit": n_fit,
    }

    if args.check_only or weight is None:
        reason = "" if weight else "缺少 ESM-2 权重，未提取特征，也未更新参数"
        frame = enrich_predictions(
            empty_prediction_frame(
                outer_val,
                **fields,
                selected_epoch="",
                best_inner_val_metric="",
                prediction_status="blocked" if weight is None else "checked",
                failure_reason=reason,
            ),
            outer_val,
            "fp16",
            manifest["train_mode"],
        )
        write_table(frame, output_dir / "oof_predictions.csv")
        write_json(manifest, output_dir / "run_manifest.json")
        if weight is None and not args.check_only:
            raise SystemExit(reason)
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    manifest["precision_mode"] = "fp16" if use_fp16 else "fp32_cpu"

    weight_sha = sha256_file(weight)
    cache_obj, cache_path = load_or_init_cache(cache_dir, args.dataset, weight_sha)

    needed_uids = [str(u) for u in list(sampled_inner_train["record_uid"]) + list(inner_val["record_uid"]) + list(outer_val["record_uid"])]
    pending = [u for u in needed_uids if u not in cache_obj.get("data", {})]
    encoder = None
    batch_converter = None
    if pending:
        import esm

        encoder, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weight))
        encoder.eval().to(device)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        batch_converter = alphabet.get_batch_converter()

    manifest["torch_seed"] = args.model_seed
    manifest["numpy_sampling_seed"] = args.sampling_seed
    manifest["split_seed"] = args.split_seed

    x_inner_train = extract_features(sampled_inner_train, args.dataset, encoder, batch_converter, device, use_fp16, cache_obj, cache_path, weight_sha)
    x_inner_val = extract_features(inner_val, args.dataset, encoder, batch_converter, device, use_fp16, cache_obj, cache_path, weight_sha)
    x_outer_val = extract_features(outer_val, args.dataset, encoder, batch_converter, device, use_fp16, cache_obj, cache_path, weight_sha)

    best_epoch, best_pcc, state, select_history = select_epoch(
        x_inner_train,
        y_true(sampled_inner_train),
        x_inner_val,
        y_true(inner_val),
        args.epochs,
        args.lr,
        args.model_seed,
        device,
        use_fp16,
    )
    if best_epoch < 0:
        frame = enrich_predictions(
            empty_prediction_frame(
                outer_val,
                **fields,
                selected_epoch=-1,
                best_inner_val_metric="",
                prediction_status="failed",
                failure_reason="内部验证 PCC 全程不可用，没有可选轮数",
            ),
            outer_val,
            manifest["precision_mode"],
            manifest["train_mode"],
        )
        write_table(frame, output_dir / "oof_predictions.csv")
        write_table(history_frame(select_history), output_dir / "training_history.csv")
        raise SystemExit("内部验证没有有效 PCC")

    predictions = predict_from_state(state, x_outer_val, device, use_fp16)
    payload = {
        "epoch": best_epoch,
        "state_dict": state,
        "selection_metric": "inner_val_pcc",
        "eval_protocol": "best_inner_checkpoint",
        "train_mode": manifest["train_mode"],
        "precision_mode": manifest["precision_mode"],
        "frozen_modules": manifest["frozen_modules"],
        "trained_modules": manifest["trained_modules"],
        "n_fit": n_fit,
        "sampling_seed": args.sampling_seed,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
    }
    torch.save(payload, output_dir / "checkpoint_final.pt")
    torch.save(payload, output_dir / "refit.pt")

    frame = enrich_predictions(
        empty_prediction_frame(
            outer_val,
            **fields,
            selected_epoch=best_epoch,
            best_inner_val_metric=best_pcc,
            prediction_status="ok",
            failure_reason="",
        ),
        outer_val,
        manifest["precision_mode"],
        manifest["train_mode"],
    )
    frame["y_pred"] = predictions
    selection = {
        "selected_epoch": best_epoch,
        "selection_metric": "inner_val_pcc",
        "best_inner_val_metric": best_pcc,
        "eval_protocol": "best_inner_checkpoint",
        "checkpoint": "内部验证 PCC 最高一轮的 SimpleFFN 权重，直接预测外层测试集",
        "precision_mode": manifest["precision_mode"],
        "n_fit": n_fit,
        "sampling_seed": args.sampling_seed,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
    }
    manifest["selected_epoch"] = best_epoch
    manifest["best_inner_val_metric"] = best_pcc
    write_fold_bundle(
        output_dir,
        predictions=frame,
        history=history_frame(select_history),
        inner_train=sampled_inner_train,
        inner_val=inner_val,
        outer_val=outer_val,
        selection=selection,
        manifest=manifest,
    )
    write_table(history_frame(select_history), output_dir / "training_history.csv")
    print(f"ok epoch={best_epoch} inner_pcc={best_pcc:.4f} n_fit={n_fit} n_pred={len(predictions)} precision={manifest['precision_mode']}")


if __name__ == "__main__":
    main()
