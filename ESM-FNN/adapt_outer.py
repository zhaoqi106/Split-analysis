"""Train ESM-FNN on one outer fold.

ESM-2 stays frozen. Only esm2line.SimpleFFN is trained, with full-batch MSE.
The checkpoint is the epoch with the best inner-validation PCC and is applied directly to the outer test set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ADAPT_DIR = Path("/data/ESM-FNN")
ESM_DIR = Path("/data/ESM-FNN")
sys.path.insert(0, str(ADAPT_DIR))
sys.path.insert(0, str(ESM_DIR))

from protocol import (  # noqa: E402
    assert_isolation,
    empty_prediction_frame,
    make_inner_split,
    regression_metrics,
    write_json,
    write_table,
    y_true,
)
from study_outputs import enrich_predictions, history_frame, write_fold_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESM-FNN 按外层五折协议训练")
    parser.add_argument("--dataset", required=True, choices=["S1131", "M1101"])
    parser.add_argument("--split-type", required=True, choices=["RCSV", "PCSV", "ICSV", "AGCV"])
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--esm-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="冒烟时每个子集最多使用的记录数，0 表示全量")
    return parser.parse_args()


def resolve_esm_weight(explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates = [
        Path("/data/ESM-FNN/esm2_t33_650M_UR50D.pt"),
        Path("/data/ESM-FNN/esm2_t33_650M_UR50D.pt"),
        Path("/data/ESM-FNN/esm2_t33_650M_UR50D.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


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


def identity_fields(args, full_counts):
    return dict(
        dataset=args.dataset,
        model="ESM-FNN",
        split_type=args.split_type,
        split_seed=args.split_seed,
        outer_fold=args.outer_fold,
        inner_split_seed=args.split_seed,
        model_seed=args.model_seed,
        **full_counts,
        selection_metric="inner_val_pcc",
    )


def main() -> None:
    args = parse_args()
    from protocol import load_outer_fold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_train, outer_val = load_outer_fold(args.dataset, args.split_seed, args.split_type, args.outer_fold)
    inner_train, inner_val, inner_info = make_inner_split(
        outer_train, args.dataset, args.split_type, args.split_seed
    )
    assert_isolation(outer_train, outer_val, inner_train, inner_val, args.split_type)
    full_counts = {
        "n_outer_train": len(outer_train),
        "n_inner_train": len(inner_train),
        "n_inner_val": len(inner_val),
        "n_outer_val": len(outer_val),
    }
    if args.limit and args.limit > 0:
        inner_train = inner_train.head(args.limit).reset_index(drop=True)
        inner_val = inner_val.head(max(2, args.limit)).reset_index(drop=True)
        outer_train = outer_train.head(args.limit).reset_index(drop=True)
        outer_val = outer_val.head(args.limit).reset_index(drop=True)
    weight = resolve_esm_weight(args.esm_model)
    fields = identity_fields(args, full_counts)
    manifest = {
        **fields,
        "smoke_limit": args.limit,
        "inner_info": inner_info,
        "esm_weight": str(weight) if weight else "",
        "precision_mode": "fp16",
        "train_mode": "best_inner_checkpoint",
        "eval_protocol": "best_inner_checkpoint",
        "frozen_modules": ["esm2_t33_650M"],
        "trained_modules": ["SimpleFFN"],
        "protocol_difference": "官方 esm2line 在旧划分上直接训练 FFN。这里在内部验证集上选轮，并用该轮权重直接预测外层测试集，不在完整外层训练集上重训。",
        "epochs_cap": args.epochs,
        "lr": args.lr,
    }
    if args.check_only or weight is None:
        reason = "" if weight else "缺少 ESM-2 权重 esm2_t33_650M_UR50D.pt，未提取特征，也未更新参数"
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

    import esm
    import torch
    from esm2line import CHAIN_CONFIG, MUT_CONFIG, get_feature, preprocess_df

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weight))
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    batch_converter = alphabet.get_batch_converter()
    use_fp16 = device.type == "cuda"
    manifest["precision_mode"] = "fp16" if use_fp16 else "fp32_cpu"
    cache: dict[str, np.ndarray] = {}

    def features(frame):
        pending = frame.loc[~frame["record_uid"].astype(str).isin(cache)].reset_index(drop=True)
        if len(pending):
            prepared = preprocess_df(pending, CHAIN_CONFIG[args.dataset], args.dataset, MUT_CONFIG[args.dataset])
            for uid, row in zip(pending["record_uid"].astype(str), prepared.itertuples(index=False)):
                cache[uid] = np.asarray(
                    get_feature(
                        encoder,
                        batch_converter,
                        device,
                        use_fp16,
                        row.wt_complex_seq,
                        row.mut_complex_seq,
                        row.mut_positions_in_complex,
                    ),
                    dtype=np.float32,
                )
        return np.stack([cache[uid] for uid in frame["record_uid"].astype(str)], axis=0)

    best_epoch, best_pcc, state, select_history = select_epoch(
        features(inner_train),
        y_true(inner_train),
        features(inner_val),
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

    predictions = predict_from_state(state, features(outer_val), device, use_fp16)
    payload = {
        "epoch": best_epoch,
        "state_dict": state,
        "selection_metric": "inner_val_pcc",
        "eval_protocol": "best_inner_checkpoint",
        "train_mode": manifest["train_mode"],
        "precision_mode": manifest["precision_mode"],
        "frozen_modules": manifest["frozen_modules"],
        "trained_modules": manifest["trained_modules"],
    }
    torch.save(payload, output_dir / "checkpoint_final.pt")
    torch.save(payload, output_dir / "refit.pt")
    frame = enrich_predictions(
        empty_prediction_frame(
            outer_val,
            **fields,
            selected_epoch=best_epoch,
            best_inner_val_metric=best_pcc,
            prediction_status="smoke" if args.limit else "ok",
            failure_reason="冒烟只训练了每个子集的前若干条" if args.limit else "",
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
    }
    manifest["selected_epoch"] = best_epoch
    manifest["best_inner_val_metric"] = best_pcc
    write_fold_bundle(
        output_dir,
        predictions=frame,
        history=history_frame(select_history),
        inner_train=inner_train,
        inner_val=inner_val,
        outer_val=outer_val,
        selection=selection,
        manifest=manifest,
    )
    print(f"ok epoch={best_epoch} inner_pcc={best_pcc:.4f} n_pred={len(predictions)} precision={manifest['precision_mode']}")


if __name__ == "__main__":
    main()
