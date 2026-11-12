"""Plan and launch the training-size control runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path("/data")
DEFAULT_SPLIT_ROOT = Path("/data/data_split/splits")
DEFAULT_ESM_MODEL = Path("/data/ESM-FNN/esm2_t33_650M_UR50D.pt")
SEEDS = [0, 42, 1024, 1234, 2026]
DATASETS = ["S1131", "M1101"]


def setup_protocol(project_root, split_root):
    adapt_dir = project_root / "adapt"
    esm_dir = project_root / "ESM-FNN"
    for p in (adapt_dir, esm_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import protocol

    protocol.set_roots(project_root, split_root)
    return protocol


def generate_plan(project_root, split_root, output_root):
    protocol = setup_protocol(project_root, split_root)
    tasks = []
    plan_rows = []
    for di, dataset in enumerate(DATASETS):
        for seed in SEEDS:
            # ICSV sizes
            icsv_sizes = []
            for fold in range(5):
                ot, oe = protocol.load_outer_fold(dataset, seed, "ICSV", fold)
                it, iv, info = protocol.make_inner_split(ot, dataset, "ICSV", seed)
                icsv_sizes.append(len(it))
            # RCSV available
            rcsv_sizes = []
            rcsv_inner_trains = []
            for fold in range(5):
                ot, oe = protocol.load_outer_fold(dataset, seed, "RCSV", fold)
                it, iv, info = protocol.make_inner_split(ot, dataset, "RCSV", seed)
                rcsv_sizes.append(len(it))
                rcsv_inner_trains.append(it)
            rcsv_min = min(rcsv_sizes)
            # candidates from ICSV sizes strictly below full scale and <= all RCSV available
            candidates = sorted(set(s for s in icsv_sizes if s < rcsv_min))
            if len(candidates) > 2:
                reduced_targets = [min(candidates), max(candidates)]
            else:
                reduced_targets = candidates
            unmatched = sorted(set(s for s in icsv_sizes if s >= rcsv_min))
            # reduced tasks
            for target in reduced_targets:
                for fold in range(5):
                    sampling_seed = 1000000 + di * 100000 + seed * 10 + fold
                    ot, oe = protocol.load_outer_fold(dataset, seed, "RCSV", fold)
                    it, iv, info = protocol.make_inner_split(ot, dataset, "RCSV", seed)
                    task_key = f"ESM-FNN__{dataset}__RCSV_size__seed{seed}__fold{fold}__nfit{target}"
                    tasks.append(
                        {
                            "task_key": task_key,
                            "dataset": dataset,
                            "split_seed": seed,
                            "outer_fold": fold,
                            "n_fit": target,
                            "sampling_seed": sampling_seed,
                            "target_type": "reduced",
                        }
                    )
            # full scale reference: reuse the official complete-scale result after consistency validation
            for fold in range(5):
                sampling_seed = 1000000 + di * 100000 + seed * 10 + fold
                task_key = f"ESM-FNN__{dataset}__RCSV_size__seed{seed}__fold{fold}__nfit{rcsv_sizes[fold]}"
                tasks.append(
                    {
                        "task_key": task_key,
                        "dataset": dataset,
                        "split_seed": seed,
                        "outer_fold": fold,
                        "n_fit": rcsv_sizes[fold],
                        "sampling_seed": sampling_seed,
                        "target_type": "full",
                    }
                )
            plan_rows.append(
                {
                    "dataset": dataset,
                    "split_seed": seed,
                    "icsv_n_inner_train_folds": ";".join(map(str, icsv_sizes)),
                    "rcsv_n_inner_train_available_folds": ";".join(map(str, rcsv_sizes)),
                    "rcsv_min_available": rcsv_min,
                    "reduced_targets": ";".join(map(str, reduced_targets)),
                    "unmatched_icsv_scales": ";".join(map(str, unmatched)),
                    "n_reduced_targets": len(reduced_targets),
                }
            )

    # dedup task keys (full vs reduced collision safety)
    seen = set()
    uniq = []
    for t in tasks:
        if t["task_key"] in seen:
            continue
        seen.add(t["task_key"])
        uniq.append(t)
    tasks = uniq

    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "size_control_plan.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tasks[0].keys()))
        writer.writeheader()
        writer.writerows(tasks)

    plan_summary_path = output_root / "size_control_plan_summary.csv"
    with plan_summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(plan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plan_rows)

    print(f"生成任务数: {len(tasks)}")
    print(f"计划表: {plan_path}")
    print(f"计划摘要: {plan_summary_path}")
    return tasks, plan_path


def is_task_done(output_dir: Path) -> bool:
    manifest = output_dir / "run_manifest.json"
    oof = output_dir / "oof_predictions.csv"
    if not manifest.is_file() or not oof.is_file():
        return False
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if "selected_epoch" not in m:
        return False
    # ensure oof has rows
    try:
        import pandas as pd

        df = pd.read_csv(oof)
        if len(df) == 0:
            return False
    except Exception:
        return False
    return True


def run_task(task, args):
    project_root = Path(args.project_root)
    split_root = Path(args.split_root)
    esm_model = Path(args.esm_model)
    output_root = Path(args.output_root)
    output_dir = output_root / "runs" / task["task_key"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if task.get("target_type") == "full":
        print(f"[REUSE FORMAL] {task['task_key']}")
        return True

    if not args.force and is_task_done(output_dir):
        print(f"[SKIP] {task['task_key']}")
        return True

    python = sys.executable
    entry = project_root / "ESM-FNN" / "run_size_control.py"
    cmd = [
        python,
        "-u",
        str(entry),
        "--dataset",
        task["dataset"],
        "--split-type",
        "RCSV",
        "--split-seed",
        str(task["split_seed"]),
        "--outer-fold",
        str(task["outer_fold"]),
        "--model-seed",
        "0",
        "--sampling-seed",
        str(task["sampling_seed"]),
        "--n-fit",
        str(task["n_fit"]),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--project-root",
        str(project_root),
        "--split-root",
        str(split_root),
        "--esm-model",
        str(esm_model),
        "--output-dir",
        str(output_dir),
        "--feature-cache-dir",
        str(output_root / "feature_cache"),
    ]
    log_path = output_dir / "console.log"
    for attempt in range(1, args.max_retries + 1):
        print(f"[RUN attempt {attempt}] {task['task_key']}")
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n===== attempt {attempt} =====\n")
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and is_task_done(output_dir):
            print(f"[OK] {task['task_key']}")
            return True
        print(f"[FAIL] {task['task_key']} rc={proc.returncode}")
        time.sleep(5)
    return False


def prepare_cache(args):
    project_root = Path(args.project_root)
    split_root = Path(args.split_root)
    esm_model = Path(args.esm_model)
    output_root = Path(args.output_root)
    protocol = setup_protocol(project_root, split_root)
    # import run_size_control functions
    esm_dir = project_root / "ESM-FNN"
    if str(esm_dir) not in sys.path:
        sys.path.insert(0, str(esm_dir))
    import run_size_control as rsc

    import esm
    import torch

    weight = Path(esm_model)
    assert weight.is_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weight))
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    batch_converter = alphabet.get_batch_converter()
    use_fp16 = device.type == "cuda"
    cache_dir = output_root / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weight_sha = rsc.sha256_file(weight)

    for dataset in DATASETS:
        # collect all records from master
        master = protocol.load_master(dataset)
        # build a frame with record_uid; need original columns. load_master already has them.
        frame = master.copy()
        cache_obj, cache_path = rsc.load_or_init_cache(cache_dir, dataset, weight_sha)
        rsc.extract_features(frame, dataset, encoder, batch_converter, device, use_fp16, cache_obj, cache_path, weight_sha)
        print(f"cache ready: {dataset} -> {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="ESM-FNN size control launcher")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--esm-model", default=str(DEFAULT_ESM_MODEL))
    parser.add_argument("--output-root", default="/data/esmfnn_size_control_20261104")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--generate-plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.prepare_cache:
        prepare_cache(args)
        return

    tasks, plan_path = generate_plan(project_root, split_root, output_root)

    if args.generate_plan and not args.run:
        return

    if not args.run:
        # default: generate plan only
        return

    # shard
    shard_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard_tasks)} tasks")

    failed = []
    for task in shard_tasks:
        ok = run_task(task, args)
        if not ok:
            failed.append(task["task_key"])

    if failed:
        print(f"FAILED TASKS ({len(failed)}):")
        for f in failed:
            print(f)
        raise SystemExit(1)
    print("ALL TASKS IN SHARD DONE")


if __name__ == "__main__":
    main()
