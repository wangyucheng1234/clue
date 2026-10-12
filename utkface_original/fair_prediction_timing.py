#!/usr/bin/env python3
"""Fair prediction-time benchmark for UTKFace checkpoints.

The existing ``prediction_time_sec`` in ``eval_checkpoint.py`` measures the whole
prediction loop, including DataLoader iteration, image decoding/resizing,
SQLite/cache reads, host-to-device transfer, and model inference. That makes
single-run comparisons noisy.

This script separates setup/data caching from repeated predictive forward timing:

  1. Load each checkpoint and build the requested evaluation split.
  2. Cache all evaluation batches once, optionally on the GPU.
  3. Run warmup passes.
  4. Time repeated predictive passes with CUDA synchronization.
  5. Report median/mean/std timing and throughput.

Example:

  python fair_prediction_timing.py \
    --output-json outputs/fair_prediction_timing.json \
    --output-csv outputs/fair_prediction_timing.csv

By default, the six UTKFace experiment checkpoints under ``outputs/`` are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from eval_checkpoint import (
    build_loader,
    build_scaler,
    default_device,
    find_default_split_manifest,
    load_model,
    normalize_baseline_method,
    resolve_eval_split,
)


DEFAULT_CHECKPOINTS: Tuple[Tuple[str, str], ...] = (
    ("BBB", "outputs/utkface_bbb/best.pt"),
    ("DER", "outputs/utkface_der/best.pt"),
    ("DER-ELBO", "outputs/utkface_der_elbo/best.pt"),
    ("MC Dropout", "outputs/utkface_mc_dropout/best.pt"),
    ("CLUE fixed", "outputs/utkface_fixed_noise/best.pt"),
    ("CLUE variable", "outputs/utkface_variable_noise/best.pt"),
)


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_path(path_value: object | None, base_dir: Path) -> str | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def parse_checkpoint_spec(spec: str) -> Tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
    else:
        path = spec.strip()
        label = Path(path).parent.name or Path(path).stem
    if not label or not path:
        raise ValueError(f"Invalid checkpoint specification: {spec!r}")
    return label, path


def resolve_checkpoint_items(specs: List[str] | None, repo_dir: Path) -> List[Tuple[str, str]]:
    items = [parse_checkpoint_spec(spec) for spec in specs] if specs else list(DEFAULT_CHECKPOINTS)
    resolved: List[Tuple[str, str]] = []
    for label, raw_path in items:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_dir / path
        if path.is_dir():
            path = path / "best.pt"
        resolved.append((label, str(path.resolve())))
    return resolved


def run_type_from_config(config: Dict[str, object]) -> str:
    if "variant" in config:
        return f"clue_{config['variant']}"
    return normalize_baseline_method(str(config["method"]))


def predictive_sample_count(config: Dict[str, object], override: int | None) -> int:
    if "variant" in config:
        return 1
    method = normalize_baseline_method(str(config["method"]))
    if method in {"der", "der_elbo"}:
        return 1
    return int(override or config.get("eval_samples", 30))


@torch.no_grad()
def predictive_forward(
    model: torch.nn.Module,
    xb: torch.Tensor,
    config: Dict[str, object],
    num_samples: int,
) -> None:
    if "variant" in config:
        _ = model(xb)
        return

    method = normalize_baseline_method(str(config["method"]))
    if method in {"der", "der_elbo"}:
        _ = model(xb)
        return

    if method == "mc_dropout":
        for _sample_idx in range(num_samples):
            _ = model(xb)
        return

    if method == "bbb":
        for _sample_idx in range(num_samples):
            _ = model(xb, sample=True)
        return

    raise ValueError(f"Unsupported method: {method}")


def set_sampling_mode(model: torch.nn.Module, config: Dict[str, object], enabled: bool) -> None:
    if "method" not in config:
        return
    method = normalize_baseline_method(str(config["method"]))
    if method == "mc_dropout" and hasattr(model, "set_mc_dropout"):
        model.set_mc_dropout(enabled)


def cache_eval_batches(
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    cache_device: str,
) -> Tuple[List[torch.Tensor], float, int]:
    if cache_device not in {"cpu", "device"}:
        raise ValueError("--cache-device must be 'cpu' or 'device'.")

    cached: List[torch.Tensor] = []
    n_samples = 0

    start = time.perf_counter()
    for xb, _yb in loader:
        n_samples += int(xb.shape[0])
        if cache_device == "device":
            xb = xb.to(device, non_blocking=True)
        cached.append(xb)
    sync_device(device)
    cache_time = time.perf_counter() - start

    return cached, cache_time, n_samples


@torch.no_grad()
def timed_prediction_pass(
    model: torch.nn.Module,
    cached_batches: List[torch.Tensor],
    config: Dict[str, object],
    device: str,
    cache_device: str,
    num_samples: int,
) -> float:
    sync_device(device)
    start = time.perf_counter()
    for xb in cached_batches:
        if cache_device == "cpu":
            xb = xb.to(device, non_blocking=True)
        predictive_forward(model, xb, config, num_samples=num_samples)
    sync_device(device)
    return time.perf_counter() - start


def benchmark_checkpoint(
    label: str,
    checkpoint_path: str,
    args: argparse.Namespace,
    repo_dir: Path,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model, config = load_model(checkpoint, args.device)
    model.eval()

    data_root = resolve_path(args.data_root, repo_dir) or resolve_path(config.get("data_root"), repo_dir)
    if not data_root:
        raise ValueError("Data root must be provided or present in the checkpoint config.")
    packed_data = resolve_path(args.packed_data, repo_dir) or resolve_path(config.get("packed_data"), repo_dir)

    split_manifest_path = (
        resolve_path(args.split_manifest, repo_dir)
        or find_default_split_manifest(checkpoint_path)
        or resolve_path(config.get("split_manifest"), repo_dir)
    )
    files, manifest = resolve_eval_split(
        data_root=data_root,
        config=config,
        split_name=args.split,
        split_manifest_path=str(split_manifest_path) if split_manifest_path else None,
    )

    scaler = build_scaler(checkpoint["scaler_mean"], checkpoint["scaler_std"])
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 4))
    img_size = int(config.get("img_size", 224))

    dataset, loader = build_loader(
        data_root=data_root,
        packed_data=packed_data,
        files=files,
        scaler=scaler,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        input_noise_std=0.0,
        input_noise_seed=0,
    )

    cached_batches, cache_time_sec, n_samples = cache_eval_batches(
        loader=loader,
        device=args.device,
        cache_device=args.cache_device,
    )

    num_samples = predictive_sample_count(config, args.predictive_samples)
    set_sampling_mode(model, config, enabled=True)
    try:
        for _ in range(args.warmup_repeats):
            timed_prediction_pass(
                model=model,
                cached_batches=cached_batches,
                config=config,
                device=args.device,
                cache_device=args.cache_device,
                num_samples=num_samples,
            )

        times = [
            timed_prediction_pass(
                model=model,
                cached_batches=cached_batches,
                config=config,
                device=args.device,
                cache_device=args.cache_device,
                num_samples=num_samples,
            )
            for _ in range(args.repeats)
        ]
    finally:
        set_sampling_mode(model, config, enabled=False)

    median_sec = float(statistics.median(times))
    mean_sec = float(statistics.mean(times))
    std_sec = float(statistics.stdev(times)) if len(times) > 1 else 0.0

    return {
        "label": label,
        "checkpoint": os.path.abspath(checkpoint_path),
        "run_type": run_type_from_config(config),
        "architecture": config.get("architecture"),
        "split": args.split,
        "split_seed": manifest.get("split_seed"),
        "n_samples": int(len(dataset)),
        "n_cached_samples": int(n_samples),
        "batch_size": int(batch_size),
        "num_batches": int(len(cached_batches)),
        "device": args.device,
        "cache_device": args.cache_device,
        "timed_region": "model_forward_only" if args.cache_device == "device" else "host_to_device_plus_model_forward",
        "predictive_samples": int(num_samples),
        "warmup_repeats": int(args.warmup_repeats),
        "repeats": int(args.repeats),
        "cache_time_sec": float(cache_time_sec),
        "fair_prediction_time_sec_median": median_sec,
        "fair_prediction_time_sec_mean": mean_sec,
        "fair_prediction_time_sec_std": std_sec,
        "fair_prediction_samples_per_sec_median": float(n_samples / max(median_sec, 1e-12)),
        "all_times_sec": [float(value) for value in times],
    }


def write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "label",
        "run_type",
        "architecture",
        "split",
        "n_samples",
        "batch_size",
        "num_batches",
        "device",
        "cache_device",
        "timed_region",
        "predictive_samples",
        "fair_prediction_time_sec_median",
        "fair_prediction_time_sec_mean",
        "fair_prediction_time_sec_std",
        "fair_prediction_samples_per_sec_median",
        "cache_time_sec",
        "checkpoint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fair prediction time for saved UTKFace checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Checkpoint or output directory to benchmark. Use label=path for a custom "
            "label. May be repeated. Defaults to the six paper checkpoints."
        ),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--packed-data", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument(
        "--cache-device",
        type=str,
        default="device",
        choices=["device", "cpu"],
        help=(
            "Where to cache evaluation batches before timing. 'device' measures model "
            "forward time only; 'cpu' measures host-to-device transfer plus forward."
        ),
    )
    parser.add_argument("--predictive-samples", type=int, default=None)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=str, default="outputs/fair_prediction_timing.json")
    parser.add_argument("--output-csv", type=str, default="outputs/fair_prediction_timing.csv")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup-repeats must be non-negative.")

    repo_dir = Path(__file__).resolve().parent
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    checkpoint_items = resolve_checkpoint_items(args.checkpoint, repo_dir)
    results: List[Dict[str, object]] = []
    for label, checkpoint_path in checkpoint_items:
        print(f"benchmarking {label}: {checkpoint_path}")
        result = benchmark_checkpoint(label, checkpoint_path, args, repo_dir)
        results.append(result)
        print(
            f"[{result['label']}] {result['timed_region']} "
            f"median={result['fair_prediction_time_sec_median']:.4f}s "
            f"mean={result['fair_prediction_time_sec_mean']:.4f}s "
            f"std={result['fair_prediction_time_sec_std']:.4f}s "
            f"samples/s={result['fair_prediction_samples_per_sec_median']:.1f}"
        )

    payload = {
        "description": (
            "Fair prediction timing. With cache_device='device', timing excludes "
            "DataLoader, image decoding, preprocessing, SQLite/file IO, and "
            "host-to-device transfer."
        ),
        "results": results,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, results)
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
