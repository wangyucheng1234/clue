#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, StudentT
from torch.utils.data import DataLoader, Dataset

from clue_utkface.data import (
    PackedUTKFaceAgeDataset,
    StandardScaler1D,
    UTKFaceAgeDataset,
    build_split_manifest,
    load_split_manifest,
    make_train_val_split,
    make_utkface_transforms,
)
from clue_utkface.models import BackboneBBBRegressor, BackboneGaussianRegressor, ResNetCLUE, ResNetDER


CALIBRATION_LEVELS: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)

_METHOD_ALIASES = {
    "bayes_by_backprop": "bbb",
    "bayes-by-backprop": "bbb",
    "mcdropout": "mc_dropout",
    "mc-dropout": "mc_dropout",
}


class DeterministicGaussianNoiseDataset(Dataset):
    def __init__(self, base_dataset: Dataset, std: float, seed: int) -> None:
        self.base_dataset = base_dataset
        self.std = float(std)
        self.seed = int(seed)
        if self.std < 0.0:
            raise ValueError("input noise std must be non-negative.")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        if self.std <= 0.0:
            return image, target

        generator = torch.Generator()
        generator.manual_seed(self.seed + int(idx))
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        image = torch.clamp(image + self.std * noise, 0.0, 1.0)
        return image, target


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def gaussian_nll(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_mean = y_mean.reshape(-1)
    y_var = np.maximum(y_var.reshape(-1), 1e-8)
    return float(np.mean(0.5 * np.log(2.0 * math.pi * y_var) + 0.5 * (y_true - y_mean) ** 2 / y_var))


def normalize_baseline_method(method: str) -> str:
    key = method.strip().lower()
    return _METHOD_ALIASES.get(key, key)


def gaussian_mixture_nll(
    y_true: np.ndarray,
    component_means: np.ndarray,
    component_vars: np.ndarray,
) -> float:
    y_true = y_true.reshape(-1)
    component_means = component_means.reshape(component_means.shape[0], -1)
    component_vars = np.maximum(component_vars.reshape(component_vars.shape[0], -1), 1e-8)

    log_probs = -0.5 * (
        np.log(2.0 * math.pi * component_vars)
        + (y_true[None, :] - component_means) ** 2 / component_vars
    )
    max_log_prob = np.max(log_probs, axis=0, keepdims=True)
    log_mean_prob = max_log_prob + np.log(np.mean(np.exp(log_probs - max_log_prob), axis=0, keepdims=True))
    return float(-np.mean(log_mean_prob))


def summarize_sampled_gaussian_predictions(
    mean_samples_norm: np.ndarray,
    var_samples_norm: np.ndarray,
    y_norm: np.ndarray,
    scaler: StandardScaler1D,
) -> Dict[str, object]:
    mean_samples_norm = mean_samples_norm.reshape(mean_samples_norm.shape[0], -1)
    var_samples_norm = np.maximum(var_samples_norm.reshape(var_samples_norm.shape[0], -1), 1e-8)
    y_norm = y_norm.reshape(-1)

    mean_norm = mean_samples_norm.mean(axis=0)
    epi_var_norm = mean_samples_norm.var(axis=0)
    ale_var_norm = var_samples_norm.mean(axis=0)
    pred_var_norm = epi_var_norm + ale_var_norm

    y_pred = scaler.inverse_transform(mean_norm.reshape(-1, 1))
    y_true = scaler.inverse_transform(y_norm.reshape(-1, 1))

    target_std = float(scaler.std_)
    epi_var = epi_var_norm.reshape(-1, 1) * (target_std ** 2)
    ale_var = ale_var_norm.reshape(-1, 1) * (target_std ** 2)
    pred_var = pred_var_norm.reshape(-1, 1) * (target_std ** 2)

    component_means = scaler.inverse_transform(mean_samples_norm.reshape(-1, 1)).reshape(mean_samples_norm.shape[0], -1)
    component_vars = var_samples_norm * (target_std ** 2)
    nll = gaussian_mixture_nll(y_true.reshape(-1), component_means, component_vars)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "epistemic_var": epi_var,
        "aleatoric_var": ale_var,
        "predictive_var": pred_var,
        "nll": nll,
    }


def gaussian_crps(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    y_true_t = torch.as_tensor(y_true, dtype=torch.float32).view(-1)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32).view(-1)
    sigma_t = torch.sqrt(torch.clamp(torch.as_tensor(y_var, dtype=torch.float32).view(-1), min=1e-8))
    z = (y_true_t - y_mean_t) / sigma_t
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma_t * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return float(crps.mean().item())


def normal_interval_metrics(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_var: np.ndarray,
    coverages: Tuple[float, ...] = CALIBRATION_LEVELS,
) -> Dict[str, object]:
    y_true_t = torch.as_tensor(y_true, dtype=torch.float32).view(-1)
    loc_t = torch.as_tensor(y_mean, dtype=torch.float32).view(-1)
    scale_t = torch.sqrt(torch.clamp(torch.as_tensor(y_var, dtype=torch.float32).view(-1), min=1e-8))
    dist = Normal(loc=loc_t, scale=scale_t)

    result: Dict[str, object] = {
        "uncertainty_metric_method": "exact_gaussian",
        "mc_metric_samples": 0,
        "calibration_levels": [float(coverage) for coverage in coverages],
    }
    gaps: List[float] = []
    for coverage in coverages:
        alpha = 0.5 * (1.0 - coverage)
        lower = dist.icdf(torch.full_like(loc_t, alpha))
        upper = dist.icdf(torch.full_like(loc_t, 1.0 - alpha))
        empirical = float(((y_true_t >= lower) & (y_true_t <= upper)).float().mean().item())
        width = float((upper - lower).mean().item())
        pct = int(round(coverage * 100))
        result[f"coverage_{pct}"] = empirical
        result[f"interval_width_{pct}"] = width
        gaps.append(abs(empirical - coverage))

    interval_ece = float(np.mean(gaps))
    result["ece"] = interval_ece
    result["interval_ece"] = interval_ece
    return result


def build_predictive_distribution(
    pred: Dict[str, object],
) -> Tuple[Normal | StudentT, str]:
    loc_t = torch.as_tensor(pred["y_pred"], dtype=torch.float32).view(-1)
    if "student_df" in pred and "student_scale" in pred:
        df_t = torch.as_tensor(pred["student_df"], dtype=torch.float32).view(-1)
        scale_t = torch.clamp(torch.as_tensor(pred["student_scale"], dtype=torch.float32).view(-1), min=1e-8)
        return StudentT(df_t, loc=loc_t, scale=scale_t), "monte_carlo_student_t"

    scale_t = torch.sqrt(torch.clamp(torch.as_tensor(pred["predictive_var"], dtype=torch.float32).view(-1), min=1e-8))
    return Normal(loc=loc_t, scale=scale_t), "monte_carlo_gaussian"


def sample_interval_and_crps_metrics(
    pred: Dict[str, object],
    num_samples: int,
    coverages: Tuple[float, ...] = CALIBRATION_LEVELS,
) -> Dict[str, object]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive for sample-based uncertainty metrics.")

    y_true_t = torch.as_tensor(pred["y_true"], dtype=torch.float32).view(-1)
    dist, method = build_predictive_distribution(pred)
    samples_a = dist.sample((num_samples,))
    samples_b = dist.sample((num_samples,))
    if samples_a.ndim == 3 and samples_a.shape[-1] == 1:
        samples_a = samples_a.squeeze(-1)
        samples_b = samples_b.squeeze(-1)

    abs_err = (samples_a - y_true_t.unsqueeze(0)).abs().mean(dim=0)
    abs_pair = (samples_a - samples_b).abs().mean(dim=0)

    result: Dict[str, object] = {
        "crps": float((abs_err - 0.5 * abs_pair).mean().item()),
        "uncertainty_metric_method": method,
        "mc_metric_samples": int(num_samples),
        "calibration_levels": [float(coverage) for coverage in coverages],
    }
    gaps: List[float] = []
    for coverage in coverages:
        alpha = 0.5 * (1.0 - coverage)
        lower = torch.quantile(samples_a, alpha, dim=0)
        upper = torch.quantile(samples_a, 1.0 - alpha, dim=0)
        empirical = float(((y_true_t >= lower) & (y_true_t <= upper)).float().mean().item())
        width = float((upper - lower).mean().item())
        pct = int(round(coverage * 100))
        result[f"coverage_{pct}"] = empirical
        result[f"interval_width_{pct}"] = width
        gaps.append(abs(empirical - coverage))

    interval_ece = float(np.mean(gaps))
    result["ece"] = interval_ece
    result["interval_ece"] = interval_ece
    return result


def uncertainty_metrics(
    pred: Dict[str, object],
    mc_metric_samples: int,
    coverages: Tuple[float, ...] = CALIBRATION_LEVELS,
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for prefix, key in (
        ("epistemic", "epistemic_var"),
        ("aleatoric", "aleatoric_var"),
        ("predictive", "predictive_var"),
    ):
        var = np.maximum(np.asarray(pred[key]).reshape(-1), 1e-8)
        std = np.sqrt(var)
        result[f"{prefix}_var_mean"] = float(np.mean(var))
        result[f"{prefix}_var_median"] = float(np.median(var))
        result[f"{prefix}_std_mean"] = float(np.mean(std))
        result[f"{prefix}_std_median"] = float(np.median(std))

    if "student_df" in pred and "student_scale" in pred:
        result.update(sample_interval_and_crps_metrics(pred, num_samples=mc_metric_samples, coverages=coverages))
        return result

    result["crps"] = gaussian_crps(pred["y_true"], pred["y_pred"], pred["predictive_var"])
    result.update(normal_interval_metrics(pred["y_true"], pred["y_pred"], pred["predictive_var"], coverages=coverages))
    return result


@torch.no_grad()
def predict_fixed_noise(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
    sigma_obs: float,
    device: str,
) -> Dict[str, object]:
    model.eval()
    mu_all, epi_all, y_all = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        mu, var_f = model(xb)
        mu_all.append(mu.cpu().numpy())
        epi_all.append(var_f.cpu().numpy())
        y_all.append(yb.numpy())

    mu_norm = np.concatenate(mu_all, axis=0).reshape(-1, 1)
    epi_var_norm = np.concatenate(epi_all, axis=0).reshape(-1, 1)
    y_norm = np.concatenate(y_all, axis=0).reshape(-1, 1)

    y_pred = scaler.inverse_transform(mu_norm)
    y_true = scaler.inverse_transform(y_norm)

    target_std = float(scaler.std_)
    epi_var = epi_var_norm * (target_std ** 2)
    ale_var = np.full_like(epi_var, fill_value=(sigma_obs * target_std) ** 2)
    pred_var = epi_var + ale_var

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "epistemic_var": epi_var,
        "aleatoric_var": ale_var,
        "predictive_var": pred_var,
        "predictive_distribution": "gaussian",
    }


@torch.no_grad()
def predict_variable_noise(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: str,
) -> Dict[str, object]:
    model.eval()
    mu_all, epi_all, logn_all, vlogn_all, y_all = [], [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        mu_f, var_f, mu_logsigma, var_logsigma = model(xb)
        mu_all.append(mu_f.cpu().numpy())
        epi_all.append(var_f.cpu().numpy())
        logn_all.append(mu_logsigma.cpu().numpy())
        vlogn_all.append(var_logsigma.cpu().numpy())
        y_all.append(yb.numpy())

    mu_norm = np.concatenate(mu_all, axis=0).reshape(-1, 1)
    epi_var_norm = np.concatenate(epi_all, axis=0).reshape(-1, 1)
    mu_logsigma = np.concatenate(logn_all, axis=0).reshape(-1, 1)
    var_logsigma = np.concatenate(vlogn_all, axis=0).reshape(-1, 1)
    y_norm = np.concatenate(y_all, axis=0).reshape(-1, 1)

    y_pred = scaler.inverse_transform(mu_norm)
    y_true = scaler.inverse_transform(y_norm)

    target_std = float(scaler.std_)
    epi_var = epi_var_norm * (target_std ** 2)
    ale_var_norm = np.exp(2.0 * mu_logsigma + 2.0 * var_logsigma)
    ale_var = ale_var_norm * (target_std ** 2)
    pred_var = epi_var + ale_var

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "epistemic_var": epi_var,
        "aleatoric_var": ale_var,
        "predictive_var": pred_var,
        "mu_logsigma": mu_logsigma,
        "var_logsigma": var_logsigma,
        "predictive_distribution": "gaussian",
    }


@torch.no_grad()
def predict_der(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: str,
) -> Dict[str, object]:
    model.eval()
    mu_all, v_all, a_all, b_all, y_all = [], [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        mu, v, alpha, beta = model(xb)
        mu_all.append(mu.cpu().numpy())
        v_all.append(v.cpu().numpy())
        a_all.append(alpha.cpu().numpy())
        b_all.append(beta.cpu().numpy())
        y_all.append(yb.numpy())

    mu_norm = np.concatenate(mu_all, axis=0).reshape(-1, 1)
    v = np.concatenate(v_all, axis=0).reshape(-1, 1)
    alpha = np.concatenate(a_all, axis=0).reshape(-1, 1)
    beta = np.concatenate(b_all, axis=0).reshape(-1, 1)
    y_norm = np.concatenate(y_all, axis=0).reshape(-1, 1)

    y_pred = scaler.inverse_transform(mu_norm)
    y_true = scaler.inverse_transform(y_norm)

    target_std = float(scaler.std_)
    ale_var_norm = beta / np.maximum(alpha - 1.0, 1e-6)
    epi_var_norm = beta / np.maximum(v * (alpha - 1.0), 1e-6)
    pred_var_norm = beta * (1.0 + v) / np.maximum(v * (alpha - 1.0), 1e-6)

    ale_var = ale_var_norm * (target_std ** 2)
    epi_var = epi_var_norm * (target_std ** 2)
    pred_var = pred_var_norm * (target_std ** 2)

    df = np.maximum(2.0 * alpha, 1e-6)
    scale_sq = beta * (1.0 + v) / np.maximum(v * alpha, 1e-6)
    scale = np.sqrt(np.maximum(scale_sq, 1e-8))
    scale_orig = scale * target_std

    y_true_t = torch.as_tensor(y_true, dtype=torch.float32).view(-1, 1)
    df_t = torch.as_tensor(df, dtype=torch.float32)
    loc_t = torch.as_tensor(y_pred, dtype=torch.float32)
    scale_t = torch.as_tensor(scale_orig, dtype=torch.float32)
    dist = StudentT(df_t, loc=loc_t, scale=scale_t)
    nll = float(-dist.log_prob(y_true_t).mean().item())

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "epistemic_var": epi_var,
        "aleatoric_var": ale_var,
        "predictive_var": pred_var,
        "student_df": df,
        "student_scale": scale_orig,
        "nll": nll,
        "predictive_distribution": "student_t",
    }


@torch.no_grad()
def predict_sampled_gaussian(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: str,
    num_samples: int,
    use_mc_dropout: bool,
) -> Dict[str, object]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive for sampled Gaussian baselines.")

    model.eval()
    if use_mc_dropout and hasattr(model, "set_mc_dropout"):
        model.set_mc_dropout(True)

    y_true_all, y_pred_all = [], []
    epi_all, ale_all, pred_all = [], [], []
    total_nll = 0.0
    total_examples = 0

    try:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            mean_samples = []
            var_samples = []
            for _ in range(num_samples):
                if use_mc_dropout:
                    mean, var = model(xb)
                else:
                    mean, var = model(xb, sample=True)
                mean_samples.append(mean.cpu().numpy())
                var_samples.append(var.cpu().numpy())

            batch_pred = summarize_sampled_gaussian_predictions(
                mean_samples_norm=np.stack(mean_samples, axis=0),
                var_samples_norm=np.stack(var_samples, axis=0),
                y_norm=yb.numpy(),
                scaler=scaler,
            )

            batch_size = len(yb)
            y_true_all.append(batch_pred["y_true"])
            y_pred_all.append(batch_pred["y_pred"])
            epi_all.append(batch_pred["epistemic_var"])
            ale_all.append(batch_pred["aleatoric_var"])
            pred_all.append(batch_pred["predictive_var"])
            total_nll += float(batch_pred["nll"]) * batch_size
            total_examples += batch_size
    finally:
        if use_mc_dropout and hasattr(model, "set_mc_dropout"):
            model.set_mc_dropout(False)

    return {
        "y_true": np.concatenate(y_true_all, axis=0),
        "y_pred": np.concatenate(y_pred_all, axis=0),
        "epistemic_var": np.concatenate(epi_all, axis=0),
        "aleatoric_var": np.concatenate(ale_all, axis=0),
        "predictive_var": np.concatenate(pred_all, axis=0),
        "nll": float(total_nll / max(1, total_examples)),
        "predictive_distribution": "gaussian",
        "predictive_samples": int(num_samples),
    }


def save_json(path: str, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_predictions_csv(path: str, pred: Dict[str, object]) -> None:
    rows = len(pred["y_true"])
    fieldnames = [
        "y_true",
        "y_pred",
        "epistemic_var",
        "aleatoric_var",
        "predictive_var",
    ]
    if "mu_logsigma" in pred:
        fieldnames.extend(["mu_logsigma", "var_logsigma"])
    if "student_df" in pred:
        fieldnames.extend(["student_df", "student_scale"])

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(rows):
            row = {
                "y_true": float(pred["y_true"][idx, 0]),
                "y_pred": float(pred["y_pred"][idx, 0]),
                "epistemic_var": float(pred["epistemic_var"][idx, 0]),
                "aleatoric_var": float(pred["aleatoric_var"][idx, 0]),
                "predictive_var": float(pred["predictive_var"][idx, 0]),
            }
            if "mu_logsigma" in pred:
                row["mu_logsigma"] = float(pred["mu_logsigma"][idx, 0])
                row["var_logsigma"] = float(pred["var_logsigma"][idx, 0])
            if "student_df" in pred:
                row["student_df"] = float(pred["student_df"][idx, 0])
                row["student_scale"] = float(pred["student_scale"][idx, 0])
            writer.writerow(row)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved UTKFace checkpoint on a held-out split")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--packed-data", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--noise-summary-csv", type=str, default=None)
    parser.add_argument("--predictive-samples", type=int, default=None)
    parser.add_argument("--mc-metrics-samples", type=int, default=256)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--input-noise-std-grid", type=float, nargs="+", default=None)
    parser.add_argument("--input-noise-seed", type=int, default=0)
    return parser.parse_args()


def find_default_split_manifest(checkpoint_path: str) -> str | None:
    candidate = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "split_manifest.json")
    return candidate if os.path.exists(candidate) else None


def resolve_eval_split(
    data_root: str,
    config: Dict[str, object],
    split_name: str,
    split_manifest_path: str | None,
) -> Tuple[List[str], Dict[str, object]]:
    if split_manifest_path:
        manifest = load_split_manifest(split_manifest_path, root=data_root)
        split_key = f"{split_name}_files"
        files = list(manifest.get(split_key, []))
        if not files:
            raise ValueError(f"Split '{split_name}' is empty in manifest: {split_manifest_path}")
        return files, manifest

    if split_name == "test":
        raise ValueError("A split manifest is required to evaluate the held-out test split.")

    split_seed = int(config.get("split_seed", config.get("seed", 0)))
    val_ratio = float(config.get("val_ratio", 0.1))
    train_files, val_files = make_train_val_split(
        data_root,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    manifest = build_split_manifest(
        data_root,
        train_files,
        val_files,
        [],
        split_seed=split_seed,
        val_ratio=val_ratio,
        test_ratio=0.0,
        source="reconstructed_for_eval",
    )
    files = train_files if split_name == "train" else val_files
    return files, manifest


def build_scaler(mean: float, std: float) -> StandardScaler1D:
    scaler = StandardScaler1D()
    scaler.mean_ = float(mean)
    scaler.std_ = float(std)
    return scaler


def build_loader(
    data_root: str,
    packed_data: str | None,
    files: List[str],
    scaler: StandardScaler1D,
    img_size: int,
    batch_size: int,
    num_workers: int,
    input_noise_std: float = 0.0,
    input_noise_seed: int = 0,
) -> Tuple[Dataset, DataLoader]:
    _, eval_tf = make_utkface_transforms(img_size=img_size)
    dataset_cls = PackedUTKFaceAgeDataset if packed_data else UTKFaceAgeDataset
    dataset_source = packed_data if packed_data else data_root
    dataset = dataset_cls(dataset_source, transform=eval_tf, file_list=files, target_scaler=scaler)
    if input_noise_std > 0.0:
        dataset = DeterministicGaussianNoiseDataset(
            dataset,
            std=input_noise_std,
            seed=input_noise_seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def load_model(
    checkpoint: Dict[str, object],
    device: str,
) -> Tuple[nn.Module, Dict[str, object]]:
    config = dict(checkpoint["config"])

    if "variant" in config:
        model = ResNetCLUE(
            variant=str(config["variant"]),
            architecture=str(config["architecture"]),
            pretrained=bool(config.get("pretrained", True)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            proj_dim=int(config.get("proj_dim", 256)),
            dropout=float(config.get("dropout", 0.0)),
            l2_normalize_features=True,
        )
    elif "method" in config:
        method = normalize_baseline_method(str(config["method"]))
        config["method"] = method
        if method in {"der", "der_elbo"}:
            model = ResNetDER(
                architecture=str(config["architecture"]),
                pretrained=bool(config.get("pretrained", True)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                dropout=float(config.get("dropout", 0.1)),
            )
        elif method == "mc_dropout":
            model = BackboneGaussianRegressor(
                architecture=str(config["architecture"]),
                pretrained=bool(config.get("pretrained", True)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                dropout=float(config.get("dropout", 0.1)),
            )
        elif method == "bbb":
            model = BackboneBBBRegressor(
                architecture=str(config["architecture"]),
                pretrained=bool(config.get("pretrained", True)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                prior_std=float(config.get("bbb_prior_std", 1.0)),
                init_rho=float(config.get("bbb_init_rho", -5.0)),
            )
        else:
            raise ValueError(f"Unsupported baseline method in checkpoint config: {method}")
    else:
        raise ValueError("Checkpoint config does not look like a CLUE or supported baseline run.")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def resolve_input_noise_stds(args: argparse.Namespace) -> List[float]:
    noise_stds = list(args.input_noise_std_grid) if args.input_noise_std_grid else [float(args.input_noise_std)]
    if not noise_stds:
        return [0.0]
    for value in noise_stds:
        if value < 0.0:
            raise ValueError("Input noise std values must be non-negative.")
    return [float(value) for value in noise_stds]


def evaluate_predictions(
    model: nn.Module,
    config: Dict[str, object],
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: str,
    predictive_samples: int | None,
) -> Tuple[Dict[str, object], float, str]:
    if "variant" in config:
        if config["variant"] == "fixed_noise":
            pred = predict_fixed_noise(
                model,
                loader,
                scaler,
                sigma_obs=float(config.get("sigma_obs", 0.15)),
                device=device,
            )
        else:
            pred = predict_variable_noise(model, loader, scaler, device)
        nll = gaussian_nll(pred["y_true"], pred["y_pred"], pred["predictive_var"])
        return pred, nll, f"clue_{config['variant']}"

    method = normalize_baseline_method(str(config["method"]))
    if method in {"der", "der_elbo"}:
        pred = predict_der(model, loader, scaler, device)
    else:
        num_samples = int(predictive_samples or config.get("eval_samples", 30))
        pred = predict_sampled_gaussian(
            model=model,
            loader=loader,
            scaler=scaler,
            device=device,
            num_samples=num_samples,
            use_mc_dropout=method == "mc_dropout",
        )
    return pred, float(pred["nll"]), method


def save_results_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Expected at least one row to write.")

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def summarize_training_time(checkpoint_path: str, checkpoint: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict):
        checkpoint_epoch = checkpoint.get("epoch", metrics.get("epoch"))
        if checkpoint_epoch is not None:
            summary["checkpoint_epoch"] = checkpoint_epoch
        if metrics.get("stage") is not None:
            summary["checkpoint_stage"] = metrics["stage"]
        if metrics.get("stage_epoch") is not None:
            summary["checkpoint_stage_epoch"] = metrics["stage_epoch"]
        train_time = finite_float(metrics.get("train_time_sec"))
        if train_time is not None:
            summary["checkpoint_train_time_sec"] = train_time

    history_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "history.json")
    if not os.path.exists(history_path):
        return summary

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError):
        return summary

    if not isinstance(history, list):
        return summary

    records = [record for record in history if isinstance(record, dict)]
    train_times = [
        value
        for value in (finite_float(record.get("train_time_sec")) for record in records)
        if value is not None
    ]
    if train_times:
        summary["history_path"] = os.path.abspath(history_path)
        summary["train_epochs_recorded"] = len(train_times)
        summary["train_time_sec_total"] = float(sum(train_times))
        summary["train_time_sec_per_epoch"] = float(sum(train_times) / len(train_times))
        summary["train_time_sec_per_epoch_std"] = (
            float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0
        )

    checkpoint_epoch = summary.get("checkpoint_epoch")
    if checkpoint_epoch is not None and "checkpoint_train_time_sec" not in summary:
        for record in records:
            if record.get("epoch") == checkpoint_epoch:
                train_time = finite_float(record.get("train_time_sec"))
                if train_time is not None:
                    summary["checkpoint_train_time_sec"] = train_time
                break

    return summary


def add_noise_baseline_deltas(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not results:
        return results

    baseline = next((row for row in results if abs(float(row["input_noise_std"])) < 1e-12), results[0])
    baseline_epistemic_std_mean = float(baseline["epistemic_std_mean"])
    baseline_epistemic_var_mean = float(baseline["epistemic_var_mean"])
    baseline_rmse = float(baseline["rmse"])

    augmented: List[Dict[str, object]] = []
    for row in results:
        item = dict(row)
        item["baseline_input_noise_std"] = float(baseline["input_noise_std"])
        item["epistemic_std_mean_delta_from_baseline"] = float(item["epistemic_std_mean"]) - baseline_epistemic_std_mean
        item["epistemic_var_mean_delta_from_baseline"] = float(item["epistemic_var_mean"]) - baseline_epistemic_var_mean
        item["rmse_delta_from_baseline"] = float(item["rmse"]) - baseline_rmse
        if baseline_epistemic_std_mean > 0.0:
            item["epistemic_std_mean_ratio_to_baseline"] = float(item["epistemic_std_mean"]) / baseline_epistemic_std_mean
        else:
            item["epistemic_std_mean_ratio_to_baseline"] = None
        augmented.append(item)
    return augmented


def main() -> None:
    args = parse_args()
    noise_stds = resolve_input_noise_stds(args)
    if len(noise_stds) > 1 and args.predictions_csv:
        raise ValueError("Predictions CSV can only be written when evaluating a single input-noise setting.")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model, config = load_model(checkpoint, args.device)
    training_time_summary = summarize_training_time(args.checkpoint, checkpoint)

    data_root = args.data_root or config.get("data_root")
    if not data_root:
        raise ValueError("Data root must be provided via --data-root or saved in the checkpoint config.")
    packed_data = args.packed_data or config.get("packed_data")

    split_manifest_path = args.split_manifest or find_default_split_manifest(args.checkpoint) or config.get("split_manifest")
    files, manifest = resolve_eval_split(
        data_root=str(data_root),
        config=config,
        split_name=args.split,
        split_manifest_path=str(split_manifest_path) if split_manifest_path else None,
    )

    scaler = build_scaler(checkpoint["scaler_mean"], checkpoint["scaler_std"])
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 4))
    img_size = int(config.get("img_size", 224))
    sweep_start = time.perf_counter()
    raw_results: List[Dict[str, object]] = []
    last_pred: Dict[str, object] | None = None

    for noise_std in noise_stds:
        run_start = time.perf_counter()
        dataset, loader = build_loader(
            data_root=str(data_root),
            packed_data=str(packed_data) if packed_data else None,
            files=files,
            scaler=scaler,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            input_noise_std=noise_std,
            input_noise_seed=args.input_noise_seed,
        )
        setup_time_sec = time.perf_counter() - run_start

        eval_start = time.perf_counter()
        prediction_start = time.perf_counter()
        pred, nll, run_type = evaluate_predictions(
            model=model,
            config=config,
            loader=loader,
            scaler=scaler,
            device=args.device,
            predictive_samples=args.predictive_samples,
        )
        prediction_time_sec = time.perf_counter() - prediction_start

        metric_start = time.perf_counter()
        metrics = regression_metrics(pred["y_true"], pred["y_pred"])
        extra_metrics = uncertainty_metrics(pred, mc_metric_samples=args.mc_metrics_samples)
        metric_time_sec = time.perf_counter() - metric_start
        evaluation_time_sec = time.perf_counter() - eval_start
        end_to_end_time_sec = time.perf_counter() - run_start

        result = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "data_root": os.path.abspath(str(data_root)),
            "packed_data": os.path.abspath(str(packed_data)) if packed_data else None,
            "split_manifest": os.path.abspath(str(split_manifest_path)) if split_manifest_path else None,
            "split": args.split,
            "run_type": run_type,
            "architecture": config.get("architecture"),
            "seed": config.get("seed"),
            "split_seed": manifest.get("split_seed"),
            "n_samples": len(dataset),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "nll": nll,
            "predictive_distribution": pred.get("predictive_distribution", "gaussian"),
            "predictive_samples": pred.get("predictive_samples"),
            "input_noise_type": "none" if noise_std <= 0.0 else "gaussian",
            "input_noise_std": float(noise_std),
            "input_noise_seed": int(args.input_noise_seed),
            **training_time_summary,
            "setup_time_sec": setup_time_sec,
            "prediction_time_sec": prediction_time_sec,
            "metric_time_sec": metric_time_sec,
            "evaluation_time_sec": evaluation_time_sec,
            "end_to_end_time_sec": end_to_end_time_sec,
            "prediction_samples_per_sec": float(len(dataset) / max(prediction_time_sec, 1e-8)),
            "evaluation_samples_per_sec": float(len(dataset) / max(evaluation_time_sec, 1e-8)),
            "end_to_end_samples_per_sec": float(len(dataset) / max(end_to_end_time_sec, 1e-8)),
        }
        result.update(extra_metrics)
        raw_results.append(result)
        last_pred = pred

    results = add_noise_baseline_deltas(raw_results)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if len(results) == 1:
            save_json(args.output, results[0])
        else:
            save_json(
                args.output,
                {
                    "checkpoint": os.path.abspath(args.checkpoint),
                    "split": args.split,
                    "input_noise_seed": int(args.input_noise_seed),
                    "sweep_total_time_sec": time.perf_counter() - sweep_start,
                    "results": results,
                },
            )
    if args.noise_summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.noise_summary_csv)), exist_ok=True)
        save_results_csv(args.noise_summary_csv, results)
    if args.predictions_csv and last_pred is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.predictions_csv)), exist_ok=True)
        save_predictions_csv(args.predictions_csv, last_pred)

    for result in results:
        print(
            f"[{result['run_type']}] split={result['split']} noise_std={result['input_noise_std']:.4f} "
            f"n={result['n_samples']} rmse={result['rmse']:.4f} mae={result['mae']:.4f} "
            f"nll={result['nll']:.4f} epi_std_mean={result['epistemic_std_mean']:.4f} "
            f"crps={result['crps']:.4f} ece={result['ece']:.4f} "
            f"eval_time={result['evaluation_time_sec']:.2f}s"
        )
    if args.output:
        print(f"metrics written to: {args.output}")
    if args.noise_summary_csv:
        print(f"noise summary written to: {args.noise_summary_csv}")
    if args.predictions_csv:
        print(f"predictions written to: {args.predictions_csv}")


if __name__ == "__main__":
    main()
