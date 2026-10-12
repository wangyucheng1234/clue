#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import StudentT
from torch.utils.data import DataLoader
from tqdm import tqdm

from clue_utkface.data import (
    PackedUTKFaceAgeDataset,
    StandardScaler1D,
    UTKFaceAgeDataset,
    build_split_manifest,
    load_split_manifest,
    make_train_val_split,
    make_utkface_transforms,
)
from clue_utkface.models import (
    AVAILABLE_ARCHITECTURES,
    BackboneBBBRegressor,
    BackboneGaussianRegressor,
    ResNetDER,
    evidential_regression_elbo_loss,
    evidential_regression_loss,
    gaussian_nll_loss,
    normalize_architecture,
)


AVAILABLE_BASELINE_METHODS = ("der", "der_elbo", "mc_dropout", "bbb")
VALID_SELECTION_METRICS = ("val_rmse", "val_mae", "val_nll")

_METHOD_ALIASES = {
    "bayes_by_backprop": "bbb",
    "bayes-by-backprop": "bbb",
    "mcdropout": "mc_dropout",
    "mc-dropout": "mc_dropout",
}


@dataclass
class BaselineConfig:
    data_root: str
    output_dir: str
    packed_data: str | None = None
    method: str = "der"
    architecture: str = "resnet18"
    seed: int = 0
    split_seed: int = 0
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_ratio: float = 0.1
    split_manifest: str | None = None
    selection_metric: str = "val_rmse"
    pretrained: bool = True
    hidden_dim: int = 256
    dropout: float = 0.1
    lamb: float = 1e-3
    eval_samples: int = 30
    train_samples: int = 1
    kl_weight: float = 1.0
    bbb_prior_std: float = 1.0
    bbb_init_rho: float = -5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_baseline_method(method: str) -> str:
    key = method.strip().lower()
    return _METHOD_ALIASES.get(key, key)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def select_metric(record: Dict[str, float], metric_name: str) -> float:
    if metric_name not in VALID_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {VALID_SELECTION_METRICS}")
    if metric_name not in record:
        raise KeyError(f"Metric '{metric_name}' was not found in the training record.")
    return float(record[metric_name])


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


@torch.no_grad()
def predict_der(model: nn.Module, loader: DataLoader, scaler: StandardScaler1D, device: str):
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


def resolve_train_val_split(cfg: BaselineConfig) -> Tuple[List[str], List[str], Dict[str, object]]:
    if cfg.split_manifest:
        split_manifest = load_split_manifest(cfg.split_manifest, root=cfg.data_root)
        train_files = list(split_manifest["train_files"])
        val_files = list(split_manifest["val_files"])
    else:
        train_files, val_files = make_train_val_split(
            cfg.data_root,
            val_ratio=cfg.val_ratio,
            seed=cfg.split_seed,
        )
        split_manifest = build_split_manifest(
            cfg.data_root,
            train_files,
            val_files,
            [],
            split_seed=cfg.split_seed,
            val_ratio=cfg.val_ratio,
            test_ratio=0.0,
            source="generated_in_train_baseline",
        )

    if not train_files or not val_files:
        raise ValueError("Training requires non-empty train and validation splits.")

    return train_files, val_files, split_manifest


def build_loaders(cfg: BaselineConfig):
    train_files, val_files, split_manifest = resolve_train_val_split(cfg)
    train_tf, eval_tf = make_utkface_transforms(img_size=cfg.img_size)
    dataset_cls = PackedUTKFaceAgeDataset if cfg.packed_data else UTKFaceAgeDataset
    dataset_source = cfg.packed_data if cfg.packed_data else cfg.data_root

    raw_train_ds = dataset_cls(dataset_source, transform=train_tf, file_list=train_files, target_scaler=None)
    scaler = StandardScaler1D()
    scaler.fit(raw_train_ds.get_raw_targets())

    train_ds = dataset_cls(dataset_source, transform=train_tf, file_list=train_files, target_scaler=scaler)
    val_ds = dataset_cls(dataset_source, transform=eval_tf, file_list=val_files, target_scaler=scaler)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_ds, val_ds, train_loader, val_loader, scaler, split_manifest


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_model(cfg: BaselineConfig) -> nn.Module:
    if cfg.method in {"der", "der_elbo"}:
        return ResNetDER(
            architecture=cfg.architecture,
            pretrained=cfg.pretrained,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

    if cfg.method == "mc_dropout":
        return BackboneGaussianRegressor(
            architecture=cfg.architecture,
            pretrained=cfg.pretrained,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

    if cfg.method == "bbb":
        return BackboneBBBRegressor(
            architecture=cfg.architecture,
            pretrained=cfg.pretrained,
            hidden_dim=cfg.hidden_dim,
            prior_std=cfg.bbb_prior_std,
            init_rho=cfg.bbb_init_rho,
        )

    raise ValueError(f"Unsupported baseline method: {cfg.method}")


def train(cfg: BaselineConfig) -> None:
    cfg.method = normalize_baseline_method(cfg.method)
    if cfg.method not in AVAILABLE_BASELINE_METHODS:
        raise ValueError(f"method must be one of {AVAILABLE_BASELINE_METHODS}")
    if cfg.selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {VALID_SELECTION_METRICS}")
    if cfg.eval_samples <= 0:
        raise ValueError("eval_samples must be positive")
    if cfg.train_samples <= 0:
        raise ValueError("train_samples must be positive")
    if cfg.bbb_prior_std <= 0:
        raise ValueError("bbb_prior_std must be positive")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    save_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))

    train_ds, val_ds, train_loader, val_loader, scaler, split_manifest = build_loaders(cfg)
    save_json(os.path.join(cfg.output_dir, "split_manifest.json"), split_manifest)
    print(
        f"using split sizes: train={len(train_ds)} val={len(val_ds)} "
        f"test={len(split_manifest.get('test_files', []))}"
    )
    if cfg.method == "bbb" and cfg.weight_decay > 0:
        print("note: BBB already has KL regularization; consider tuning with --weight-decay 0.0 as well.")

    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict[str, float]] = []
    best_metric_value = float("inf")
    best_epoch = -1

    loss_fn = None
    if cfg.method == "der":
        loss_fn = evidential_regression_loss
    elif cfg.method == "der_elbo":
        loss_fn = evidential_regression_elbo_loss

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running: Dict[str, float] = {"train_loss": 0.0}
        steps = 0

        synchronize_if_cuda(cfg.device)
        train_start = time.perf_counter()
        prog = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for xb, yb in prog:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            if cfg.method in {"der", "der_elbo"}:
                pred = model(xb)
                loss = loss_fn(pred, yb, lamb=cfg.lamb)
                postfix = {"loss": f"{loss.item():.3f}"}
            elif cfg.method == "mc_dropout":
                mean, var = model(xb)
                loss = gaussian_nll_loss(mean, yb, var)
                postfix = {"loss": f"{loss.item():.3f}"}
            else:
                data_loss = 0.0
                for _ in range(cfg.train_samples):
                    mean, var = model(xb, sample=True)
                    data_loss = data_loss + gaussian_nll_loss(mean, yb, var)
                data_loss = data_loss / cfg.train_samples
                kl_term = model.kl_divergence() / max(1, len(train_ds))
                loss = data_loss + cfg.kl_weight * kl_term
                running["train_data_loss"] = running.get("train_data_loss", 0.0) + float(data_loss.item())
                running["train_kl"] = running.get("train_kl", 0.0) + float(kl_term.item())
                postfix = {"loss": f"{loss.item():.3f}", "kl": f"{kl_term.item():.3f}"}

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            steps += 1
            running["train_loss"] += float(loss.item())
            prog.set_postfix(postfix)
        synchronize_if_cuda(cfg.device)
        train_time_sec = time.perf_counter() - train_start

        if cfg.method in {"der", "der_elbo"}:
            pred = predict_der(model, val_loader, scaler, cfg.device)
        else:
            pred = predict_sampled_gaussian(
                model=model,
                loader=val_loader,
                scaler=scaler,
                device=cfg.device,
                num_samples=cfg.eval_samples,
                use_mc_dropout=cfg.method == "mc_dropout",
            )

        val_basic = regression_metrics(pred["y_true"], pred["y_pred"])

        record: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": running["train_loss"] / max(1, steps),
            "train_time_sec": train_time_sec,
            "val_rmse": val_basic["rmse"],
            "val_mae": val_basic["mae"],
            "val_nll": float(pred["nll"]),
        }
        for key, value in running.items():
            if key != "train_loss":
                record[key] = value / max(1, steps)

        history.append(record)
        save_json(os.path.join(cfg.output_dir, "history.json"), history)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.std_,
            "config": asdict(cfg),
            "metrics": record,
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "last.pt"))

        current_metric = select_metric(record, cfg.selection_metric)
        if current_metric < best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch
            torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={record['train_loss']:.4f} "
            f"train_time={record['train_time_sec']:.2f}s "
            f"val_rmse={record['val_rmse']:.4f} "
            f"val_mae={record['val_mae']:.4f} "
            f"val_nll={record['val_nll']:.4f}"
        )

    print(f"best validation {cfg.selection_metric}: {best_metric_value:.4f} at epoch {best_epoch}")
    print(f"artifacts written to: {cfg.output_dir}")


def parse_args() -> BaselineConfig:
    p = argparse.ArgumentParser(description="Train baseline uncertainty models on UTKFace age regression")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--packed-data",
        type=str,
        default=None,
        help="Optional SQLite dataset created by pack_utkface.py. Splits still use UTKFace filenames.",
    )
    p.add_argument(
        "--method",
        type=normalize_baseline_method,
        default="der",
        choices=AVAILABLE_BASELINE_METHODS,
        help="Baseline to train: der, der_elbo, mc_dropout, or bbb.",
    )
    p.add_argument(
        "--architecture",
        type=normalize_architecture,
        default="resnet18",
        choices=AVAILABLE_ARCHITECTURES,
        help="Backbone to use: resnet18, cnn3, cnn4, vgg11, vgg13, or vgg16.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-manifest", type=str, default=None)
    p.add_argument(
        "--selection-metric",
        type=str,
        default="val_rmse",
        choices=VALID_SELECTION_METRICS,
        help="Metric used to decide which epoch becomes best.pt.",
    )
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lamb", type=float, default=1e-3)
    p.add_argument("--eval-samples", type=int, default=30, help="Predictive samples for MC Dropout and BBB validation.")
    p.add_argument("--train-samples", type=int, default=1, help="Posterior samples per minibatch for BBB training.")
    p.add_argument("--kl-weight", type=float, default=1.0, help="Multiplier on the BBB KL term.")
    p.add_argument("--bbb-prior-std", type=float, default=1.0)
    p.add_argument("--bbb-init-rho", type=float, default=-5.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return BaselineConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
