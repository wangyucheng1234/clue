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
    ResNetCLUE,
    gaussian_nll_loss,
    normalize_architecture,
    rbf_rho_features,
    sample_anchor_images,
)


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    packed_data: str | None = None
    variant: str = "fixed_noise"
    architecture: str = "resnet18"
    seed: int = 0
    split_seed: int = 0
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    pretrain_epochs: int = 0
    lr: float = 1e-3
    pretrain_lr: float | None = None
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_ratio: float = 0.1
    split_manifest: str | None = None
    selection_metric: str = "val_rmse"
    pretrained: bool = True
    hidden_dim: int = 256
    proj_dim: int = 256
    dropout: float = 0.0
    n_anchors: int = 512
    ell: float = 0.5
    ell_mu: float = 0.5
    ell_sigma: float = 1.0
    sigma_obs: float = 0.15
    sigma_f2: float = 1.0
    prior_mean: float = 0.0
    prior_var: float = 1.0
    logsigma_prior_mean: float = 0.0
    logsigma_prior_var: float = 1.0
    anchor_severity: float = 0.10
    mc_samples: int = 16
    kl_sigma_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


VALID_SELECTION_METRICS = ("val_rmse", "val_mae", "val_nll")


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


def gaussian_nll(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_mean = y_mean.reshape(-1)
    y_var = np.maximum(y_var.reshape(-1), 1e-8)
    return float(np.mean(0.5 * np.log(2.0 * math.pi * y_var) + 0.5 * (y_true - y_mean) ** 2 / y_var))


def select_metric(record: Dict[str, object], metric_name: str) -> float:
    if metric_name not in VALID_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {VALID_SELECTION_METRICS}")
    if metric_name not in record:
        raise KeyError(f"Metric '{metric_name}' was not found in the training record.")
    return float(record[metric_name])


def kl_normal_diag(mean: torch.Tensor, var: torch.Tensor, prior_mean: float, prior_var: float) -> torch.Tensor:
    pm = torch.as_tensor(prior_mean, dtype=mean.dtype, device=mean.device)
    pv = torch.as_tensor(prior_var, dtype=mean.dtype, device=mean.device)
    var = var.clamp_min(1e-8)
    return 0.5 * ((var + (mean - pm).pow(2)) / pv - 1.0 + torch.log(pv) - torch.log(var))


@torch.no_grad()
def predict_fixed_noise(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
    sigma_obs: float,
    device: str,
):
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
    }


@torch.no_grad()
def predict_variable_noise(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: str,
):
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

    # E[sigma^2] where logsigma ~ N(mu_logsigma, var_logsigma)
    # sigma^2 = exp(2 * logsigma), so E[sigma^2] = exp(2*mu + 2*var).
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
    }


def elbo_ood_batch_images_fixednoise(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    n_anchors: int,
    dataset_size: int,
    ell: float,
    sigma_obs: float,
    sigma_f2: float,
    prior_mean: float,
    prior_var: float,
    anchor_severity: float,
):
    batch_size = xb.shape[0]
    device = xb.device

    anchor_imgs = sample_anchor_images(xb, severity=anchor_severity)
    if anchor_imgs.shape[0] > n_anchors:
        keep = torch.randperm(anchor_imgs.shape[0], device=device)[:n_anchors]
        anchor_imgs = anchor_imgs[keep]

    mu_a, var_a, feat_a = model(anchor_imgs, return_features=True)
    feat_b = model.extract_features(xb)

    m_a = mu_a.squeeze(-1)
    v_a = var_a.squeeze(-1)
    rho = rbf_rho_features(feat_a.detach(), feat_b.detach(), ell=ell)
    
    # rho = rbf_rho_features(anchor_imgs.view(anchor_imgs.size(0), -1), xb.view(xb.size(0), -1), ell=ell)
    # print(rho)

    tau2 = sigma_f2 * (1.0 - rho.pow(2))
    tau2 = tau2.clamp_min(1e-8)
    var_y = tau2 + sigma_obs ** 2

    y_vec = yb.squeeze(-1)
    y_exp = y_vec.unsqueeze(0)
    m_exp = m_a.unsqueeze(1)
    v_exp = v_a.unsqueeze(1)

    resid2 = (y_exp - rho * m_exp).pow(2) + rho.pow(2) * v_exp
    loglik = -0.5 * torch.log(2.0 * math.pi * var_y) - 0.5 * resid2 / var_y

    data_term = (dataset_size / batch_size) * (loglik.sum() / max(1, anchor_imgs.shape[0]))

    kl_term = kl_normal_diag(m_a, v_a, prior_mean=prior_mean, prior_var=prior_var).mean()
    elbo = (data_term - kl_term)
    return elbo, {
        "data_term": float(data_term.detach().item()),
        "kl_term": float(kl_term.detach().item()),
        "mean_anchor_var": float(v_a.mean().detach().item()),
        "mean_rho": float(rho.mean().detach().item()),
    }


def elbo_ood_batch_images_variable_noise(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    n_anchors: int,
    dataset_size: int,
    ell_mu: float,
    ell_sigma: float,
    prior_mean: float,
    prior_var: float,
    logsigma_prior_mean: float,
    logsigma_prior_var: float,
    anchor_severity: float,
    mc_samples: int,
    kl_sigma_weight: float,
):
    batch_size = xb.shape[0]
    device = xb.device

    anchor_imgs = sample_anchor_images(xb, severity=anchor_severity)
    if anchor_imgs.shape[0] > n_anchors:
        keep = torch.randperm(anchor_imgs.shape[0], device=device)[:n_anchors]
        anchor_imgs = anchor_imgs[keep]

    mu_a, var_a, mu_b, var_b, feat_a = model(anchor_imgs, return_features=True)
    feat_b = model.extract_features(xb)

    rho_mu = rbf_rho_features(feat_a.detach(), feat_b.detach(), ell=ell_mu)
    rho_sigma = rbf_rho_features(feat_a.detach(), feat_b.detach(), ell=ell_sigma)

    mu_a = mu_a.squeeze(-1)
    var_a = var_a.squeeze(-1).clamp_min(1e-8)
    mu_b = mu_b.squeeze(-1)
    var_b = var_b.squeeze(-1).clamp_min(1e-8)

    eps_mu = torch.randn((mu_a.shape[0], mc_samples), device=device, dtype=mu_a.dtype)
    eps_sigma = torch.randn((mu_b.shape[0], mc_samples), device=device, dtype=mu_b.dtype)
    f_anchor = mu_a.unsqueeze(-1) + eps_mu * torch.sqrt(var_a).unsqueeze(-1)               # [K,S]
    logs_anchor = mu_b.unsqueeze(-1) + eps_sigma * torch.sqrt(var_b).unsqueeze(-1)         # [K,S]

    cond_mu_mean = rho_mu.unsqueeze(-1) * f_anchor.unsqueeze(1)                              # [K,B,S]
    cond_mu_var = (prior_var * (1.0 - rho_mu.pow(2))).clamp_min(1e-8).unsqueeze(-1)         # [K,B,1]

    cond_logs_mean = rho_sigma.unsqueeze(-1) * logs_anchor.unsqueeze(1)                      # [K,B,S]
    cond_logs_var = (logsigma_prior_var * (1.0 - rho_sigma.pow(2))).clamp_min(1e-8).unsqueeze(-1)

    eps_cond_mu = torch.randn_like(cond_mu_mean)
    eps_cond_logs = torch.randn_like(cond_logs_mean)
    mu_samples = cond_mu_mean + eps_cond_mu * torch.sqrt(cond_mu_var)
    logsigma_samples = cond_logs_mean + eps_cond_logs * torch.sqrt(cond_logs_var)

    y = yb.squeeze(-1).unsqueeze(0).unsqueeze(-1)                                            # [1,B,1]
    var_y = torch.exp(2.0 * logsigma_samples).clamp_min(1e-8)
    sq = (y - mu_samples).pow(2)
    logp = -0.5 * math.log(2.0 * math.pi) - logsigma_samples - 0.5 * sq / var_y             # [K,B,S]
    log_mix = torch.logsumexp(logp, dim=-1) - math.log(mc_samples)                           # [K,B]
    data_term = (dataset_size / batch_size) * (log_mix.sum() / max(1, anchor_imgs.shape[0]))

    kl_mu = kl_normal_diag(mu_a, var_a, prior_mean=prior_mean, prior_var=prior_var).mean()
    kl_sigma = kl_normal_diag(
        mu_b,
        var_b,
        prior_mean=logsigma_prior_mean,
        prior_var=logsigma_prior_var,
    ).mean()
    kl_term = kl_mu + kl_sigma_weight * kl_sigma
    elbo = data_term - kl_term
    return elbo, {
        "data_term": float(data_term.detach().item()),
        "kl_mu": float(kl_mu.detach().item()),
        "kl_sigma": float(kl_sigma.detach().item()),
        "mean_mu_rho": float(rho_mu.mean().detach().item()),
        "mean_sigma_rho": float(rho_sigma.mean().detach().item()),
        "mean_mu_var": float(var_a.mean().detach().item()),
        "mean_logsigma_var": float(var_b.mean().detach().item()),
    }


def resolve_train_val_split(cfg: TrainConfig) -> Tuple[List[str], List[str], Dict[str, object]]:
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
            source="generated_in_train_clue",
        )

    if not train_files or not val_files:
        raise ValueError("Training requires non-empty train and validation splits.")

    return train_files, val_files, split_manifest


def build_loaders(cfg: TrainConfig):
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


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def enable_full_model_training(model: ResNetCLUE) -> None:
    set_requires_grad(model, True)


def enable_head_only_finetuning(model: ResNetCLUE) -> None:
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.projector, False)
    set_requires_grad(model.shared_head, True)
    set_requires_grad(model.out, True)


def set_model_mode_for_phase(model: ResNetCLUE, head_only: bool) -> None:
    model.train()
    if head_only:
        # Keep the frozen feature extractor deterministic during head-only finetuning.
        model.encoder.eval()
        model.projector.eval()


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters were found for the optimizer.")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def supervised_pretrain_loss(
    cfg: TrainConfig,
    model: ResNetCLUE,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if cfg.variant == "fixed_noise":
        mu, var_f = model(xb)
        pred_var = var_f + cfg.sigma_obs ** 2
    else:
        mu, var_f, mu_logsigma, var_logsigma = model(xb)
        ale_var = torch.exp(2.0 * mu_logsigma + 2.0 * var_logsigma).clamp_min(1e-8)
        pred_var = var_f + ale_var

    loss = gaussian_nll_loss(mu, yb, pred_var)
    return loss, {
        "mean_pred_var": float(pred_var.mean().detach().item()),
    }


def evaluate_model(
    cfg: TrainConfig,
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler1D,
) -> Dict[str, object]:
    if cfg.variant == "fixed_noise":
        pred = predict_fixed_noise(model, loader, scaler, cfg.sigma_obs, cfg.device)
    else:
        pred = predict_variable_noise(model, loader, scaler, cfg.device)

    val_basic = regression_metrics(pred["y_true"], pred["y_pred"])
    val_nll = gaussian_nll(pred["y_true"], pred["y_pred"], pred["predictive_var"])
    return {
        "val_rmse": val_basic["rmse"],
        "val_mae": val_basic["mae"],
        "val_nll": val_nll,
    }


def build_epoch_record(
    stage: str,
    stage_epoch: int,
    global_epoch: int,
    running: Dict[str, float],
    steps: int,
    train_time_sec: float,
    eval_result: Dict[str, object],
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "epoch": global_epoch,
        "stage": stage,
        "stage_epoch": stage_epoch,
        "train_loss": running["train_loss"] / max(1, steps),
        "train_time_sec": float(train_time_sec),
        "val_rmse": float(eval_result["val_rmse"]),
        "val_mae": float(eval_result["val_mae"]),
        "val_nll": float(eval_result["val_nll"]),
    }
    for key, value in running.items():
        if key != "train_loss":
            record[f"train_{key}"] = value / max(1, steps)
    return record


def build_checkpoint(
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler1D,
    record: Dict[str, object],
) -> Dict[str, object]:
    return {
        "epoch": int(record["epoch"]),
        "stage": record["stage"],
        "stage_epoch": int(record["stage_epoch"]),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "config": asdict(cfg),
        "metrics": record,
    }


def train(cfg: TrainConfig) -> None:
    if cfg.variant not in {"fixed_noise", "variable_noise"}:
        raise ValueError("variant must be 'fixed_noise' or 'variable_noise'")
    if cfg.selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {VALID_SELECTION_METRICS}")
    if cfg.pretrain_epochs < 0:
        raise ValueError("pretrain_epochs must be non-negative")
    if cfg.epochs < 0:
        raise ValueError("epochs must be non-negative")
    if cfg.pretrain_epochs == 0 and cfg.epochs == 0:
        raise ValueError("At least one of pretrain_epochs or epochs must be positive.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    save_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))

    train_ds, val_ds, train_loader, val_loader, scaler, split_manifest = build_loaders(cfg)
    save_json(os.path.join(cfg.output_dir, "split_manifest.json"), split_manifest)
    print(
        f"using split sizes: train={len(train_ds)} val={len(val_ds)} "
        f"test={len(split_manifest.get('test_files', []))}"
    )
    model = ResNetCLUE(
        variant=cfg.variant,
        architecture=cfg.architecture,
        pretrained=cfg.pretrained,
        hidden_dim=cfg.hidden_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
        l2_normalize_features=True,
    ).to(cfg.device)

    history: List[Dict[str, object]] = []
    best_metric_value = float("inf")
    best_epoch = -1

    if cfg.pretrain_epochs > 0:
        enable_full_model_training(model)
        pretrain_lr = cfg.pretrain_lr if cfg.pretrain_lr is not None else cfg.lr
        optimizer = build_optimizer(model, lr=pretrain_lr, weight_decay=cfg.weight_decay)
        total_params, trainable_params = count_parameters(model)
        print(
            f"starting supervised pretraining for {cfg.pretrain_epochs} epochs "
            f"(trainable_params={trainable_params}/{total_params}, lr={pretrain_lr:.3e})"
        )

        pretrain_best_metric = float("inf")
        pretrain_best_epoch = -1
        track_pretrain_as_main = cfg.epochs == 0

        for stage_epoch in range(1, cfg.pretrain_epochs + 1):
            global_epoch = len(history) + 1
            set_model_mode_for_phase(model, head_only=False)
            running: Dict[str, float] = {"train_loss": 0.0}
            steps = 0

            synchronize_if_cuda(cfg.device)
            train_start = time.perf_counter()
            prog = tqdm(train_loader, desc=f"pretrain {stage_epoch}/{cfg.pretrain_epochs}", leave=False)
            for xb, yb in prog:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)

                loss, stats = supervised_pretrain_loss(cfg, model, xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                steps += 1
                running["train_loss"] += float(loss.item())
                for key, value in stats.items():
                    running[key] = running.get(key, 0.0) + float(value)
                prog.set_postfix({"loss": f"{loss.item():.3f}", "var": f"{stats['mean_pred_var']:.3f}"})
            synchronize_if_cuda(cfg.device)
            train_time_sec = time.perf_counter() - train_start

            eval_result = evaluate_model(cfg, model, val_loader, scaler)
            record = build_epoch_record(
                stage="pretrain",
                stage_epoch=stage_epoch,
                global_epoch=global_epoch,
                running=running,
                steps=steps,
                train_time_sec=train_time_sec,
                eval_result=eval_result,
            )

            history.append(record)
            save_json(os.path.join(cfg.output_dir, "history.json"), history)

            ckpt = build_checkpoint(cfg, model, optimizer, scaler, record)
            torch.save(ckpt, os.path.join(cfg.output_dir, "pretrain_last.pt"))
            torch.save(ckpt, os.path.join(cfg.output_dir, "last.pt"))

            current_metric = select_metric(record, cfg.selection_metric)
            if current_metric < pretrain_best_metric:
                pretrain_best_metric = current_metric
                pretrain_best_epoch = global_epoch
                torch.save(ckpt, os.path.join(cfg.output_dir, "pretrain_best.pt"))

            if track_pretrain_as_main and current_metric < best_metric_value:
                best_metric_value = current_metric
                best_epoch = global_epoch
                torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))

            print(
                f"[pretrain epoch {stage_epoch:03d}] "
                f"train_loss={float(record['train_loss']):.4f} "
                f"train_time={float(record['train_time_sec']):.2f}s "
                f"val_rmse={float(record['val_rmse']):.4f} "
                f"val_mae={float(record['val_mae']):.4f} "
                f"val_nll={float(record['val_nll']):.4f}"
            )

        print(f"best pretrain validation {cfg.selection_metric}: {pretrain_best_metric:.4f} at epoch {pretrain_best_epoch}")

    if cfg.epochs > 0:
        head_only = cfg.pretrain_epochs > 0
        if head_only:
            enable_head_only_finetuning(model)
        else:
            enable_full_model_training(model)

        optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_params, trainable_params = count_parameters(model)
        phase_name = "head-only CLUE finetuning" if head_only else "full-model CLUE training"
        print(
            f"starting {phase_name} for {cfg.epochs} epochs "
            f"(trainable_params={trainable_params}/{total_params}, lr={cfg.lr:.3e})"
        )

        for stage_epoch in range(1, cfg.epochs + 1):
            global_epoch = len(history) + 1
            set_model_mode_for_phase(model, head_only=head_only)
            running: Dict[str, float] = {"train_loss": 0.0}
            steps = 0

            synchronize_if_cuda(cfg.device)
            train_start = time.perf_counter()
            prog = tqdm(train_loader, desc=f"clue {stage_epoch}/{cfg.epochs}", leave=False)
            for xb, yb in prog:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)

                if cfg.variant == "fixed_noise":
                    elbo, stats = elbo_ood_batch_images_fixednoise(
                        model=model,
                        xb=xb,
                        yb=yb,
                        n_anchors=cfg.n_anchors,
                        dataset_size=len(train_ds),
                        ell=cfg.ell,
                        sigma_obs=cfg.sigma_obs,
                        sigma_f2=cfg.sigma_f2,
                        prior_mean=cfg.prior_mean,
                        prior_var=cfg.prior_var,
                        anchor_severity=cfg.anchor_severity,
                    )
                    postfix = {"loss": f"{-elbo.item():.3f}", "rho": f"{stats['mean_rho']:.3f}"}
                else:
                    elbo, stats = elbo_ood_batch_images_variable_noise(
                        model=model,
                        xb=xb,
                        yb=yb,
                        n_anchors=cfg.n_anchors,
                        dataset_size=len(train_ds),
                        ell_mu=cfg.ell_mu,
                        ell_sigma=cfg.ell_sigma,
                        prior_mean=cfg.prior_mean,
                        prior_var=cfg.prior_var,
                        logsigma_prior_mean=cfg.logsigma_prior_mean,
                        logsigma_prior_var=cfg.logsigma_prior_var,
                        anchor_severity=cfg.anchor_severity,
                        mc_samples=cfg.mc_samples,
                        kl_sigma_weight=cfg.kl_sigma_weight,
                    )
                    postfix = {"loss": f"{-elbo.item():.3f}", "rho_mu": f"{stats['mean_mu_rho']:.3f}"}

                loss = -elbo
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                steps += 1
                running["train_loss"] += float(loss.item())
                for key, value in stats.items():
                    running[key] = running.get(key, 0.0) + float(value)
                prog.set_postfix(postfix)
            synchronize_if_cuda(cfg.device)
            train_time_sec = time.perf_counter() - train_start

            eval_result = evaluate_model(cfg, model, val_loader, scaler)
            stage_name = "head_finetune" if head_only else "clue_train"
            record = build_epoch_record(
                stage=stage_name,
                stage_epoch=stage_epoch,
                global_epoch=global_epoch,
                running=running,
                steps=steps,
                train_time_sec=train_time_sec,
                eval_result=eval_result,
            )

            history.append(record)
            save_json(os.path.join(cfg.output_dir, "history.json"), history)

            ckpt = build_checkpoint(cfg, model, optimizer, scaler, record)
            torch.save(ckpt, os.path.join(cfg.output_dir, "last.pt"))

            current_metric = select_metric(record, cfg.selection_metric)
            if current_metric < best_metric_value:
                best_metric_value = current_metric
                best_epoch = global_epoch
                torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))

            print(
                f"[clue epoch {stage_epoch:03d}] "
                f"train_loss={float(record['train_loss']):.4f} "
                f"train_time={float(record['train_time_sec']):.2f}s "
                f"val_rmse={float(record['val_rmse']):.4f} "
                f"val_mae={float(record['val_mae']):.4f} "
                f"val_nll={float(record['val_nll']):.4f}"
            )

    print(f"best validation {cfg.selection_metric}: {best_metric_value:.4f} at epoch {best_epoch}")
    print(f"artifacts written to: {cfg.output_dir}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train CLUE on UTKFace age regression")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--packed-data",
        type=str,
        default=None,
        help="Optional SQLite dataset created by pack_utkface.py. Splits still use UTKFace filenames.",
    )
    p.add_argument("--variant", type=str, default="fixed_noise", choices=["fixed_noise", "variable_noise"])
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
    p.add_argument("--epochs", type=int, default=20, help="Number of CLUE finetuning epochs.")
    p.add_argument("--pretrain-epochs", type=int, default=0, help="Optional supervised full-model warmup before CLUE.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--pretrain-lr",
        type=float,
        default=None,
        help="Optional learning rate for the supervised pretraining stage. Defaults to --lr.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-manifest", type=str, default=None)
    p.add_argument("--selection-metric", type=str, default="val_rmse", choices=VALID_SELECTION_METRICS)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-anchors", type=int, default=512)
    p.add_argument("--ell", type=float, default=0.01)
    p.add_argument("--ell-mu", type=float, default=0.01)
    p.add_argument("--ell-sigma", type=float, default=0.01)
    p.add_argument("--sigma-obs", type=float, default=0.15)
    p.add_argument("--sigma-f2", type=float, default=1.0)
    p.add_argument("--prior-mean", type=float, default=0.0)
    p.add_argument("--prior-var", type=float, default=1.0)
    p.add_argument("--logsigma-prior-mean", type=float, default=0.0)
    p.add_argument("--logsigma-prior-var", type=float, default=1.0)
    p.add_argument("--anchor-severity", type=float, default=0.0001)
    p.add_argument("--mc-samples", type=int, default=512)
    p.add_argument("--kl-sigma-weight", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
