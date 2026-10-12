#!/usr/bin/env python3
# run_uci_all.py
# Unified UCI regression runner:
#   Methods: der (NIG), gaussian, mc_dropout, bbb, ensemble, swa, swag, laplace_ll
#   Predictives: Student-t (DER) or Gaussian (others)
#   Metrics: RMSE/MAE/NLL/R2/EVS/MAPE/sMAPE/RMSLE/Pearson r/CRPS/ECE/test time
#   Sweeps over splits/seeds, CSV logging, robust summary

import argparse, csv, os, math
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from uci_datasets import Dataset
import torch.nn.functional as F
from torch.distributions import StudentT
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import time

# ----------------------------
# Utilities / dataclass
# ----------------------------
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

@dataclass
class Metrics:
    dataset: str
    method: str
    split: int
    seed: int
    members: int            # #models/MC samples used to aggregate predictive
    rmse: float
    mae: float
    nll: float
    r2: float
    evs: float
    mape: float
    smape: float
    rmsle: float
    pearson_r: float
    spearman_r: float
    crps: float
    ece: float
    test_time_sec: float
    n: int

# ----------------------------
# Common metric helpers
# ----------------------------
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape_fn(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def safe_rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        return float("nan")
    return float(mean_squared_error(np.log1p(y_true - 1 + 1), np.log1p(y_pred - 1 + 1), squared=False))

# def pearson_r(y_true, y_pred):
#     y_true = np.asarray(y_true, float).ravel()
#     y_pred = np.asarray(y_pred, float).ravel()
#     if y_true.size < 2: return float("nan")
#     return float(np.corrcoef(y_true, y_pred)[0, 1])

def pearson_r(error, uncertainty):
    """Compute Pearson correlation coefficient between error and uncertainty."""
    # error = np.asarray(error, float).ravel()
    # uncertainty = np.asarray(uncertainty, float).ravel()
    # if error.size < 2: return float("nan")
    return float(np.corrcoef(error, uncertainty)[0, 1])

def spearman_r(error, uncertainty):
    """Compute Spearman correlation coefficient between error and uncertainty."""
    from scipy.stats import spearmanr
    # error = np.asarray(error, float).ravel()
    # uncertainty = np.asarray(uncertainty, float).ravel()
    # if error.size < 2: return float("nan")
    r, _ = spearmanr(error, uncertainty)
    return float(r)

# Closed-form CRPS for Gaussian predictive
def crps_gaussian(y, mu, sigma, eps=1e-12):
    from math import sqrt, pi
    y = np.asarray(y, float).ravel()
    mu = np.asarray(mu, float).ravel()
    sigma = np.maximum(np.asarray(sigma, float).ravel(), eps)
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + torch.erf(torch.from_numpy(z) / np.sqrt(2.0))).numpy()
    return float(np.mean(sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / sqrt(pi))))

# CRPS via MC for Student-t predictive in the chosen evaluation space
def crps_student_t_mc(df, loc, scale, y, n_samples=256, device="cuda"):
    y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)  # (N,1)
    dist = StudentT(df, loc=loc, scale=scale)
    X  = dist.rsample((n_samples,)).to(device)        # (S,N,1)
    Xp = dist.rsample((n_samples,)).to(device)        # (S,N,1)
    term1 = torch.mean(torch.abs(X - y), dim=0)
    term2 = 0.5 * torch.mean(torch.abs(X - Xp), dim=0)
    return float((term1 - term2).mean().item())

def expected_calibration_error_from_cdf(cdf_values, n_bins=10):
    """
    Regression ECE from predictive CDF values at the observed targets.
    Perfectly calibrated predictive distributions have P(F(Y) <= q) = q.
    """
    cdf_values = np.asarray(cdf_values, dtype=float).ravel()
    cdf_values = cdf_values[np.isfinite(cdf_values)]
    if cdf_values.size == 0 or n_bins < 2:
        return float("nan")
    cdf_values = np.clip(cdf_values, 0.0, 1.0)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    empirical = np.array([np.mean(cdf_values <= q) for q in quantiles], dtype=float)
    return float(np.mean(np.abs(empirical - quantiles)))

def sync_cuda_if_needed(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

def timed_test_eval(device, eval_fn):
    sync_cuda_if_needed(device)
    start = time.perf_counter()
    metrics = eval_fn()
    sync_cuda_if_needed(device)
    metrics["test_time_sec"] = time.perf_counter() - start
    return metrics

# ----------------------------
# Data
# ----------------------------
def load_split(name, split):
    data = Dataset(name)
    Xtr, Ytr, Xte, Yte = data.get_split(split=split)
    Xtr, Xte = np.asarray(Xtr, np.float32), np.asarray(Xte, np.float32)
    Ytr, Yte = np.asarray(Ytr, np.float32).reshape(-1, 1), np.asarray(Yte, np.float32).reshape(-1, 1)
    xsc = StandardScaler().fit(Xtr)
    ysc = StandardScaler().fit(Ytr)
    return xsc.transform(Xtr), ysc.transform(Ytr), xsc.transform(Xte), ysc.transform(Yte), ysc

def load_split_with_val(name, split, val_ratio=0.1, seed=0):
    data = Dataset(name)
    Xtr_full, Ytr_full, Xte, Yte = data.get_split(split=split)
    Xtr_full = np.asarray(Xtr_full, np.float32)
    Ytr_full = np.asarray(Ytr_full, np.float32).reshape(-1, 1)
    Xte = np.asarray(Xte, np.float32)
    Yte = np.asarray(Yte, np.float32).reshape(-1, 1)

    n = len(Xtr_full)
    if val_ratio <= 0.0 or n < 5:
        Xtr_raw, Ytr_raw = Xtr_full, Ytr_full
        Xval_raw = np.empty((0, Xtr_full.shape[1]), dtype=np.float32)
        Yval_raw = np.empty((0, 1), dtype=np.float32)
    else:
        idx = np.arange(n)
        tr_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, shuffle=True)
        Xtr_raw, Ytr_raw = Xtr_full[tr_idx], Ytr_full[tr_idx]
        Xval_raw, Yval_raw = Xtr_full[val_idx], Ytr_full[val_idx]

    xsc = StandardScaler().fit(Xtr_raw)
    ysc = StandardScaler().fit(Ytr_raw)
    Xtr = xsc.transform(Xtr_raw).astype(np.float32)
    Ytr = ysc.transform(Ytr_raw).astype(np.float32)
    Xval = xsc.transform(Xval_raw).astype(np.float32) if len(Xval_raw) else Xval_raw
    Yval = ysc.transform(Yval_raw).astype(np.float32) if len(Yval_raw) else Yval_raw
    Xte = xsc.transform(Xte).astype(np.float32)
    Yte = ysc.transform(Yte).astype(np.float32)
    return Xtr, Ytr, Xval, Yval, Xte, Yte, ysc

def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def restore_state_dict(model, state_dict):
    model.load_state_dict(state_dict)
    return model

class EarlyStoppingRMSE:
    def __init__(self, patience=0, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_rmse = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.best_state = None
        self.last_improved = False

    @property
    def enabled(self):
        return self.patience > 0

    def step(self, model, epoch, val_rmse):
        self.last_improved = val_rmse < (self.best_rmse - self.min_delta)
        if self.last_improved:
            self.best_rmse = float(val_rmse)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = clone_state_dict(model)
        else:
            self.bad_epochs += 1
        return self.enabled and self.bad_epochs >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            restore_state_dict(model, self.best_state)
        return model

# ==============================================================
# ====== DER / NIG (Student-t predictive) ======================
# ==============================================================
class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 4)
    def forward(self, x):
        out = self.fc(x)
        mu, logv, loga, logb = torch.chunk(out, 4, dim=-1)
        v = F.softplus(logv) + 1e-6
        alpha = F.softplus(loga) + 1.0
        beta = F.softplus(logb) + 1e-6
        return mu, v, alpha, beta

def evidential_regression(pred, y, lamb=1e-3):
    mu, v, alpha, beta = pred
    twoBlambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=y.device) / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    ).mean()
    error = torch.abs(y - mu)
    evidence = 2.0 * v + alpha
    reg = (error * evidence).mean()
    return nll + lamb * reg

def evidential_regression_elbo(pred, y, lamb=1e-3):
    mu, v, alpha, beta = pred
    twoBlambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=y.device) / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    ).mean()
    # KL divergence term
    kl = (
        -0.5 * torch.log(v)
        + alpha * torch.log(beta)
        - torch.lgamma(alpha)
        + torch.lgamma(alpha + 0.5)
        - (alpha + 0.5)
        + (alpha + 0.5) * (v * (y - mu) ** 2 + 2.0 * beta * (1.0 + v)) / (2.0 * beta * (1.0 + v))
    ).mean()
    return nll + lamb * kl

class MLP_NIG(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers, last = [], in_dim
        for h in hidden: layers += [nn.Linear(last, h), act()]; last = h
        layers += [NormalInvGamma(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


@torch.no_grad()
def predict_params_nig(model, X, device):
    model.eval()
    mu_all, v_all, a_all, b_all = [], [], [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    for (xb,) in loader:
        xb = xb.to(device)
        mu, v, alpha, beta = model(xb)
        mu_all.append(mu.cpu()); v_all.append(v.cpu()); a_all.append(alpha.cpu()); b_all.append(beta.cpu())
    return [torch.cat(t, dim=0) for t in (mu_all, v_all, a_all, b_all)]

def evaluate_der(model, X, Y, ysc, device, crps_samples=256, ece_bins=10):
    mu, v, alpha, beta = predict_params_nig(model, X, device)
    # Student-t predictive
    df = 2.0 * alpha
    scale_sq = beta * (1.0 + v) / (v * alpha)
    scale = torch.sqrt(torch.clamp(scale_sq, min=1e-12))
    loc = mu
    # NLL in normalized space
    dist = StudentT(df, loc=loc, scale=scale)
    nll = float(-dist.log_prob(torch.from_numpy(Y).float()).mean().item())
    cdf_values = student_t_dist.cdf(
        Y.ravel(),
        df=df.numpy().ravel(),
        loc=loc.numpy().ravel(),
        scale=scale.numpy().ravel(),
    )
    ece_v = expected_calibration_error_from_cdf(cdf_values, n_bins=ece_bins)
    # Mean back to original scale
    y_pred = ysc.inverse_transform(mu.numpy())
    y_true = ysc.inverse_transform(Y)
    y_scale = float(ysc.scale_.ravel()[0])
    scale_orig = scale.numpy() * y_scale
    # rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    evs  = float(explained_variance_score(y_true, y_pred))
    mape_v  = safe_mape(y_true, y_pred)
    smape_v = smape_fn(y_true, y_pred)
    rmsle_v = safe_rmsle(y_true, y_pred)
    # pr      = pearson_r(y_true, y_pred)

    #Uncertainty Quantification
    errors = np.abs(y_true - y_pred).ravel()
    uncerts = (beta / (v * (alpha - 1.0))).ravel()
    # print(uncerts.shape)
    # print(errors.shape)
    pr = pearson_r(errors, uncerts.numpy())
    sr = spearman_r(errors, uncerts.numpy())

    crps_v  = crps_student_t_mc(
        df=df.to(device),
        loc=torch.as_tensor(y_pred, dtype=torch.float32, device=device),
        scale=torch.as_tensor(scale_orig, dtype=torch.float32, device=device),
        y=y_true,
        n_samples=crps_samples,
        device=device,
    )
    return {"rmse": rmse, "mae": mae, "nll": nll, "r2": r2, "evs": evs,
            "mape": mape_v, "smape": smape_v, "rmsle": rmsle_v,
            "pearson_r": pr, "spearman_r": sr, "crps": crps_v, "ece": ece_v, "n": len(Y)}


def predict_der(model, X, device):
    mu, v, alpha, beta = predict_params_nig(model, X, device)

    uncerts = (beta / (v * (alpha - 1.0))).ravel()

    return mu.numpy(), uncerts.numpy()

# ==============================================================
# ====== Gaussian family (Gaussian / MC / Ens / SWA / SWAG) ====
# ==============================================================
class MLP_Gauss(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), activation="relu", dropout_p=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            if dropout_p > 0: layers += [nn.Dropout(dropout_p)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, 2)  # mu, log_sigma
        self.softplus = nn.Softplus()

        
    def forward(self, x):
        h = self.backbone(x)
        mu, log_s = torch.chunk(self.head(h), 2, dim=-1)
        sigma = self.softplus(log_s) + 1e-6
        return mu, sigma
    def features(self, x):
        return self.backbone(x)

class BayesianLinearBBB(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0, rho_init=-5.0):
        super().__init__()
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be positive for Bayes-by-Backprop")
        self.prior_sigma = float(prior_sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters(rho_init=rho_init)

    def reset_parameters(self, rho_init=-5.0):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in = self.weight_mu.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, rho_init)
        nn.init.constant_(self.bias_rho, rho_init)

    @staticmethod
    def _sigma(rho):
        return F.softplus(rho) + 1e-6

    def forward(self, x, sample=True):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        if sample:
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        prior_var = self.prior_sigma ** 2
        prior_log_sigma = math.log(self.prior_sigma)
        weight_kl = 0.5 * torch.sum(
            (weight_sigma.pow(2) + self.weight_mu.pow(2)) / prior_var
            - 1.0 + 2.0 * prior_log_sigma - 2.0 * torch.log(weight_sigma)
        )
        bias_kl = 0.5 * torch.sum(
            (bias_sigma.pow(2) + self.bias_mu.pow(2)) / prior_var
            - 1.0 + 2.0 * prior_log_sigma - 2.0 * torch.log(bias_sigma)
        )
        return weight_kl + bias_kl

class MLP_BBB(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), activation="relu", prior_sigma=1.0):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        self.activation = act()
        self.layers = nn.ModuleList()
        last = in_dim
        for h in hidden:
            self.layers.append(BayesianLinearBBB(last, h, prior_sigma=prior_sigma))
            last = h
        self.head = BayesianLinearBBB(last, 2, prior_sigma=prior_sigma)
        self.softplus = nn.Softplus()

    def forward(self, x, sample=True):
        h = x
        for layer in self.layers:
            h = self.activation(layer(h, sample=sample))
        mu, log_s = torch.chunk(self.head(h, sample=sample), 2, dim=-1)
        sigma = self.softplus(log_s) + 1e-6
        return mu, sigma

    def kl_loss(self):
        return sum(layer.kl_loss() for layer in self.layers) + self.head.kl_loss()

# class MLP_Gauss_LearnedNoise(nn.Module):
#     def __init__(self, in_dim, hidden=(128,128), activation="relu", dropout_p=0.0):
#         super().__init__()
#         act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
#         layers, last = [], in_dim
#         for h in hidden:
#             layers += [nn.Linear(last, h), act()]
#             if dropout_p > 0: layers += [nn.Dropout(dropout_p)]
#             last = h
#         self.backbone = nn.Sequential(*layers)
        
#         layers2, last2 = [], in_dim
#         for h in hidden:
#             layers2 += [nn.Linear(last2, h), act()]
#             if dropout_p > 0: layers2 += [nn.Dropout(dropout_p)]
#             last2 = h
#         self.backbone2 = nn.Sequential(*layers2)
        
#         self.head = nn.Linear(last, 1)  # mu, log_sigma
#         self.head2 = nn.Linear(last2, 1)  # mu, log_sigma
#         self.softplus = nn.Softplus()

#         self.noise = nn.Parameter(torch.tensor(-10.0))
#     def forward(self, x):
#         h = self.backbone(x)
#         mu = self.head(h)
#         h2 = self.backbone2(x)
#         log_s = self.head2(h2)
#         # sigma = torch.nn.functional.relu(log_s) + 1e-6
#         sigma = self.softplus(log_s) + 1e-6
#         return mu, sigma
#     def features(self, x):
#         return self.backbone(x)

class MLP_Gauss_LearnedNoise(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), activation="relu", dropout_p=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            if dropout_p > 0: layers += [nn.Dropout(dropout_p)]
            last = h
        self.backbone = nn.Sequential(*layers)
        
        self.head = nn.Linear(last, 2)  # mu, log_sigma
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.backbone(x)
        # mu = self.head(h)
        mu, log_s = torch.chunk(self.head(h), 2, dim=-1)
        # sigma = torch.nn.functional.relu(log_s) + 1e-6
        sigma = self.softplus(log_s) + 1e-6
        return mu, sigma 
    def features(self, x):
        return self.backbone(x)

# class MLP_Two_Gauss(nn.Module):
#     def __init__(self, in_dim, hidden=(128,128), activation="relu", dropout_p=0.0):
#         super().__init__()
#         act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
#         layers, last = [], in_dim
#         for h in hidden:
#             layers += [nn.Linear(last, h), act()]
#             if dropout_p > 0: layers += [nn.Dropout(dropout_p)]
#             last = h
#         self.backbone = nn.Sequential(*layers)

#         layers2, last2 = [], in_dim
#         for h in hidden:
#             layers2 += [nn.Linear(last2, h), act()]
#             if dropout_p > 0: layers2 += [nn.Dropout(dropout_p)]
#             last2 = h
#         self.backbone2 = nn.Sequential(*layers2)

#         self.head1 = nn.Linear(last, 2)  # mu1, log_sigma1
#         self.head2 = nn.Linear(last, 2)  # mu2, log_sigma2
#         # self.head2 = nn.Linear(last2, 2)  # mu2, log_sigma2
#         self.softplus = nn.Softplus()
#     def forward(self, x):
#         h = self.backbone(x)
#         mu1, mu2 = torch.chunk(self.head1(h), 2, dim=-1)
#         h2 = self.backbone2(x)
#         log_s1, log_s2 = torch.chunk(self.head2(h2), 2, dim=-1)
#         # mu2, log_s2 = torch.chunk(self.head2(h), 2, dim=-1)
#         var1 = self.softplus(log_s1) + 1e-6
#         var2 = self.softplus(log_s2) + 1e-6
#         # mu2 = -1.89 * torch.ones_like(mu1)
#         # var2 = 0.0001 * torch.ones_like(var1)
#         return mu1, var1, mu2, var2
#     def features(self, x):
#         return self.backbone(x)

class MLP_Two_Gauss(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), activation="relu", dropout_p=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            if dropout_p > 0: layers += [nn.Dropout(dropout_p)]
            last = h
        self.backbone = nn.Sequential(*layers)

        self.head1 = nn.Linear(last, 4)  # mu1, log_sigma1
        self.softplus = nn.Softplus()
    def forward(self, x):
        h = self.backbone(x)
        mu1, log_s1, mu2, log_s2 = torch.chunk(self.head1(h), 4, dim=-1)
        # mu2, log_s2 = torch.chunk(self.head2(h), 2, dim=-1)
        var1 = self.softplus(log_s1) + 1e-6
        var2 = self.softplus(log_s2) + 1e-6
        # mu2 = -1.89 * torch.ones_like(mu1)
        # var2 = 0.0001 * torch.ones_like(var1)
        return mu1, var1, mu2, var2
    def features(self, x):
        return self.backbone(x)

def gaussian_nll_torch(mu, sigma, y):
    return 0.5 * (torch.log(2*torch.tensor(math.pi, device=y.device)) + 2*torch.log(sigma) + ((y - mu)**2) / (sigma**2))

# ----------------------------
# RBF correlation between anchor x and data x_n
# ----------------------------
ell = 0.01  # lengthscale

ell_mu = 0.01  # for heterogeneous noise, lengthscale for mu
ell_logsigma = 10 # for heterogeneous noise, lengthscale for log_sigma

# def rbf_rho(xq, Xtrain, ell):
#     """
#     xq:      (K,1)   query / anchor inputs
#     Xtrain:  (B,1)   training batch inputs
#     returns: (K,B)   rho_ij = exp(-||xq_i - Xtrain_j||^2 / (2 ell^2))
#     """
#     print(xq.shape, Xtrain.shape)
#     diff = xq - Xtrain.T         # (K,B)
#     rho = torch.exp(-0.5 * (diff**2) / (ell**2))
#     return rho

# def rbf_rho(xq, Xtrain, ell):
#     """
#     xq:      (K,C)   query / anchor inputs
#     Xtrain:  (B,C)   training batch inputs
#     returns: (K,B)   rho_ij = exp(-||xq_i - Xtrain_j||^2 / (2 ell^2))
#     """
#     diff = xq.unsqueeze(1) - Xtrain.unsqueeze(0)  # (K,B,C)
#     rho = torch.exp(-0.5 * (diff**2).sum(dim=-1) / (ell**2))
#     return rho

def pairwise_sqdist(xq: torch.Tensor, Xtrain: torch.Tensor) -> torch.Tensor:
    """
    returns (K,B) squared Euclidean distances
    """
    diff = xq.unsqueeze(1) - Xtrain.unsqueeze(0)   # (K,B,C)
    return (diff ** 2).sum(dim=-1)                 # (K,B)

def pairwise_l1dist(xq: torch.Tensor, Xtrain: torch.Tensor) -> torch.Tensor:
    """
    returns (K,B) L1 distances
    """
    diff = xq.unsqueeze(1) - Xtrain.unsqueeze(0)   # (K,B,C)
    return diff.abs().sum(dim=-1)                  # (K,B)

def pairwise_dot(xq: torch.Tensor, Xtrain: torch.Tensor) -> torch.Tensor:
    """
    returns (K,B) dot products
    """
    return (xq.unsqueeze(1) * Xtrain.unsqueeze(0)).sum(dim=-1)  # (K,B)

def rbf_rho(xq, Xtrain, ell):
    d2 = pairwise_sqdist(xq, Xtrain)
    return torch.exp(-0.5 * d2 / (ell ** 2))

def matern12_rho(xq, Xtrain, ell):
    """
    Matérn ν=1/2 (Exponential): k(r)= exp(-r/ell)
    """
    d2 = pairwise_sqdist(xq, Xtrain)
    r = torch.sqrt(d2.clamp_min(1e-12))
    return torch.exp(-r / ell)

def matern32_rho(xq, Xtrain, ell):
    """
    Matérn ν=3/2: k(r)= (1 + sqrt(3) r/ell) exp(-sqrt(3) r/ell)
    """
    d2 = pairwise_sqdist(xq, Xtrain)
    r = torch.sqrt(d2.clamp_min(1e-12))
    a = math.sqrt(3.0) * r / ell
    return (1.0 + a) * torch.exp(-a)

def matern52_rho(xq, Xtrain, ell):
    """
    Matérn ν=5/2: k(r)= (1 + sqrt(5) r/ell + 5 r^2/(3 ell^2)) exp(-sqrt(5) r/ell)
    """
    d2 = pairwise_sqdist(xq, Xtrain)
    r = torch.sqrt(d2.clamp_min(1e-12))
    a = math.sqrt(5.0) * r / ell
    return (1.0 + a + (5.0/3.0) * (d2 / (ell ** 2))) * torch.exp(-a)

def laplacian_rho(xq, Xtrain, ell):
    """
    Laplacian: exp(-||x-x'||_1 / ell)
    """
    d1 = pairwise_l1dist(xq, Xtrain)
    return torch.exp(-d1 / ell)

def rational_quadratic_rho(xq, Xtrain, ell, alpha=1.0):
    """
    Rational Quadratic: (1 + ||x-x'||^2 / (2 alpha ell^2))^{-alpha}
    """
    d2 = pairwise_sqdist(xq, Xtrain)
    return (1.0 + d2 / (2.0 * alpha * (ell ** 2))).pow(-alpha)

def polynomial_rho(xq, Xtrain, degree=2, c=1.0, gamma=None):
    """
    Polynomial: (gamma <x,x'> + c)^degree
    If gamma is None, uses 1/C (common default).
    """
    C = xq.shape[-1]
    if gamma is None:
        gamma = 1.0 / C
    dot = pairwise_dot(xq, Xtrain)
    return (gamma * dot + c).pow(degree)

def cosine_rho(xq, Xtrain, eps=1e-12):
    """
    Cosine similarity kernel: <x,x'> / (||x|| ||x'||)
    """
    dot = pairwise_dot(xq, Xtrain)
    xq_n = xq.norm(dim=-1).clamp_min(eps).unsqueeze(1)       # (K,1)
    Xt_n = Xtrain.norm(dim=-1).clamp_min(eps).unsqueeze(0)   # (1,B)
    return dot / (xq_n * Xt_n)

def periodic_rho(xq, Xtrain, ell, p):
    """
    Periodic (stationary) kernel (common GP form):
    exp( -2 * sum_j sin^2(pi*(x_j-x'_j)/p) / ell^2 )
    p can be float or tensor broadcastable to (C,)
    """
    diff = xq.unsqueeze(1) - Xtrain.unsqueeze(0)  # (K,B,C)
    s2 = torch.sin(math.pi * diff / p).pow(2).sum(dim=-1)  # (K,B)
    return torch.exp(-2.0 * s2 / (ell ** 2))

# def sample_ood_x(x, device="cpu"):
#     """
#     Sample x outside the ID interval [id_range[0], id_range[1]]
#     from a larger outer interval ood_range.
#     """
#     mask = torch.randn_like(x) <  -1
#     # print(mask.float().mean())
#     # print(mask.float())
#     return x + torch.randn_like(x) * 0.2 * mask

def sample_ood_x(x, n_anchors, device="cpu"):
    """
    Sample x outside the ID interval [id_range[0], id_range[1]]
    from a larger outer interval ood_range.
    """
    indices = torch.randint(low=0, high=x.shape[0], size=(n_anchors,))

    # # Select elements
    choices = x[indices]
    mask = torch.randn_like(choices) <  -1
    # choices = x
    return choices + torch.randn_like(choices) * 0.2 * mask

# ----------------------------
# Prior over latent f(x)
# ----------------------------
# m0 = 0.0          # prior mean for f
# v0 = 1.0          # prior variance for f


mu_mean_prior = 0 #prior mean for f
mu_var_prior = 1  #prior std for

logsigma_mean_prior = 0 #heterogeneous noise case, prior mean for log-scale of observation noise
logsigma_var_prior = 1  #heterogeneous noise case, prior variance for log-scale of observation noise

v0 = torch.tensor(mu_var_prior)
m0 = torch.tensor(mu_mean_prior)
v1 = torch.tensor(logsigma_var_prior)
m1 = torch.tensor(logsigma_mean_prior)

sigma_f2 = v0     # notation match
SIGMA = 0.15                     # homogeneous noise case, assumed gaussian noise std
SIGMA2 = float(SIGMA**2)

# ----------------------------
# ELBO per minibatch, with OOD anchors
#
# For each OOD anchor x_a:
#   ELBO(x_a) = sum_{i in batch} E_q(f|x_a)[log p(y_i | f, x_a, x_i)]
#               - KL(q(f|x_a) || p(f))
# We approximate dataset sum by (N/B)*sum_{i in batch},
# and expectation over x_a by average over sampled anchors.
# ----------------------------

def elbo_ood_batch(model, xb, yb, n_anchors, N, kl_weight=1.0):
    """
    xb, yb: ID minibatch (B,1)
    n_anchors: number of OOD anchors sampled this step
    """
    B = xb.shape[0]

    # 1) Sample OOD anchors x_a
    # X_ood = sample_ood_x(n_anchors, device=xb.device)   # (K,1)
    X_ood = sample_ood_x(xb, n_anchors, device=xb.device)   # (K,1)
    # print("xb", xb)
    # print("X_ood", X_ood)
    K = X_ood.shape[0]

    # 2) Posterior q(f|x_a) for anchors
    mean_a, sigma_a = model(X_ood)                  # (K,1), (K,1)
    # print("sigma_a", sigma_a)
    # var_a = sigma_a**2                                  # (K,1)
    var_a = sigma_a
    m_a = mean_a.squeeze(-1)                            # (K,)
    v_a = var_a.squeeze(-1)                             # (K,)

    # 3) Pairwise correlations between anchors and batch points
    rho = rbf_rho(X_ood, xb, ell)                       # (K,B)
    # rho = matern32_rho(X_ood, xb, ell)                  # (K,B)
    # rho = matern12_rho(X_ood, xb, ell)                  # (K,B)

    tau2 = sigma_f2 * (1.0 - rho**2)                    # (K,B)
    var_y = tau2 + SIGMA2          # (K,B)  variance of y_i | f, x_a, x_i

    # 4) Expected log-likelihood term over q(f|x_a)
    #    E_q[(y_i - rho f)^2] = (y_i - rho m)^2 + rho^2 v
    Yb_vec = yb.squeeze(-1)                             # (B,)
    Yb_exp = Yb_vec.unsqueeze(0)                        # (1,B) -> (K,B)
    m_a_exp = m_a.unsqueeze(1)                          # (K,1) -> (K,B)
    v_a_exp = v_a.unsqueeze(1)                          # (K,1) -> (K,B)

    resid2 = (Yb_exp - rho * m_a_exp)**2 + (rho**2) * v_a_exp  # (K,B)

    loglik = -0.5 * torch.log(2 * math.pi * var_y) - 0.5 * resid2 / var_y  # (K,B)

    # Sum over batch points and anchors, scale by N/B (dataset size correction) and 1/K
    data_term = (N / B) * (loglik.sum() / K)

    # 5) KL term: average over anchors, independent prior p(f) = N(m0, v0)
    v0_t = torch.tensor(v0, dtype=torch.float32, device=xb.device)
    m0_t = torch.tensor(m0, dtype=torch.float32, device=xb.device)
    kl_each = 0.5 * (
        (v_a + (m_a - m0_t)**2) / v0_t - 1.0 + torch.log(v0_t) - torch.log(v_a)
    )  # (K,)
    kl_term = kl_weight * kl_each.mean()

    elbo_batch = data_term - kl_term
    # elbo_batch = elbo_batch / N  # normalize per data point in batch

    return elbo_batch, X_ood  # return anchors if you want to inspect them

# heterogeneous noise
N_SAMPLES = 512 #Number of Monte Carlo samples to estimate ELBO

def elbo_ood_batch_var_noise(model, xb, yb, n_anchors, N, kl_weight=1.0):
    """
    xb, yb: ID minibatch (B,1)
    n_anchors: number of OOD anchors sampled this step
    """
    B = xb.shape[0]

    # 1) Sample OOD anchors x_a
    X_ood = sample_ood_x(xb, n_anchors, device=xb.device)   # (K,1)
    K = X_ood.shape[0]

    # 2) Posterior q(mu, sigma|x_a) for anchors
    mean_a, var_a, mean_b, var_b = model(X_ood)                        # (K,1)
    m_a = mean_a                            # (K,1)
    v_a = var_a                             # (K,1)
    m_b = mean_b                            # (K,1)
    v_b = var_b                             # (K,1)

    # 3) Sample mu, logsigma from q(mu, logsigma|x_a) for anchors
    m_a_sample = m_a + torch.randn_like(m_a) * torch.sqrt(v_a) #(K, 1)
    m_b_sample = m_b + torch.randn_like(m_b) * torch.sqrt(v_b) #(K, 1)

    # 4) Pairwise correlations between anchors and batch points
    rho_mu = rbf_rho(X_ood, xb, ell_mu)                       # (K,B)
    rho_sigma = rbf_rho(X_ood, xb, ell_logsigma)                 # (K,B)

    # 5) Sample mu_i|mu and sigma_i|sigma for data
    mu_i_mu = torch.div(rho_mu, v0) * m_a_sample
    mu_i_sigma2 = v0 - torch.div(rho_mu**2, v0)
    logsigma_i_mu = torch.div(rho_sigma, v1) * m_b_sample
    logsigma_i_sigma2 = v1 - torch.div(rho_sigma**2, v1) #(K,B)
    mu_samples = mu_i_mu.unsqueeze(-1) + torch.randn_like(mu_i_mu.unsqueeze(-1).repeat(1, 1, N_SAMPLES)) * torch.sqrt(mu_i_sigma2).unsqueeze(-1)# (K, B, N_SAMPLES)
    logsigma_samples = logsigma_i_mu.unsqueeze(-1) + torch.randn_like(logsigma_i_mu.unsqueeze(-1).repeat(1, 1, N_SAMPLES)) * torch.sqrt(logsigma_i_sigma2).unsqueeze(-1)# (K, B, N_SAMPLES)

    # 6) Expected log-likelihood term over q(f|x_a)
    #    E_q[(y_i - rho f)^2] = (y_i - rho m)^2 + rho^2 v
    Yb_vec = yb.squeeze(-1)                             # (B,)
    Yb_exp = Yb_vec.unsqueeze(0).unsqueeze(-1)                        # (1,B) -> (K,B, N_SAMPLES)
    log_sigma = logsigma_samples               # (K,B,S)
    # Optional numerical safety clamp:
    # log_sigma = torch.clamp(log_sigma, min=-20.0, max=20.0)

    # log N(y | mu, sigma) where sigma = exp(log_sigma)
    # = -0.5*log(2π) - log_sigma - 0.5*((y-mu)^2 / exp(2 log_sigma))
    inv_var = torch.exp(-2.0 * log_sigma)      # 1/sigma^2
    # print(Yb_exp.shape)
    # print(mu_samples.shape)
    sq = (Yb_exp - mu_samples) ** 2                 # (K,B,S)
    logp = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * sq * inv_var  # (K,B,S)
    # logp = -s
    # Mixture log-likelihood: logmeanexp over samples
    log_mix = torch.log(torch.mean(torch.exp(logp), dim=-1) + 1e-6)   # (K,B)

    # If you want summed over batch:
    log_mix_sumB = log_mix.sum(dim=-1)                      # (K,)

    # Sum over batch points and anchors, scale by N/B (dataset size correction) and 1/K
    data_term = (N / B) * (log_mix_sumB.sum() / (K))

    # 5) KL term: average over anchors, independent prior p(f) = N(m0, v0)
    v0_t = torch.tensor(v0, dtype=torch.float32, device=xb.device)
    m0_t = torch.tensor(m0, dtype=torch.float32, device=xb.device)
    kl_each_mu = 0.5 * (
        (v_a + (m_a - m0_t)**2) / v0_t - 1.0 + torch.log(v0_t) - torch.log(v_a)
    )  # (K,)
    kl_term_mu = kl_each_mu.mean()

    v1_t = torch.tensor(v1, dtype=torch.float32, device=xb.device)
    m1_t = torch.tensor(m1, dtype=torch.float32, device=xb.device)
    kl_each_sigma = 0.5 * (
        (v_b + (m_b - m1_t)**2) / v1_t - 1.0 + torch.log(v1_t) - torch.log(v_b)
    )
    kl_term_sigma = kl_each_sigma.mean()

    elbo_batch = data_term - kl_weight * (kl_term_mu + kl_term_sigma)
    return elbo_batch, X_ood  # return anchors if you want to inspect them

def train_gaussian(model, Xtr, Ytr, device, epochs=200, batch_size=128, lr=1e-3, verbose=False,
                   Xval=None, Yval=None, early_stop_patience=0, min_delta=0.0,
                   early_stop_check_every=1, label="Gaussian"):
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    stopper = make_early_stopper(early_stop_patience, min_delta, Xval)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sigma = model(xb)
            loss = gaussian_nll_torch(mu, sigma, yb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        if verbose and ep % max(epochs//5,1) == 0:
            print(f"  [Gaussian] ep{ep:03d} loss={total/len(ds):.4f}")
        if maybe_stop_early(
            model, stopper, ep, val_rmse_gaussian, Xval, Yval, device,
            label=label, verbose=verbose, check_every=early_stop_check_every
        ):
            break
    if stopper is not None:
        stopper.restore_best(model)
    model.trained_epochs = ep
    return model

def train_bbb(model, Xtr, Ytr, device, epochs=200, batch_size=128, lr=1e-3,
              bbb_kl_weight=1.0, verbose=False,
              Xval=None, Yval=None, early_stop_patience=0, min_delta=0.0,
              early_stop_check_every=1, label="BBB"):
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = make_early_stopper(early_stop_patience, min_delta, Xval)
    num_train = max(len(ds), 1)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        total_mse = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sigma = model(xb, sample=True)
            nll = gaussian_nll_torch(mu, sigma, yb).mean()
            kl = model.kl_loss() / num_train
            loss = nll + bbb_kl_weight * kl
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
            total_mse += ((mu - yb)**2).mean().item() * xb.size(0)
        if verbose and ep % max(epochs//5,1) == 0:
            print(f"  [BBB] ep{ep:03d} loss={total/len(ds):.4f} mse={total_mse/len(ds):.4f}")
        if maybe_stop_early(
            model, stopper, ep, val_rmse_bbb, Xval, Yval, device,
            label=label, verbose=verbose, check_every=early_stop_check_every
        ):
            break
    if stopper is not None:
        stopper.restore_best(model)
    model.trained_epochs = ep
    return model

def train_gaussian_swa(model, Xtr, Ytr, device, epochs=200, batch_size=128, lr=1e-3, swa_start=0.75, verbose=False):
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(base_opt, anneal_strategy="cos", anneal_epochs=max(1, int((1.0-swa_start)*epochs)), swa_lr=lr*0.5)
    model.train()
    swa_begin = int(swa_start * epochs)
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sigma = model(xb)
            loss = gaussian_nll_torch(mu, sigma, yb).mean()
            base_opt.zero_grad(); loss.backward(); base_opt.step()
            total += loss.item() * xb.size(0)
        if ep >= swa_begin:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if verbose and ep % max(epochs//5,1) == 0:
            print(f"  [SWA] ep{ep:03d} loss={total/len(ds):.4f}")
    update_bn(loader, swa_model, device=device)  # refresh BN stats
    return swa_model

@torch.no_grad()
def predict_gaussian(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_all, sig_all = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        mu, sigma = model(xb)
        mu_all.append(mu.cpu().numpy()); sig_all.append(sigma.cpu().numpy())
    return np.vstack(mu_all), np.vstack(sig_all)

@torch.no_grad()
def predict_gaussian_muonly(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_all = []
    for (xb,) in loader:
        xb = xb.to(device)
        mu = model(xb)
        mu_all.append(mu.cpu().numpy())
    return np.vstack(mu_all)

@torch.no_grad()
def predict_gaussian_def_fixnoise(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_all, sig_all, unc_all = [], [], []
    for (xb,) in loader:
        xb = xb.to(device)
        mu, epistemic_var = model(xb)
        total_var = torch.clamp(epistemic_var + SIGMA2, min=1e-12)
        sigma_all = torch.sqrt(total_var)
        mu_all.append(mu.cpu().numpy()); sig_all.append(sigma_all.cpu().numpy()); unc_all.append(epistemic_var.cpu().numpy())
    return np.vstack(mu_all), np.vstack(sig_all), np.vstack(unc_all)

@torch.no_grad()
def predict_gaussian_def_variable_noise(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_all, sig_all, unc_all = [], [], []
    for (xb,) in loader:
        xb = xb.to(device)
        mu1, mu_var, log_sigma_mean, log_sigma_var = model(xb)
        # Match the first two moments of the Gaussian mixture induced by
        # uncertainty in both the predictive mean and log noise scale.
        obs_var = torch.exp(torch.clamp(2.0 * log_sigma_mean + 1.0 * log_sigma_var, max=30.0))
        total_var = torch.clamp(mu_var + obs_var, min=1e-12)
        sigma_all = torch.sqrt(total_var)
        mu_all.append(mu1.cpu().numpy()); sig_all.append(sigma_all.cpu().numpy()); unc_all.append(mu_var.cpu().numpy())
    return np.vstack(mu_all), np.vstack(sig_all), np.vstack(unc_all)

@torch.no_grad()
def predict_mc_dropout(model, X, device, T=20):
    model.train()  # keep dropout on
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_runs, var_runs = [], []
    for _ in range(T):
        mu_all, var_all = [], []
        for (xb,) in loader:
            xb = xb.to(device)
            mu, sigma = model(xb)
            mu_all.append(mu.cpu().numpy())
            var_all.append((sigma.cpu().numpy())**2)
        mu_t = np.vstack(mu_all); var_t = np.vstack(var_all)
        mu_runs.append(mu_t); var_runs.append(var_t)
    mu_runs = np.stack(mu_runs, axis=0)   # (T,N,1)
    var_runs = np.stack(var_runs, axis=0) # (T,N,1)
    mu_mean = mu_runs.mean(axis=0)
    mu_var = mu_runs.var(axis=0, ddof=1)
    total_var = var_runs.mean(axis=0) + mu_runs.var(axis=0, ddof=1)
    sigma = np.sqrt(np.maximum(total_var, 1e-12))
    return mu_mean, sigma, mu_var

@torch.no_grad()
def predict_bbb_mean(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_all, sig_all = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        mu, sigma = model(xb, sample=False)
        mu_all.append(mu.cpu().numpy())
        sig_all.append(sigma.cpu().numpy())
    return np.vstack(mu_all), np.vstack(sig_all)

@torch.no_grad()
def predict_bbb(model, X, device, T=20):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=4096)
    mu_runs, var_runs = [], []
    for _ in range(max(int(T), 1)):
        mu_all, var_all = [], []
        for (xb,) in loader:
            xb = xb.to(device)
            mu, sigma = model(xb, sample=True)
            mu_all.append(mu.cpu().numpy())
            var_all.append((sigma.cpu().numpy())**2)
        mu_runs.append(np.vstack(mu_all))
        var_runs.append(np.vstack(var_all))
    mu_runs = np.stack(mu_runs, axis=0)
    var_runs = np.stack(var_runs, axis=0)
    mu_mean = mu_runs.mean(axis=0)
    mu_var = mu_runs.var(axis=0, ddof=1) if mu_runs.shape[0] > 1 else np.zeros_like(mu_mean)
    total_var = var_runs.mean(axis=0) + mu_var
    sigma = np.sqrt(np.maximum(total_var, 1e-12))
    return mu_mean, sigma, mu_var

@torch.no_grad()
def predict_ensemble(models, X, device):
    preds = [predict_gaussian(m, X, device) for m in models]
    mus = np.stack([p[0] for p in preds], axis=0)    # (M,N,1)
    vars_= np.stack([p[1]**2 for p in preds], axis=0)# (M,N,1)
    mu_mean = mus.mean(axis=0)
    mu_var = mus.var(axis=0, ddof=1)
    total_var = vars_.mean(axis=0) + mus.var(axis=0, ddof=1)
    sigma = np.sqrt(np.maximum(total_var, 1e-12))
    return mu_mean, sigma, mu_var

def predict_ensemble_no_sigma(models, X, device):
    preds = [predict_gaussian_muonly(m, X, device) for m in models]
    mus = np.stack([p for p in preds], axis=0)    # (M,N,1)
    mu_mean = mus.mean(axis=0)
    mu_var = mus.var(axis=0, ddof=1)
    return mu_mean, mu_var

def normalized_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

@torch.no_grad()
def val_rmse_der(model, Xval, Yval, device):
    if Xval is None or len(Xval) == 0:
        return float("inf")
    mu, _, _, _ = predict_params_nig(model, Xval, device)
    return normalized_rmse(Yval, mu.numpy())

@torch.no_grad()
def val_rmse_gaussian(model, Xval, Yval, device):
    if Xval is None or len(Xval) == 0:
        return float("inf")
    mu, _ = predict_gaussian(model, Xval, device)
    return normalized_rmse(Yval, mu)

@torch.no_grad()
def val_rmse_bbb(model, Xval, Yval, device):
    if Xval is None or len(Xval) == 0:
        return float("inf")
    mu, _ = predict_bbb_mean(model, Xval, device)
    return normalized_rmse(Yval, mu)

@torch.no_grad()
def val_rmse_corr_fix(model, Xval, Yval, device):
    if Xval is None or len(Xval) == 0:
        return float("inf")
    mu, _, _ = predict_gaussian_def_fixnoise(model, Xval, device)
    return normalized_rmse(Yval, mu)

@torch.no_grad()
def val_rmse_corr_var(model, Xval, Yval, device):
    if Xval is None or len(Xval) == 0:
        return float("inf")
    mu, _, _ = predict_gaussian_def_variable_noise(model, Xval, device)
    return normalized_rmse(Yval, mu)

@torch.no_grad()
def val_rmse_muonly(model, Xval, Yval, device):
    if Xval is None or len(Xval) == 0:
        return float("inf")
    mu = predict_gaussian_muonly(model, Xval, device)
    return normalized_rmse(Yval, mu)

def make_early_stopper(patience, min_delta, Xval):
    if patience <= 0 or Xval is None or len(Xval) == 0:
        return None
    return EarlyStoppingRMSE(patience=patience, min_delta=min_delta)

def maybe_stop_early(model, stopper, epoch, val_rmse_fn, Xval, Yval, device,
                     label, verbose=False, check_every=1):
    if stopper is None or epoch % max(int(check_every), 1) != 0:
        return False
    val_rmse = val_rmse_fn(model, Xval, Yval, device)
    should_stop = stopper.step(model, epoch, val_rmse)
    if verbose and (stopper.last_improved or should_stop):
        print(
            f"  [{label}] ep{epoch:04d} val_rmse={val_rmse:.4f} "
            f"best_ep={stopper.best_epoch} bad_epochs={stopper.bad_epochs}"
        )
    if should_stop:
        print(
            f"  [{label}] early stopping at ep{epoch:04d}; "
            f"best_val_rmse={stopper.best_rmse:.4f} at ep{stopper.best_epoch}"
        )
    return should_stop

# ---- SWAG (diagonal) utils ----
def flatten_params(model):
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters() if p.requires_grad])
def assign_params(model, flat):
    i = 0
    for p in model.parameters():
        if not p.requires_grad: continue
        num = p.numel()
        p.data.copy_(flat[i:i+num].view_as(p).to(p.device))
        i += num
def swag_collect_snapshots(model, snaps, max_snaps=None):
    flat = flatten_params(model).numpy()
    snaps.append(flat)
    if (max_snaps is not None) and (len(snaps) > max_snaps): snaps.pop(0)
def swag_stats(snaps):
    W = np.stack(snaps, axis=0)
    mean = W.mean(axis=0)
    var = W.var(axis=0, ddof=1) if W.shape[0] > 1 else np.zeros_like(mean)
    return mean, var
@torch.no_grad()
def predict_swag(model, mean, var, X, device, K=30):
    D = mean.shape[0]; std = np.sqrt(np.maximum(var, 1e-12))
    mu_runs, var_runs = [], []
    base = flatten_params(model)
    for _ in range(K):
        sample = mean + std * np.random.randn(D)
        assign_params(model, torch.from_numpy(sample).float())
        mu, sigma = predict_gaussian(model, X, device)
        mu_runs.append(mu); var_runs.append(sigma**2)
    assign_params(model, base)  # restore
    mu_runs = np.stack(mu_runs, axis=0); var_runs = np.stack(var_runs, axis=0)
    mu_mean = mu_runs.mean(axis=0)
    total_var = var_runs.mean(axis=0) + mu_runs.var(axis=0, ddof=1)
    sigma = np.sqrt(np.maximum(total_var, 1e-12))
    return mu_mean, sigma

# ==============================================================
# ====== Last-layer Laplace (Bayesian linear on features) =======
# ==============================================================
class MLP_MuOnly(nn.Module):
    def __init__(self, in_dim, hidden=(128,128), activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers, last = [], in_dim
        for h in hidden: layers += [nn.Linear(last, h), act()]; last = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, 1)
    def forward(self, x):
        h = self.backbone(x); return self.mu_head(h)
    def features(self, x):
        return self.backbone(x)

def fit_bayesian_linear(Phi, y, alpha_grid=(1e-6,1e-4,1e-2,1,10)):
    N, D = Phi.shape
    best = None
    for alpha in alpha_grid:
        A = alpha * np.eye(D, dtype=np.float64) + Phi.T @ Phi
        w = np.linalg.solve(A, Phi.T @ y)
        resid = y - Phi @ w
        sigma2 = float((resid**2).mean())
        if (best is None) or (sigma2 < best[0]):
            best = (sigma2, alpha, w, sigma2)
    _, alpha, w_map, sigma2 = best
    S = np.linalg.inv(alpha * np.eye(Phi.shape[1], dtype=np.float64) + (Phi.T @ Phi) / sigma2)
    w_mean = S @ (Phi.T @ y / sigma2)
    return alpha, sigma2, w_mean, S

def predict_bayesian_linear(Phi, w_mean, S, sigma2):
    mu = Phi @ w_mean
    var = sigma2 + np.sum(Phi @ S * Phi, axis=1, keepdims=True)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu, sigma

# ==============================================================
# ====== Evaluation wrappers ===================================
# ==============================================================
# def evaluate_gaussian(y_true_norm, mu_norm, sigma_norm, y_scaler):
#     mu = y_scaler.inverse_transform(mu_norm)
#     y = y_scaler.inverse_transform(y_true_norm)
#     sigma = sigma_norm * float(y_scaler.scale_.ravel()[0])
#     rmse = float(mean_squared_error(y, mu, squared=False))
#     mae  = float(mean_absolute_error(y, mu))
#     r2   = float(r2_score(y, mu))
#     evs  = float(explained_variance_score(y, mu))
#     mape_v  = safe_mape(y, mu)
#     smape_v = smape_fn(y, mu)
#     rmsle_v = safe_rmsle(y, mu)
#     pr      = pearson_r(y, mu)
#     var = np.maximum(sigma**2, 1e-12)
#     nll = float(0.5 * np.mean(np.log(2*np.pi) + np.log(var) + ((y - mu)**2)/var))
#     crps_v = crps_gaussian(y, mu, sigma)
#     return {"rmse": rmse, "mae": mae, "nll": nll, "r2": r2, "evs": evs,
#             "mape": mape_v, "smape": smape_v, "rmsle": rmsle_v,
#             "pearson_r": pr, "crps": crps_v, "n": len(y)}

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, root_mean_squared_error
from scipy.stats import norm, t as student_t_dist

# assuming these helpers are already defined somewhere:
# safe_mape, smape_fn, safe_rmsle, pearson_r, crps_gaussian

def evaluate_gaussian(y_true_norm, mu_norm, sigma_norm, y_scaler, epistemic_uncertainty=None,
                      nll_space="norm", ece_bins=10):
    """
    Evaluate Gaussian regression predictions.

    Args:
        y_true_norm: ndarray, normalized target values.
        mu_norm: ndarray, predicted means in normalized space.
        sigma_norm: ndarray, predicted stds in normalized space.
        y_scaler: fitted sklearn StandardScaler for y.
        nll_space: "norm" (default) or "orig" — compute NLL in normalized or original space.

    Returns:
        dict with metrics: rmse, mae, nll, r2, evs, mape, smape, rmsle, pearson_r, crps, ece, n
    """
    # --- transform for original-scale metrics ---
    y_true = y_scaler.inverse_transform(y_true_norm)
    mu     = y_scaler.inverse_transform(mu_norm)
    sigma  = sigma_norm * float(y_scaler.scale_.ravel()[0])  # broadcast scalar
    sigma_norm_safe = np.maximum(np.asarray(sigma_norm, dtype=float), 1e-12)
    cdf_values = norm.cdf(
        np.asarray(y_true_norm, dtype=float).ravel(),
        loc=np.asarray(mu_norm, dtype=float).ravel(),
        scale=sigma_norm_safe.ravel(),
    )
    ece_v = expected_calibration_error_from_cdf(cdf_values, n_bins=ece_bins)

    # --- NLL computation ---
    if nll_space == "norm":
        # all normalized
        y_n, mu_n, sig_n = y_true_norm, mu_norm, sigma_norm
        var = np.maximum(sig_n**2, 1e-3)
        nll = float(0.5 * np.mean(np.log(2 * np.pi) + np.log(var) + ((y_n - mu_n) ** 2) / var))
    elif nll_space == "orig":
        var = np.maximum(sigma**2, 1e-12)
        nll = float(0.5 * np.mean(np.log(2 * np.pi) + np.log(var) + ((y_true - mu) ** 2) / var))
    else:
        raise ValueError(f"Invalid nll_space={nll_space}, must be 'norm' or 'orig'.")

    # --- other metrics in original space ---
    # rmse = float(mean_squared_error(y_true, mu, squared=False))
    rmse = float(root_mean_squared_error(y_true, mu))
    mae  = float(mean_absolute_error(y_true, mu))
    r2   = float(r2_score(y_true, mu))
    evs  = float(explained_variance_score(y_true, mu))
    mape_v  = safe_mape(y_true, mu)
    smape_v = smape_fn(y_true, mu)
    rmsle_v = safe_rmsle(y_true, mu)
    # pr      = pearson_r(y_true, mu)
    crps_v  = crps_gaussian(y_true, mu, sigma)

    #Uncertainty Quantification
    errors = np.abs(y_true - mu).ravel()
    pr = float("nan")
    sr = float("nan")
    if epistemic_uncertainty is not None:
        uncerts = epistemic_uncertainty.ravel()
        pr = pearson_r(errors, uncerts)
        sr = spearman_r(errors, uncerts)

    return {
        "rmse": rmse, "mae": mae, "nll": nll, "r2": r2, "evs": evs,
        "mape": mape_v, "smape": smape_v, "rmsle": rmsle_v,
        "pearson_r": pr, "spearman_r": sr, "crps": crps_v, "ece": ece_v, "n": len(y_true),
        "nll_space": nll_space
    }


def evaluate_gaussian_outputs(y_true_norm, prediction, y_scaler, ece_bins=10):
    mu_norm, sigma_norm = prediction[0], prediction[1]
    epistemic_uncertainty = prediction[2] if len(prediction) > 2 else None
    return evaluate_gaussian(
        y_true_norm, mu_norm, sigma_norm, y_scaler,
        epistemic_uncertainty=epistemic_uncertainty,
        ece_bins=ece_bins,
    )


def empty_test_metrics(n):
    return {
        "rmse": float("nan"), "mae": float("nan"), "nll": float("nan"), "r2": float("nan"),
        "evs": float("nan"), "mape": float("nan"), "smape": float("nan"), "rmsle": float("nan"),
        "pearson_r": float("nan"), "spearman_r": float("nan"), "crps": float("nan"),
        "ece": float("nan"), "n": n
    }


# ==============================================================
# ====== One run per (method, split, seed) =====================
# ==============================================================
def run_once(
    method, dataset, split, seed, device,
    hidden=(128,128), activation="relu",
    epochs=200, batch_size=128, lr=1e-3,
    lamb=1e-3,                # DER regularizer
    clue_kl_weight=1.0,
    bbb_kl_weight=1.0, bbb_prior_sigma=1.0,
    dropout_p=0.1, mc_samples=20, ensemble_members=5,
    swa_start=0.75, swag_burn=0.5, swag_snap_every=5, swag_samples=30,
    crps_samples=256, ece_bins=10,
    val_ratio=0.1, early_stop_patience=0, min_delta=0.0,
    early_stop_check_every=1
):
    set_seed(seed)
    use_early_stop = early_stop_patience > 0 and val_ratio > 0.0
    if use_early_stop:
        Xtr, Ytr, Xval, Yval, Xte, Yte, ysc = load_split_with_val(
            dataset, split, val_ratio=val_ratio, seed=seed
        )
    else:
        Xtr, Ytr, Xte, Yte, ysc = load_split(dataset, split)
        Xval, Yval = None, None

    if method == "der":
        model = MLP_NIG(Xtr.shape[1], hidden=hidden, activation=activation).to(device)
        ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        stopper = make_early_stopper(early_stop_patience, min_delta, Xval)
        start = time.perf_counter()
        for ep in range(1, epochs+1):
            model.train()
            total = 0.0
            total_mse = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = evidential_regression(model(xb), yb, lamb=lamb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)

                #Calculate MSE for monitoring
                mu, _, _, _ = model(xb)
                mse = ((mu - yb)**2).mean().item()
                total_mse += mse * xb.size(0)
            if ep % max(epochs//5,1) == 0:
                print(f"  [DER] ep{ep:03d} loss={total/len(ds):.4f} mse={total_mse/len(ds):.4f}")
            if maybe_stop_early(
                model, stopper, ep, val_rmse_der, Xval, Yval, device,
                label="DER", verbose=False, check_every=early_stop_check_every
            ):
                break
        if stopper is not None:
            stopper.restore_best(model)
        end = time.perf_counter()
        print(f"Training Clock time: Epoch {ep}: {end - start:.4f} sec")
        metrics = timed_test_eval(
            device,
            lambda: evaluate_der(model, Xte, Yte, ysc, device, crps_samples=crps_samples, ece_bins=ece_bins)
        )
        members = 1

        #Testing purpose only
        mu, unc = predict_der(model, Xtr, device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.1 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.2 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.3 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.5 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())

    elif method == "der_elbo":
        model = MLP_NIG(Xtr.shape[1], hidden=hidden, activation=activation).to(device)
        ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        stopper = make_early_stopper(early_stop_patience, min_delta, Xval)
        start = time.perf_counter()
        for ep in range(1, epochs+1):
            model.train()
            total = 0.0
            total_mse = 0.0
            
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = evidential_regression_elbo(model(xb), yb, lamb=lamb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)

                #Calculate MSE for monitoring
                mu, _, _, _ = model(xb)
                mse = ((mu - yb)**2).mean().item()
                total_mse += mse * xb.size(0)
            if ep % max(epochs//5,1) == 0:
                print(f"  [DER-ELBO] ep{ep:03d} loss={total/len(ds):.4f} mse={total_mse/len(ds):.4f}")
            if maybe_stop_early(
                model, stopper, ep, val_rmse_der, Xval, Yval, device,
                label="DER-ELBO", verbose=False, check_every=early_stop_check_every
            ):
                break
        if stopper is not None:
            stopper.restore_best(model)
        end = time.perf_counter()
        print(f"Training Clock time: Epoch {ep}: {end - start:.4f} sec")
        metrics = timed_test_eval(
            device,
            lambda: evaluate_der(model, Xte, Yte, ysc, device, crps_samples=crps_samples, ece_bins=ece_bins)
        )
        members = 1

        #Testing purpose only
        start = time.perf_counter()
        mu, unc = predict_der(model, Xtr, device)
        end = time.perf_counter()
        print(f"Prediction Clock time: {end - start:.4f} sec")

        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.1 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.2 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.3 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, unc = predict_der(model, Xtr + 0.5 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())       

    elif method == "corr_der_fix_noise":
        model = MLP_Gauss_LearnedNoise(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=0.0).to(device)
        ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        num_train = Xtr.shape[0]
        stopper = make_early_stopper(early_stop_patience, min_delta, Xval)
        start = time.perf_counter()
        for ep in range(1, epochs+1):
            model.train()
            total = 0.0
            total_mse = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                # ELBO with OOD anchors
                elbo_batch, _ = elbo_ood_batch(
                    model, xb, yb, n_anchors=batch_size, N=num_train,
                    kl_weight=clue_kl_weight
                )
                loss = -elbo_batch
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)

                # Calculate MSE for monitoring
                mu, sigma = model(xb)
                mse = ((mu - yb)**2).mean().item()
                total_mse += mse * xb.size(0)
            # if ep % max(epochs//5,1) == 0:
            #     print(f"  [Corr-DER-FixNoise] ep{ep:03d} loss={total/len(ds):.4f} noise={torch.nn.functional.softplus(model.noise).item():.4f} mse={total_mse/len(ds):.4f}")
            if maybe_stop_early(
                model, stopper, ep, val_rmse_corr_fix, Xval, Yval, device,
                label="Corr-DER-FixNoise", verbose=False, check_every=early_stop_check_every
            ):
                break
        if stopper is not None:
            stopper.restore_best(model)
        end = time.perf_counter()
        print(f"Training Clock time: Epoch {ep}: {end - start:.4f} sec")
        # print("sigma", sigma)
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(
                Yte, predict_gaussian_def_fixnoise(model, Xte, device), ysc, ece_bins=ece_bins
            )
        )
        print(f"Prediction+evaluation Clock time: {metrics['test_time_sec']:.4f} sec")
        members = 1

        #Testing purpose only  
        mu, sigma, unc = predict_gaussian_def_fixnoise(model, Xtr, device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_fixnoise(model, Xtr + 0.1 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_fixnoise(model, Xtr + 0.2 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_fixnoise(model, Xtr + 0.3 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_fixnoise(model, Xtr + 0.5 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        # raise NotImplementedError("corr_der_fix_noise method is not implemented in this code snippet.")

    elif method == "corr_der_variable_noise":
        model = MLP_Two_Gauss(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=0.0).to(device)
        ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        num_train = Xtr.shape[0]
        stopper = make_early_stopper(early_stop_patience, min_delta, Xval)
        start = time.perf_counter()
        for ep in range(1, epochs+1):
            model.train()
            total = 0.0
            total_mse = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                # ELBO with OOD anchors
                elbo_batch, _ = elbo_ood_batch_var_noise(
                    model, xb, yb, n_anchors=batch_size, N=num_train,
                    kl_weight=clue_kl_weight
                )
                loss = -elbo_batch
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)

                # Calculate MSE for monitoring
                mu1, sigma1, mu2, sigma2 = model(xb)
                mse = ((mu1 - yb)**2).mean().item()
                total_mse += mse * xb.size(0)
            if ep % max(epochs//5,1) == 0:
                print(f"  [Corr-DER-VariableNoise] ep{ep:03d} loss={total/len(ds):.4f} mse={total_mse/len(ds):.4f}")
            if maybe_stop_early(
                model, stopper, ep, val_rmse_corr_var, Xval, Yval, device,
                label="Corr-DER-VariableNoise", verbose=False, check_every=early_stop_check_every
            ):
                break
        if stopper is not None:
            stopper.restore_best(model)
        end = time.perf_counter()
        # print(f"Training Clock time: Epoch {ep+1}: {end - start:.4f} sec")
        # print("sigma", sigma)
        # print("uct", unc)
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(
                Yte, predict_gaussian_def_variable_noise(model, Xte, device), ysc, ece_bins=ece_bins
            )
        )
        print(f"Prediction+evaluation Clock time: {metrics['test_time_sec']:.4f} sec")
        members = 1        

        #Testing Purpose Only
        mu, sigma, unc = predict_gaussian_def_variable_noise(model, Xtr, device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_variable_noise(model, Xtr + 0.1 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_variable_noise(model, Xtr + 0.2 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_variable_noise(model, Xtr + 0.3 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_gaussian_def_variable_noise(model, Xtr + 0.5 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())

        # raise NotImplementedError("corr_der_variable_noise method is not implemented in this code snippet.")
    elif method == "gaussian":
        model = MLP_Gauss(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=0.0).to(device)
        train_gaussian(
            model, Xtr, Ytr, device, epochs, batch_size, lr,
            Xval=Xval, Yval=Yval,
            early_stop_patience=early_stop_patience,
            min_delta=min_delta,
            early_stop_check_every=early_stop_check_every,
            label="Gaussian",
        )
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(Yte, predict_gaussian(model, Xte, device), ysc, ece_bins=ece_bins)
        )
        members = 1

    elif method == "mc_dropout":
        model = MLP_Gauss(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=dropout_p).to(device)
        start = time.perf_counter()
        train_gaussian(
            model, Xtr, Ytr, device, epochs, batch_size, lr,
            Xval=Xval, Yval=Yval,
            early_stop_patience=early_stop_patience,
            min_delta=min_delta,
            early_stop_check_every=early_stop_check_every,
            label="MC-Dropout",
        )
        end = time.perf_counter()
        print(f"Training Clock time: Epoch {getattr(model, 'trained_epochs', epochs)}: {end - start:.4f} sec")
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(
                Yte, predict_mc_dropout(model, Xte, device, T=mc_samples), ysc, ece_bins=ece_bins
            )
        )
        print(f"Prediction+evaluation Clock time: {metrics['test_time_sec']:.4f} sec")
        members = mc_samples

        #Testing purpose only
        print("MC Dropout uncertainty test:")
        mu, sigma, unc = predict_mc_dropout(model, Xtr, device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_mc_dropout(model, Xtr + 0.1 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_mc_dropout(model, Xtr + 0.2 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_mc_dropout(model, Xtr + 0.3 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_mc_dropout(model, Xtr + 0.5 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())

    elif method == "bbb":
        model = MLP_BBB(
            Xtr.shape[1], hidden=hidden, activation=activation, prior_sigma=bbb_prior_sigma
        ).to(device)
        start = time.perf_counter()
        train_bbb(
            model, Xtr, Ytr, device, epochs, batch_size, lr,
            bbb_kl_weight=bbb_kl_weight,
            Xval=Xval, Yval=Yval,
            early_stop_patience=early_stop_patience,
            min_delta=min_delta,
            early_stop_check_every=early_stop_check_every,
            label="BBB",
        )
        end = time.perf_counter()
        print(f"Training Clock time: Epoch {getattr(model, 'trained_epochs', epochs)}: {end - start:.4f} sec")
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(
                Yte, predict_bbb(model, Xte, device, T=mc_samples), ysc, ece_bins=ece_bins
            )
        )
        print(f"Prediction+evaluation Clock time: {metrics['test_time_sec']:.4f} sec")
        members = mc_samples

        # Testing purpose only
        print("BBB uncertainty test:")
        mu, sigma, unc = predict_bbb(model, Xtr, device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_bbb(model, Xtr + 0.1 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_bbb(model, Xtr + 0.2 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_bbb(model, Xtr + 0.3 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())
        mu, sigma, unc = predict_bbb(model, Xtr + 0.5 * np.random.randn(*Xtr.shape), device, T=mc_samples)
        print(unc.mean())

    elif method == "ensemble":
        models = []
        start = time.perf_counter()
        for m in range(ensemble_members):
            set_seed(seed + m + 12345)
            model = MLP_Gauss(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=0.0).to(device)
            train_gaussian(
                model, Xtr, Ytr, device, epochs, batch_size, lr,
                Xval=Xval, Yval=Yval,
                early_stop_patience=early_stop_patience,
                min_delta=min_delta,
                early_stop_check_every=early_stop_check_every,
                label=f"Ensemble member {m+1}",
            )
            models.append(model)
        end = time.perf_counter()
        print(f"Training Clock time: max Epoch {epochs} with {ensemble_members} ensemble members: {end - start:.4f} sec")
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(Yte, predict_ensemble(models, Xte, device), ysc, ece_bins=ece_bins)
        )
        print(f"Prediction+evaluation Clock time: {metrics['test_time_sec']:.4f} sec")
        members = ensemble_members

        #Testing purpose only
        print("Ensemble members uncertainty test:")
        mu, sigma, unc = predict_ensemble(models, Xtr, device)
        print(unc.mean())
        mu, sigma, unc = predict_ensemble(models, Xtr + 0.1 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_ensemble(models, Xtr + 0.2 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_ensemble(models, Xtr + 0.3 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())
        mu, sigma, unc = predict_ensemble(models, Xtr + 0.5 * np.random.randn(*Xtr.shape), device)
        print(unc.mean())

    elif method == "ensemble_no_sigma":
        models = []
        for m in range(ensemble_members):
            set_seed(seed + m + 12345)
            model = MLP_MuOnly(Xtr.shape[1], hidden=hidden, activation=activation).to(device)
            # train_gaussian(model, Xtr, Ytr, device, epochs, batch_size, lr)
            for ep in range(1, epochs+1):
                ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                total = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    mu = model(xb)
                    loss = torch.nn.functional.mse_loss(mu, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item() * xb.size(0)
                if ep % max(epochs//5,1) == 0:
                    print(f"  [Ensemble member {m+1}] ep{ep:03d} loss={total/len(ds):.4f}")
            models.append(model)
        def eval_ensemble_no_sigma_test():
            mu, unc = predict_ensemble_no_sigma(models, Xte, device)
            print("spearman_r between errors and epistemic uncertainty:", spearman_r(np.abs(Yte - mu), unc))
            return empty_test_metrics(len(Yte))
        metrics = timed_test_eval(device, eval_ensemble_no_sigma_test)
        # metrics = evaluate_gaussian(Yte, mu, None, ysc, epistemic_uncertainty=unc)
        # print("rmse:", float(root_mean_squared_error(y_scaler.inverse_transform(Yte), y_scaler.inverse_transform(mu))))
        members = ensemble_members

    elif method == "swa":
        base = MLP_Gauss(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=0.0).to(device)
        swa_model = train_gaussian_swa(base, Xtr, Ytr, device, epochs, batch_size, lr, swa_start=swa_start)
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(Yte, predict_gaussian(swa_model, Xte, device), ysc, ece_bins=ece_bins)
        )
        members = 1

    elif method == "swag":
        model = MLP_Gauss(Xtr.shape[1], hidden=hidden, activation=activation, dropout_p=0.0).to(device)
        ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        snaps = []
        burn_ep = int(swag_burn * epochs)
        for ep in range(1, epochs+1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, sigma = model(xb)
                loss = gaussian_nll_torch(mu, sigma, yb).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep >= burn_ep and (ep - burn_ep) % swag_snap_every == 0:
                swag_collect_snapshots(model, snaps)
        mean, var = swag_stats(snaps if len(snaps) > 0 else [flatten_params(model).numpy()])
        metrics = timed_test_eval(
            device,
            lambda: evaluate_gaussian_outputs(
                Yte, predict_swag(model, mean, var, Xte, device, K=swag_samples), ysc, ece_bins=ece_bins
            )
        )
        members = swag_samples

    elif method == "laplace_ll":
        net = MLP_MuOnly(Xtr.shape[1], hidden=hidden, activation=activation).to(device)
        ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        for ep in range(1, epochs+1):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                mu = net(xb)
                loss = ((yb - mu)**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)
            if ep % max(epochs//5,1) == 0:
                print(f"  [Laplace-LL] ep{ep:03d} loss={total/len(ds):.4f}")
        net.eval()
        with torch.no_grad():
            Phi_tr = net.features(torch.from_numpy(Xtr).float().to(device)).cpu().numpy()
        y_tr = Ytr.astype(np.float64)
        alpha, sigma2, w_mean, S = fit_bayesian_linear(Phi_tr, y_tr)
        def eval_laplace_test():
            with torch.no_grad():
                Phi_te = net.features(torch.from_numpy(Xte).float().to(device)).cpu().numpy()
            return evaluate_gaussian_outputs(
                Yte, predict_bayesian_linear(Phi_te, w_mean, S, sigma2), ysc, ece_bins=ece_bins
            )
        metrics = timed_test_eval(device, eval_laplace_test)
        members = 1

    else:
        raise ValueError(f"Unknown method: {method}")

    return Metrics(
        dataset=dataset, method=method, split=split, seed=seed, members=members,
        rmse=metrics["rmse"], mae=metrics["mae"], nll=metrics["nll"],
        r2=metrics["r2"], evs=metrics["evs"], mape=metrics["mape"], smape=metrics["smape"],
        rmsle=metrics["rmsle"], pearson_r=metrics["pearson_r"], spearman_r=metrics["spearman_r"], crps=metrics["crps"],
        ece=metrics["ece"], test_time_sec=metrics["test_time_sec"],
        n=metrics["n"]
    )

# ==============================================================
# ====== CSV + Summary =========================================
# ==============================================================

import os, csv
from dataclasses import asdict, is_dataclass

def write_csv(path, rows):
    if not rows:
        return

    # Convert to dicts
    def to_dict(r):
        return asdict(r) if is_dataclass(r) else dict(r)

    dict_rows = [to_dict(r) for r in rows]
    fieldnames = list(dict_rows[0].keys())

    # write header if file doesn't exist or is empty
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new_file:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        missing = [name for name in fieldnames if name not in existing_fieldnames]
        if missing:
            fieldnames = existing_fieldnames + missing
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in existing_rows:
                    w.writerow({name: r.get(name, "") for name in fieldnames})
            print(f"[ok] expanded CSV header in {path}: added {', '.join(missing)}")
        else:
            fieldnames = existing_fieldnames

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in dict_rows:
            w.writerow({name: r.get(name, "") for name in fieldnames})

    print(f"[ok] appended {len(rows)} rows to {path}")

def summarize(rows):
    by = {}
    for r in rows:
        key = f"{r.dataset}/{r.method}"
        by.setdefault(key, []).append(r)

    print("\n=== Summary (by dataset/method) ===")
    for k, vals in by.items():
        arr = lambda a: np.array([getattr(v, a) for v in vals], float)
        ddof = 1 if len(vals) > 1 else 0
        rmse = arr("rmse"); mae = arr("mae"); nll = arr("nll")
        r2 = arr("r2"); evs = arr("evs"); mape = arr("mape"); smape = arr("smape")
        rmsle = arr("rmsle"); pr = arr("pearson_r"); crps = arr("crps")
        ece = arr("ece"); test_time = arr("test_time_sec")
        ece_mean = np.nanmean(ece) if np.any(np.isfinite(ece)) else float("nan")
        members = int(np.mean(arr("members")))
        n_total = int(sum(v.n for v in vals))
        print(
            f"{k:>18} (M≈{members}) | RMSE {rmse.mean():.4f}±{rmse.std(ddof=ddof):.4f} | "
            f"MAE {mae.mean():.4f} | NLL {nll.mean():.4f} | "
            f"R2 {r2.mean():.4f} EVS {evs.mean():.4f} | "
            f"MAPE {mape.mean():.2f}% sMAPE {smape.mean():.2f}% | "
            f"RMSLE {np.nanmean(rmsle):.4f} r {pr.mean():.3f} | "
            f"CRPS {crps.mean():.4f} ECE {ece_mean:.4f} | "
            f"test {test_time.sum():.3f}s total | n={n_total}"
        )

# ==============================================================
# ====== CLI ====================================================
# ==============================================================
def main():
    ap = argparse.ArgumentParser(description="Unified UCI Runner: DER + Gaussian-family baselines")
    ap.add_argument("--method", type=str, default="der",
                    choices=["der", "der_elbo","corr_der_fix_noise", "corr_der_variable_noise", "gaussian","mc_dropout","bbb","ensemble","ensemble_no_sigma","swa","swag","laplace_ll"])
    ap.add_argument("--dataset", type=str, default="concrete")
    ap.add_argument("--splits", type=str, default="0", help="'all' or comma list, e.g. 0,1,2")
    ap.add_argument("--seeds", type=str, default="0", help="comma list, e.g. 0,1,2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="512,512")
    ap.add_argument("--activation", type=str, default="relu", choices=["relu","gelu","tanh"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="fraction of the official training split used for early-stopping validation")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop after this many validation checks without lower RMSE; 0 disables early stopping")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="minimum validation RMSE improvement required to reset early-stopping patience")
    ap.add_argument("--early-stop-check-every", type=int, default=1,
                    help="evaluate validation RMSE every N epochs while early stopping is enabled")
    # DER
    ap.add_argument("--lamb", type=float, default=1e-2, help="DER regularizer weight")
    ap.add_argument("--clue-kl-weight", type=float, default=0.01,
                    help="KL regularizer weight for corr_der_fix_noise/corr_der_variable_noise")
    ap.add_argument("--bbb-kl-weight", type=float, default=0.01,
                    help="KL weight for Bayes-by-Backprop variational layers")
    ap.add_argument("--bbb-prior-sigma", type=float, default=1.0,
                    help="zero-mean Gaussian prior std for Bayes-by-Backprop weights")
    ap.add_argument("--crps-samples", type=int, default=256, help="MC samples for CRPS (DER)")
    ap.add_argument("--ece-bins", type=int, default=10, help="number of quantile bins for regression ECE")
    # MC Dropout / Ensemble
    ap.add_argument("--dropout-p", type=float, default=0.1)
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--members", type=int, default=5)
    # SWA / SWAG
    ap.add_argument("--swa-start", type=float, default=0.75)
    ap.add_argument("--swag-burn", type=float, default=0.5)
    ap.add_argument("--swag-snap-every", type=int, default=5)
    ap.add_argument("--swag-samples", type=int, default=30)
    # Logging
    ap.add_argument("--csv", type=str, default="", help="optional CSV path to append rows")
    args = ap.parse_args()

    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    splits = list(range(10)) if args.splits == "all" else [int(s) for s in args.splits.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for split in splits:
        for seed in seeds:
            m = run_once(
                method=args.method, dataset=args.dataset, split=split, seed=seed, device=args.device,
                hidden=hidden, activation=args.activation,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                lamb=args.lamb, clue_kl_weight=args.clue_kl_weight,
                bbb_kl_weight=args.bbb_kl_weight, bbb_prior_sigma=args.bbb_prior_sigma,
                dropout_p=args.dropout_p, mc_samples=args.mc_samples, ensemble_members=args.members,
                swa_start=args.swa_start, swag_burn=args.swag_burn, swag_snap_every=args.swag_snap_every,
                swag_samples=args.swag_samples, crps_samples=args.crps_samples, ece_bins=args.ece_bins,
                val_ratio=args.val_ratio,
                early_stop_patience=args.early_stop_patience,
                min_delta=args.min_delta,
                early_stop_check_every=args.early_stop_check_every
            )
            print(
                f"→ [{m.dataset} split={m.split} seed={m.seed}] {m.method} (M={m.members}) | "
                f"RMSE={m.rmse:.4f} MAE={m.mae:.4f} NLL={m.nll:.4f} | "
                f"R2={m.r2:.4f} EVS={m.evs:.4f} | MAPE={m.mape:.2f}% sMAPE={m.smape:.2f}% | "
                f"RMSLE={m.rmsle:.4f} r={m.pearson_r:.3f} sr={m.spearman_r:.3f} | "
                f"CRPS={m.crps:.4f} ECE={m.ece:.4f} test={m.test_time_sec:.3f}s"
            )
            rows.append(m)

    if args.csv and rows: write_csv(args.csv, rows)
    if rows: summarize(rows)

if __name__ == "__main__":
    main()
