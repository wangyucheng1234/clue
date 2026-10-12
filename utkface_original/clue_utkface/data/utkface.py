from __future__ import annotations

import json
import os
import sqlite3
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class StandardScaler1D:
    def __init__(self) -> None:
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, y: np.ndarray) -> None:
        arr = np.asarray(y, dtype=np.float32).reshape(-1)
        self.mean_ = float(arr.mean())
        self.std_ = float(arr.std() + 1e-8)

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler1D must be fit before calling transform().")
        arr = np.asarray(y, dtype=np.float32)
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler1D must be fit before calling inverse_transform().")
        arr = np.asarray(y, dtype=np.float32)
        return arr * self.std_ + self.mean_


@dataclass(frozen=True)
class UTKFaceSample:
    path: str
    age: float


class UTKFaceAgeDataset(Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        root: str,
        transform=None,
        file_list: Optional[Sequence[str]] = None,
        target_scaler: Optional[StandardScaler1D] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_scaler = target_scaler

        if file_list is None:
            file_list = sorted(
                f for f in os.listdir(root)
                if f.lower().endswith(self.IMG_EXTS)
            )

        samples: List[UTKFaceSample] = []
        for fname in file_list:
            age = self._parse_age(fname)
            if age is None:
                continue
            samples.append(UTKFaceSample(path=os.path.join(root, fname), age=float(age)))

        if not samples:
            raise RuntimeError(f"No valid UTKFace samples found in {root}")

        self.samples = samples

    @staticmethod
    def _parse_age(fname: str) -> Optional[int]:
        try:
            return int(os.path.basename(fname).split("_")[0])
        except Exception:
            return None

    def get_raw_targets(self) -> np.ndarray:
        return np.array([s.age for s in self.samples], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        y = np.array([sample.age], dtype=np.float32)
        if self.target_scaler is not None:
            y = self.target_scaler.transform(y)
        target = torch.tensor(y, dtype=torch.float32)
        return image, target


class PackedUTKFaceAgeDataset(Dataset):
    def __init__(
        self,
        packed_path: str,
        transform=None,
        file_list: Optional[Sequence[str]] = None,
        target_scaler: Optional[StandardScaler1D] = None,
    ) -> None:
        self.packed_path = packed_path
        self.transform = transform
        self.target_scaler = target_scaler
        self._conn: Optional[sqlite3.Connection] = None

        with sqlite3.connect(f"file:{packed_path}?mode=ro", uri=True) as conn:
            if file_list is None:
                rows = conn.execute(
                    "SELECT filename, age FROM samples ORDER BY filename"
                ).fetchall()
            else:
                rows = []
                missing = []
                for fname in file_list:
                    row = conn.execute(
                        "SELECT filename, age FROM samples WHERE filename = ?",
                        (fname,),
                    ).fetchone()
                    if row is None:
                        missing.append(fname)
                    else:
                        rows.append(row)
                if missing:
                    preview = ", ".join(missing[:5])
                    raise ValueError(
                        f"Packed dataset {packed_path} is missing files from the split: {preview}"
                    )

        if not rows:
            raise RuntimeError(f"No valid UTKFace samples found in packed dataset {packed_path}")

        self.samples = [UTKFaceSample(path=str(filename), age=float(age)) for filename, age in rows]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.packed_path}?mode=ro", uri=True)
        return self._conn

    def get_raw_targets(self) -> np.ndarray:
        return np.array([s.age for s in self.samples], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        row = self._connect().execute(
            "SELECT image FROM samples WHERE filename = ?",
            (sample.path,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Sample {sample.path} was not found in packed dataset {self.packed_path}")

        image = Image.open(BytesIO(row[0])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        y = np.array([sample.age], dtype=np.float32)
        if self.target_scaler is not None:
            y = self.target_scaler.transform(y)
        target = torch.tensor(y, dtype=torch.float32)
        return image, target


def list_valid_utkface_files(root: str) -> List[str]:
    return [
        f for f in sorted(os.listdir(root))
        if f.lower().endswith(UTKFaceAgeDataset.IMG_EXTS)
        and UTKFaceAgeDataset._parse_age(f) is not None
    ]


def make_utkface_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf


def _shuffle_files(files: Sequence[str], rng: np.random.Generator) -> List[str]:
    if len(files) <= 1:
        return list(files)
    perm = rng.permutation(len(files))
    return [files[i] for i in perm]


def _split_shuffled_files(
    files: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    ensure_minimum: bool,
) -> Tuple[List[str], List[str], List[str]]:
    n_files = len(files)
    n_test = int(n_files * test_ratio)
    n_val = int(n_files * val_ratio)

    if ensure_minimum:
        if test_ratio > 0 and n_test == 0 and n_files >= 3:
            n_test = 1
        if val_ratio > 0 and n_val == 0 and (n_files - n_test) >= 2:
            n_val = 1
        while n_val + n_test >= n_files and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

    test_files = list(files[:n_test])
    val_start = n_test
    val_end = n_test + n_val
    val_files = list(files[val_start:val_end])
    train_files = list(files[val_end:])
    return train_files, val_files, test_files


def _rebalance_empty_splits(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    rng: np.random.Generator,
    want_val: bool,
    want_test: bool,
) -> Tuple[List[str], List[str], List[str]]:
    train_files = list(train_files)
    val_files = list(val_files)
    test_files = list(test_files)

    def move_one(dst: List[str]) -> None:
        if len(train_files) <= 1:
            return
        idx = int(rng.integers(len(train_files)))
        dst.append(train_files.pop(idx))

    if want_test and not test_files:
        move_one(test_files)
    if want_val and not val_files:
        move_one(val_files)
    return train_files, val_files, test_files


def build_split_manifest(
    root: str,
    train_files: Sequence[str],
    val_files: Sequence[str],
    test_files: Sequence[str],
    *,
    split_seed: Optional[int],
    val_ratio: float,
    test_ratio: float,
    stratify_by_age: bool = False,
    age_bin_size: int = 5,
    source: str = "generated",
) -> Dict[str, object]:
    return {
        "data_root": os.path.abspath(root),
        "source": source,
        "split_seed": split_seed,
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "stratify_by_age": bool(stratify_by_age),
        "age_bin_size": int(age_bin_size),
        "train_files": list(train_files),
        "val_files": list(val_files),
        "test_files": list(test_files),
    }


def validate_split_manifest(manifest: Dict[str, object], root: Optional[str] = None) -> None:
    required = {"train_files", "val_files"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Split manifest missing required keys: {sorted(missing)}")

    if "test_files" not in manifest:
        manifest["test_files"] = []

    all_lists: Dict[str, List[str]] = {}
    for key in ("train_files", "val_files", "test_files"):
        value = manifest[key]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"Split manifest key '{key}' must be a list of strings.")
        all_lists[key] = value

    train_set = set(all_lists["train_files"])
    val_set = set(all_lists["val_files"])
    test_set = set(all_lists["test_files"])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Split manifest contains overlapping files across splits.")

    if root is not None:
        available = set(list_valid_utkface_files(root))
        missing_files = sorted((train_set | val_set | test_set) - available)
        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise ValueError(f"Split manifest references files not found in {root}: {preview}")


def save_split_manifest(path: str, manifest: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_split_manifest(path: str, root: Optional[str] = None) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_split_manifest(manifest, root=root)
    return manifest


def make_train_val_test_split(
    root: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    stratify_by_age: bool = False,
    age_bin_size: int = 5,
) -> Tuple[List[str], List[str], List[str]]:
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")
    if stratify_by_age and age_bin_size <= 0:
        raise ValueError("age_bin_size must be > 0 when stratify_by_age is enabled.")

    files = list_valid_utkface_files(root)
    min_files = 3 if test_ratio > 0 else 2
    if len(files) < min_files:
        raise RuntimeError(f"Need at least {min_files} valid UTKFace images to make this split.")

    rng = np.random.default_rng(seed)

    if stratify_by_age:
        grouped_files: Dict[int, List[str]] = {}
        for fname in files:
            age = UTKFaceAgeDataset._parse_age(fname)
            if age is None:
                continue
            age_bin = age // age_bin_size
            grouped_files.setdefault(age_bin, []).append(fname)

        train_files: List[str] = []
        val_files: List[str] = []
        test_files: List[str] = []
        for age_bin in sorted(grouped_files):
            shuffled_group = _shuffle_files(grouped_files[age_bin], rng)
            group_train, group_val, group_test = _split_shuffled_files(
                shuffled_group,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                ensure_minimum=False,
            )
            train_files.extend(group_train)
            val_files.extend(group_val)
            test_files.extend(group_test)

        train_files, val_files, test_files = _rebalance_empty_splits(
            train_files,
            val_files,
            test_files,
            rng=rng,
            want_val=val_ratio > 0,
            want_test=test_ratio > 0,
        )
        train_files = _shuffle_files(train_files, rng)
        val_files = _shuffle_files(val_files, rng)
        test_files = _shuffle_files(test_files, rng)
        return train_files, val_files, test_files

    shuffled_files = _shuffle_files(files, rng)
    return _split_shuffled_files(
        shuffled_files,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        ensure_minimum=True,
    )


def create_split_manifest(
    root: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    stratify_by_age: bool = False,
    age_bin_size: int = 5,
) -> Dict[str, object]:
    train_files, val_files, test_files = make_train_val_test_split(
        root=root,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by_age=stratify_by_age,
        age_bin_size=age_bin_size,
    )
    return build_split_manifest(
        root,
        train_files,
        val_files,
        test_files,
        split_seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by_age=stratify_by_age,
        age_bin_size=age_bin_size,
        source="generated",
    )


def make_train_val_split(
    root: str,
    val_ratio: float = 0.1,
    seed: int = 0,
    stratify_by_age: bool = False,
    age_bin_size: int = 5,
) -> Tuple[List[str], List[str]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    train_files, val_files, _ = make_train_val_test_split(
        root=root,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        stratify_by_age=stratify_by_age,
        age_bin_size=age_bin_size,
    )
    return train_files, val_files
