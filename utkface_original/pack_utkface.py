#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_age(fname: str) -> int | None:
    try:
        return int(os.path.basename(fname).split("_")[0])
    except Exception:
        return None


def list_valid_utkface_files(root: str) -> list[str]:
    return [
        f for f in sorted(os.listdir(root))
        if f.lower().endswith(IMG_EXTS) and parse_age(f) is not None
    ]


def pack_utkface(data_root: str, output: str, overwrite: bool = False) -> None:
    root = Path(data_root)
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    files = list_valid_utkface_files(str(root))
    if not files:
        raise RuntimeError(f"No valid UTKFace images found in {root}")

    conn = sqlite3.connect(str(tmp_path))
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE samples (filename TEXT PRIMARY KEY, age REAL NOT NULL, image BLOB NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("format", "utkface_sqlite_v1"),
                ("source_root", str(root.resolve())),
                ("num_samples", str(len(files))),
            ],
        )

        for fname in tqdm(files, desc="packing UTKFace"):
            age = parse_age(fname)
            if age is None:
                continue
            with open(root / fname, "rb") as handle:
                image_bytes = handle.read()
            conn.execute(
                "INSERT INTO samples (filename, age, image) VALUES (?, ?, ?)",
                (fname, float(age), sqlite3.Binary(image_bytes)),
            )

        conn.execute("CREATE INDEX idx_samples_filename ON samples(filename)")
        conn.commit()
    except Exception:
        conn.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        conn.close()
        os.replace(tmp_path, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"packed {len(files)} samples into {output_path} ({size_mb:.1f} MiB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack UTKFace images into one SQLite file")
    parser.add_argument("--data-root", required=True, help="Directory containing UTKFace image files")
    parser.add_argument("--output", required=True, help="Output .sqlite path")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack_utkface(args.data_root, args.output, overwrite=args.overwrite)
