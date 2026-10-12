#!/usr/bin/env python3
"""Main UTKFace reproduction launcher; the scientific source files are unchanged.

All training settings below follow the recovered completed 20-epoch runs.
This launcher uses only the standard library; ML imports occur in the original
scripts when training, evaluation or timing is explicitly requested.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import sqlite3
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "splits/utkface_seed0.json"
METHODS = ("bbb", "der", "der_elbo", "mc_dropout", "fixed_noise", "variable_noise")
LABELS = {"bbb": "MFVI-BNN", "der": "EDR", "der_elbo": "EDR-ELBO",
          "mc_dropout": "MC Dropout", "fixed_noise": "CLUE", "variable_noise": "CLUE-H"}
REFERENCE_NAMES = {"BBB": "bbb", "DER": "der", "DER-ELBO": "der_elbo",
                   "MC-Dropout": "mc_dropout", "CLUE-fixed": "fixed_noise",
                   "CLUE-variable": "variable_noise"}
RUN_TYPES = {m: "clue_" + m if m.endswith("_noise") else m for m in METHODS}
METRICS = ("rmse", "mae", "nll", "crps", "ece")
COVERAGE = tuple("coverage_" + str(level) for level in (50, 80, 90, 95))


def run_dir(root, seed, method, standalone=False):
    parent = root if standalone else root / ("seed_" + str(seed))
    return parent / ("utkface_" + method)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_data(data_root, packed_data=None):
    manifest = json.loads(SPLIT.read_text())
    splits = [set(manifest[k]) for k in ("train_files", "val_files", "test_files")]
    if [len(s) for s in splits] != [18988, 2360, 2360]:
        raise ValueError("Unexpected fixed split counts")
    if any(a & b for i, a in enumerate(splits) for b in splits[i + 1:]):
        raise ValueError("Overlapping split filenames")
    names = set.union(*splits)
    with os.scandir(str(data_root)) as entries:
        available = {entry.name for entry in entries if entry.is_file()}
    missing = sorted(names - available)
    if missing:
        raise ValueError("Raw images missing from --data-root: " + ", ".join(missing[:5]))
    if packed_data is not None:
        with sqlite3.connect(packed_data.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            packed_names = {row[0] for row in conn.execute("SELECT filename FROM samples")}
        if not names.issubset(packed_names):
            raise ValueError("Packed cache is missing split filenames")
    return {"train": 18988, "validation": 2360, "test": 2360,
            "raw_filenames_present": len(names), "packed_filenames_checked": packed_data is not None}


def data_flags(args):
    flags = ["--data-root", str(args.data_root.resolve()), "--split-manifest", str(SPLIT)]
    if args.packed_data is not None:
        flags += ["--packed-data", str(args.packed_data.resolve())]
    return flags


def train_commands(args):
    for seed in range(3):
        for method in METHODS:
            clue = method.endswith("_noise")
            cmd = [sys.executable, str(HERE / ("train_clue.py" if clue else "train_baseline.py"))]
            cmd += data_flags(args)
            cmd += ["--output-dir", str(run_dir(args.output.resolve(), seed, method)),
                    "--variant" if clue else "--method", method,
                    "--seed", str(seed), "--split-seed", str(seed), "--architecture", "resnet18",
                    "--img-size", "224", "--batch-size", "512", "--epochs", "20", "--lr", "0.001",
                    "--weight-decay", "0" if method == "bbb" else "0.0001",
                    "--num-workers", str(args.workers), "--val-ratio", "0.1",
                    "--selection-metric", "val_rmse", "--pretrained", "--hidden-dim", "256",
                    "--device", args.device]
            if clue:
                cmd += ["--pretrain-epochs", "0", "--proj-dim", "256", "--dropout", "0",
                        "--n-anchors", "512", "--ell", "0.01", "--ell-mu", "0.01",
                        "--ell-sigma", "0.01", "--sigma-obs", "0.15", "--sigma-f2", "1",
                        "--prior-mean", "0", "--prior-var", "1", "--logsigma-prior-mean", "0",
                        "--logsigma-prior-var", "1", "--anchor-severity", "0.0001",
                        "--mc-samples", "512", "--kl-sigma-weight", "1"]
            else:
                cmd += ["--dropout", "0.1" if method == "mc_dropout" else "0",
                        "--lamb", "0.001", "--eval-samples", "30", "--train-samples", "1",
                        "--kl-weight", "1", "--bbb-prior-std", "1", "--bbb-init-rho", "-5"]
            yield cmd


def execute(command):
    print(" ".join(shlex.quote(str(x)) for x in command), flush=True)
    subprocess.run(command, cwd=str(HERE), check=True)


def new_output(path):
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise ValueError("Choose a new output directory; this path already exists: " + str(path))
    path.mkdir(parents=True)
    return path


def require_checkpoints(root, seeds, standalone=False):
    paths = {(seed, method): run_dir(root.resolve(), seed, method, standalone) / "best.pt"
             for seed in seeds for method in METHODS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise ValueError("Missing best.pt checkpoints: " + ", ".join(missing[:3]))
    return paths


def validated_groups(rows):
    groups = collections.defaultdict(list)
    seen = set()
    for row in rows:
        method = row.get("release_method") or REFERENCE_NAMES[row["model"]]
        key = (method, int(row["seed"]))
        if key in seen:
            raise ValueError("Duplicate run: " + str(key))
        seen.add(key)
        if row["split"] != "test" or int(row["n_samples"]) != 2360 or int(row["split_seed"]) != 0:
            raise ValueError("Unexpected held-out split: " + str(key))
        if row["run_type"] != RUN_TYPES[method] or row["architecture"] != "resnet18":
            raise ValueError("Unexpected method/backbone: " + str(key))
        for metric in METRICS + COVERAGE:
            if not math.isfinite(float(row[metric])):
                raise ValueError("Non-finite metric: " + str(key))
        groups[method].append(row)
    if seen != set(itertools.product(METHODS, range(3))):
        raise ValueError("Expected exactly six methods and seeds 0,1,2")
    return groups


def export(rows, output, historical_timing=None):
    groups = validated_groups(rows)
    summary = []
    lookup = {}
    for method in METHODS:
        for metric in METRICS + COVERAGE:
            values = [float(row[metric]) for row in groups[method]]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            lookup[(method, metric)] = (mean, sd)
            summary.append({"method": method, "label": LABELS[method], "metric": metric,
                            "n_runs": 3, "seeds": "0,1,2", "mean": mean, "sd": sd})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summary)
    md = ["# UTKFace main results", "", "Mean ± sample SD (ddof=1) across seeds 0, 1, 2 on the fixed test split.", ""]
    tex = ["% Requires booktabs. Performance and coverage report mean +/- sample SD across three seeds."]
    for metrics, heading in ((METRICS, "Performance"), (COVERAGE, "Coverage")):
        names = [m.upper() if not m.startswith("coverage_") else m.split("_")[1] + "%" for m in metrics]
        md += ["## " + heading, "", "| Method | " + " | ".join(names) + " |", "| " + " | ".join(["---"] * (len(metrics) + 1)) + " |"]
        tex += [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
                "Method & " + " & ".join(n.replace("%", r"\%") for n in names) + r" \\", r"\midrule"]
        for method in METHODS:
            cells = ["%.3f ± %.3f" % lookup[method, m] for m in metrics]
            md.append("| " + LABELS[method] + " | " + " | ".join(cells) + " |")
            tex.append(LABELS[method] + " & " + " & ".join("$%.3f\\pm%.3f$" % lookup[method, m] for m in metrics) + r" \\")
        md.append("")
        tex += [r"\bottomrule", r"\end{tabular}", r"\caption{UTKFace " + heading.lower() + r": mean $\pm$ sample standard deviation across three training seeds on a fixed split.}", r"\end{table}", ""]
    if historical_timing is not None:
        md += ["## Historical forward-only timing", "", "Separate single-run checkpoints; median of ten timed repeats after three warmups. These are not three-seed SD results.", "",
               "| Method | Median test time (s) |", "| --- | --- |"]
        for row in historical_timing:
            method = REFERENCE_NAMES[row["label"]]
            md.append("| " + LABELS[method] + " | %.3f |" % float(row["fair_prediction_time_sec_median"]))
    (output / "main.md").write_text("\n".join(md) + "\n")
    (output / "main.tex").write_text("\n".join(tex) + "\n")
    print("Exported 18 runs and 54 mean/SD groups to " + str(output))


def collect(root, output):
    rows = []
    for seed in range(3):
        for method in METHODS:
            path = run_dir(root.resolve(), seed, method) / "test_metrics.json"
            if not path.is_file():
                raise ValueError("Missing evaluation record: " + str(path))
            row = json.loads(path.read_text())
            if int(row["seed"]) != seed or row["run_type"] != RUN_TYPES[method]:
                raise ValueError("Record disagrees with directory: " + str(path))
            rows.append(dict(row, release_method=method))
    export(rows, output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command")
    reference = subs.add_parser("reference", help="Export bundled historical results without ML dependencies")
    reference.add_argument("--output", type=Path, default=HERE / "results/reference")
    for name in ("plan", "train", "evaluate", "timing", "check-data"):
        sub = subs.add_parser(name)
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--packed-data", type=Path)
        if name != "check-data":
            sub.add_argument("--device", default="cuda")
            sub.add_argument("--workers", type=int, default=8)
        if name in ("plan", "train"):
            sub.add_argument("--output", type=Path, default=HERE / "results/train")
        elif name in ("evaluate", "timing"):
            sub.add_argument("--checkpoint-root", type=Path, default=HERE / "results/train")
            sub.add_argument("--output", type=Path, default=HERE / ("results/evaluation" if name == "evaluate" else "results/timing"))
            if name == "timing":
                sub.add_argument("--seed", type=int, default=0, choices=range(3))
                sub.add_argument("--standalone", action="store_true", help="Use six utkface_<method> directories directly under checkpoint-root")
    sub = subs.add_parser("collect", help="Aggregate a complete 18-record evaluation directory")
    sub.add_argument("--evaluation-root", type=Path, default=HERE / "results/evaluation")
    sub.add_argument("--output", type=Path, default=HERE / "results/summary")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "reference":
        export(read_csv(HERE / "reference.csv"), args.output.resolve(), read_csv(HERE / "reference_timing.csv"))
    elif args.command == "collect":
        collect(args.evaluation_root, args.output)
    elif args.command == "plan":
        for cmd in train_commands(args):
            print(" ".join(shlex.quote(x) for x in cmd))
    elif args.command == "check-data":
        print(json.dumps(check_data(args.data_root, args.packed_data), indent=2))
    elif args.command == "train":
        check_data(args.data_root, args.packed_data)
        new_output(args.output)
        for cmd in train_commands(args):
            execute(cmd)
    elif args.command == "evaluate":
        check_data(args.data_root, args.packed_data)
        checkpoints = require_checkpoints(args.checkpoint_root, range(3))
        output = new_output(args.output)
        for (seed, method), checkpoint in checkpoints.items():
            dest = run_dir(output, seed, method)
            dest.mkdir(parents=True)
            execute([sys.executable, str(HERE / "eval_checkpoint.py"), "--checkpoint", str(checkpoint)]
                    + data_flags(args) + ["--split", "test", "--batch-size", "512",
                    "--num-workers", str(args.workers), "--device", args.device,
                    "--predictive-samples", "30", "--mc-metrics-samples", "256",
                    "--input-noise-std", "0", "--input-noise-seed", "0",
                    "--output", str(dest / "test_metrics.json"), "--predictions-csv", str(dest / "test_predictions.csv")])
        collect(output, output)
    elif args.command == "timing":
        check_data(args.data_root, args.packed_data)
        checkpoints = require_checkpoints(args.checkpoint_root, (args.seed,), args.standalone)
        output = new_output(args.output)
        cmd = [sys.executable, str(HERE / "fair_prediction_timing.py")] + data_flags(args)
        cmd += ["--split", "test", "--batch-size", "512", "--num-workers", str(args.workers),
                "--device", args.device, "--cache-device", "device", "--warmup-repeats", "3",
                "--repeats", "10", "--output-json", str(output / "timing.json"), "--output-csv", str(output / "timing.csv")]
        for (_, method), checkpoint in checkpoints.items():
            cmd += ["--checkpoint", LABELS[method] + "=" + str(checkpoint)]
        execute(cmd)


if __name__ == "__main__":
    main()
