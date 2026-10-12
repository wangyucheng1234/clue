#!/usr/bin/env python3
"""
aggregate_results.py — aggregate UCI regression results to publication tables.

Input CSV must have columns like:
  dataset, split, seed, method, rmse, mae, nll, r2, evs, mape, smape, rmsle,
  pearson_r, crps, ece, test_time_sec, n

Outputs:
  - Markdown table (stdout and --md)
  - LaTeX table (--tex)

Usage examples:
  python aggregate_results.py --csv iavb_all_results.csv --metric nll --md out.md --tex out.tex
  python aggregate_results.py --csv iavb_all_results.csv --metrics rmse,nll,crps,ece,test_time_sec --group-by dataset,method
"""

import argparse, math, sys, os
import numpy as np
import pandas as pd

# -------- helpers --------

LOWER_BETTER = {"rmse","mae","nll","mape","smape","rmsle","crps","ece","test_time_sec"}
HIGHER_BETTER = {"r2","evs","pearson_r"}

def mean_se(x):
    x = np.asarray(x, float)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1)/np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, se

def mean_ci(x, alpha=0.05):
    x = np.asarray(x, float)
    m = float(np.mean(x))
    if len(x) < 2:
        return m, m, m
    se = float(np.std(x, ddof=1)/np.sqrt(len(x)))
    # normal approx
    z = 1.959963984540054 # 95%
    lo, hi = m - z*se, m + z*se
    return m, lo, hi

def fmt_mean_se(m, se, digits=4):
    return f"{m:.{digits}f}±{se:.{digits}f}"

def bold(s, do_bold):
    return f"**{s}**" if do_bold else s

def pick_best(values, metric):
    # values: list of floats; return index of best according to metric
    if metric in HIGHER_BETTER:
        return int(np.nanargmax(values))
    # default lower is better
    return int(np.nanargmin(values))

# -------- main aggregation --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input results CSV (e.g., iavb_all_results.csv)")
    ap.add_argument("--group-by", default="dataset,method",
                    help="keys to define a configuration before aggregating over split/seed")
    ap.add_argument("--metrics", default="nll,rmse,mae,crps",
                    help="comma list of metrics to show in the table")
    ap.add_argument("--digits", type=int, default=4, help="decimal places for means/SEs")
    ap.add_argument("--md", default="", help="optional path to write Markdown table")
    ap.add_argument("--tex", default="", help="optional path to write LaTeX table")
    ap.add_argument("--metric-for-bold", default="nll",
                    help="which metric decides bolding (best per dataset)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[error] CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    needed_cols = {"dataset","split","seed"}
    if not needed_cols.issubset(df.columns):
        sys.exit(f"[error] CSV missing columns {needed_cols - set(df.columns)}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in df.columns:
            sys.exit(f"[error] metric '{m}' not found. Available: {list(df.columns)}")

    group_keys = [g.strip() for g in args.group_by.split(",") if g.strip()]
    for g in group_keys:
        if g not in df.columns:
            sys.exit(f"[error] group key '{g}' not in CSV columns.")

    # Aggregate over everything NOT in group_keys (i.e., over split, seed, hyperparams, etc.)
    agg_rows = []
    for keys, g in df.groupby(group_keys, dropna=False):
        row = {k: v for k, v in zip(group_keys, (keys if isinstance(keys, tuple) else (keys,)))}
        for m in metrics:
            mu, se = mean_se(g[m].values)
            row[f"{m}_mean"] = mu
            row[f"{m}_se"] = se
            m_, lo, hi = mean_ci(g[m].values, alpha=0.05)
            row[f"{m}_lo95"] = lo
            row[f"{m}_hi95"] = hi
        row["runs"] = len(g)
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)

    # ---- Markdown: one table per metric, rows=datasets, cols=methods (if present) ----
    has_method = "method" in group_keys and "method" in df.columns
    md_blocks = []
    for m in metrics:
        md_lines = []
        md_lines.append(f"### {m.upper()} (mean±SE)")
        # Build pivot with strings (mean±se)
        pivot_vals = []
        datasets = sorted(agg["dataset"].unique().tolist())
        methods = sorted(agg["method"].unique().tolist()) if has_method else ["iavb"]
        # Precompute best per dataset for bolding
        best_idx = {}
        for ds in datasets:
            rowvals = []
            for me in methods:
                sel = agg[(agg["dataset"]==ds) & ((agg["method"]==me) if has_method else True)]
                if sel.empty:
                    rowvals.append(np.nan)
                else:
                    rowvals.append(sel.iloc[0][f"{m}_mean"])
            best_idx[ds] = pick_best(rowvals, m) if not all(np.isnan(rowvals)) else None

        # header
        header = "| Dataset | " + (" | ".join(methods)) + " |"
        sep = "|" + " --- |"*(2 if not has_method else (1+len(methods)))
        md_lines.append(header)
        md_lines.append(sep)

        for ds in datasets:
            cells = []
            for j, me in enumerate(methods):
                sel = agg[(agg["dataset"]==ds) & ((agg["method"]==me) if has_method else True)]
                if sel.empty:
                    cells.append("—")
                else:
                    r = sel.iloc[0]
                    s = fmt_mean_se(r[f"{m}_mean"], r[f"{m}_se"], digits=args.digits)
                    cells.append(bold(s, (best_idx[ds]==j)))
            md_lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        md_blocks.append("\n".join(md_lines))

    md_out = "\n\n".join(md_blocks)
    print(md_out)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md_out + "\n")

    # ---- LaTeX: one table (multi-metric) OR one per metric (choose one style)
    # Here: one table per metric (compact and journal-friendly)
    tex_blocks = []
    for m in metrics:
        datasets = sorted(agg["dataset"].unique().tolist())
        methods = sorted(agg["method"].unique().tolist()) if has_method else ["IAVB"]

        # precompute best indices
        best_idx = {}
        for ds in datasets:
            rowvals = []
            for me in methods:
                sel = agg[(agg["dataset"]==ds) & ((agg["method"]==me) if has_method else True)]
                rowvals.append(sel.iloc[0][f"{m}_mean"] if not sel.empty else np.nan)
            best_idx[ds] = pick_best(rowvals, m) if not all(np.isnan(rowvals)) else None

        lines = []
        lines.append("\\begin{tabular}{l " + " ".join(["r"]*len(methods)) + "}")
        lines.append("\\toprule")
        lines.append("Dataset & " + " & ".join(methods) + " \\\\")
        lines.append("\\midrule")
        for ds in datasets:
            cells = []
            for j, me in enumerate(methods):
                sel = agg[(agg["dataset"]==ds) & ((agg["method"]==me) if has_method else True)]
                if sel.empty:
                    cells.append("--")
                else:
                    r = sel.iloc[0]
                    s = f"{r[f'{m}_mean']:.{args.digits}f}$\\pm${r[f'{m}_se']:.{args.digits}f}"
                    if best_idx[ds] == j:
                        s = "\\textbf{" + s + "}"
                    cells.append(s)
            lines.append(ds + " & " + " & ".join(cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("")  # spacer
        tex_blocks.append("\n".join(lines))

    if args.tex:
        with open(args.tex, "w") as f:
            f.write("\n\n".join(tex_blocks) + "\n")

if __name__ == "__main__":
    main()
