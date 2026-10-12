#!/usr/bin/env bash
# Main UCI protocol; the scientific Python files are unchanged originals.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin=${PYTHON_BIN:-python}
mode=${1:-help}

usage() {
    cat <<'EOF'
Usage:
  bash reproduce.sh reference [output-directory]
  bash reproduce.sh plan [cpu|cuda|cuda:N] [output-directory]
  bash reproduce.sh train [cpu|cuda|cuda:N] [output-directory]

reference: aggregate the bundled historical runs (no training).
plan:      print the 90 training commands without importing dependencies.
train:     run the main experiment and aggregate its results.

PYTHON_BIN selects the Python executable. Relative output paths are resolved
inside this folder. Training requires an output directory that does not exist.
EOF
}

aggregate() {
    "$python_bin" aggregate_results.py --csv "$1" --metrics rmse,nll,ece \
        --digits 3 --md "$2/main.md" --tex "$2/main.tex"
}

case "$mode" in
    reference)
        if (( $# > 2 )); then usage >&2; exit 2; fi
        output=${2:-results/reference}
        mkdir -p -- "$output"
        aggregate reference.csv "$output"
        exit 0
        ;;
    train|plan)
        if (( $# > 3 )); then usage >&2; exit 2; fi
        device=${2:-cpu}
        output=${3:-results/main}
        if [[ ! "$device" =~ ^(cpu|cuda(:[0-9]+)?)$ ]]; then
            printf 'Unsupported device: %s\n' "$device" >&2
            exit 2
        fi
        ;;
    help|-h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [[ "$mode" == train ]]; then
    if [[ -e "$output" || -L "$output" ]]; then
        printf 'Choose a new output directory; this path already exists: %s\n' "$output" >&2
        exit 1
    fi
    mkdir -p -- "$(dirname -- "$output")"
    mkdir -- "$output"
fi

for dataset in concrete housing yacht energy airfoil; do
    for method in corr_der_fix_noise corr_der_variable_noise der der_elbo bbb ensemble; do
        patience=0
        if [[ "$method" == der || "$method" == der_elbo ]]; then patience=50; fi
        for seed in 0 1 2; do
            command=("$python_bin" run_uci_all.py
                --dataset "$dataset" --method "$method" --splits 0 --seeds "$seed"
                --epochs 5000 --batch-size 256 --lr 0.001 --hidden 512,512
                --activation relu --device "$device"
                --val-ratio 0.1 --early-stop-patience "$patience"
                --min-delta 0 --early-stop-check-every 1
                --lamb 0.01 --clue-kl-weight 0.01 --bbb-kl-weight 0.01
                --bbb-prior-sigma 1 --dropout-p 0.1 --mc-samples 20 --members 5
                --crps-samples 256 --ece-bins 10 --csv "$output/main.csv")
            if [[ "$mode" == plan ]]; then
                printf '%q ' "${command[@]}"
                printf '\n'
            else
                "${command[@]}"
            fi
        done
    done
done

if [[ "$mode" == train ]]; then aggregate "$output/main.csv" "$output"; fi
