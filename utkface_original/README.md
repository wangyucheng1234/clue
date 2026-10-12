# Main UTKFace experiment — original implementation

This standalone folder preserves the original implementation from
`IEDL/utkface/CorrEDL`. All 13 original Python source files and the fixed split
JSON are copied byte for byte. Models, losses, KL, predictive variances,
preprocessing and metric calculations are unchanged. The new `reproduce.py`
only supplies commands and aggregates saved metrics as mean ± sample SD.

There are **20 files**: the 13 original Python files, split manifest, launcher,
requirements, README, two historical CSVs and `.gitignore`. The package files
are required by the original imports and are retained without merging them.

```text
train_clue.py
train_baseline.py
eval_checkpoint.py
fair_prediction_timing.py
pack_utkface.py
clue_utkface/
  __init__.py
  data/{__init__,utkface}.py
  models/{__init__,backbones,der,probabilistic,resnet_clue}.py
splits/utkface_seed0.json
reproduce.py
requirements.txt
reference.csv
reference_timing.csv
README.md
.gitignore
```

Sweep/Slurm generators, duplicate launchers, old logs, alternative runs and Git
history are omitted. Images and checkpoints stay outside this small source
folder. No surrounding-repository imports are required after dependency setup.

## Setup and data

Use Python 3.11. Run commands inside this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Requires no ML dependencies or image data.
python reproduce.py reference
```

The requirements pin the current inspected direct dependencies; a fresh install
and full retraining with these pins are unverified. For the inspected CUDA 12.6
build, install `torch==2.9.1+cu126` and `torchvision==0.24.1+cu126` from
`https://download.pytorch.org/whl/cu126` before installing the requirements.
The original backbone loads ResNet-18 `IMAGENET1K_V1` pretrained weights,
requiring internet access on first use or an existing Torch weights cache.

Supply a flat directory containing all **23,708** filenames in the bundled
manifest. The fixed split has **18,988 train / 2,360 validation / 2,360 test**
images. Its old `data_root` metadata is descriptive; commands pass your actual
root explicitly and preserve the filename memberships. In this workspace the
raw data are in `IEDL/utkface/UTKFace` and the packed cache is `IEDL/utkface/sq`.

```bash
python reproduce.py check-data --data-root /path/to/UTKFace --packed-data /path/to/sq

# Recreate the optional cache with the unchanged original packer.
python pack_utkface.py --data-root /path/to/UTKFace --output /path/to/utkface.sqlite
```

The existing packed path is a SQLite **file**. Raw images are still needed for
the original split validation, even when using the cache. The cache stores raw
image bytes and uses the same implemented preprocessing. A filename check does
not establish historical image-content identity.

## Main training and evaluation

```bash
# Print the 18 commands without ML imports or data checks.
python reproduce.py plan --data-root /path/to/UTKFace --packed-data /path/to/sq

# Train six methods × seeds 0/1/2 with the saved 20-epoch settings.
python reproduce.py train --data-root /path/to/UTKFace --packed-data /path/to/sq --device cuda

# Evaluate every best.pt on the fixed test split and export mean ± SD tables.
python reproduce.py evaluate --data-root /path/to/UTKFace --packed-data /path/to/sq --device cuda
```

Training writes `results/train/seed_<seed>/utkface_<method>/`. The original
trainers save `config.json`, `split_manifest.json`, `history.json`, `last.pt`
and validation-selected `best.pt`. Evaluation writes a separate
`results/evaluation/` tree, metrics, prediction CSVs, `runs.csv`, `summary.csv`,
`main.md` and `main.tex`. Summary CSVs retain full precision for performance
and coverage; displayed mean ± SD values use three decimals and `ddof=1`.

`--output /path/to/new-directory` selects another output location.
Training/evaluation/timing require a new directory to prevent accidental reuse.
`--workers 8` is the recovered setting; lower it if your machine requires it.
Use `--device cpu` for CPU execution. To re-export a complete evaluation tree:

```bash
python reproduce.py collect --evaluation-root results/evaluation --output results/summary
```

The protocol follows the recovered completed runs: pretrained ResNet-18,
224×224 images, 20 epochs, Adam learning rate 0.001, batch 512, best validation
RMSE, fixed split, and no CLUE pretraining. CLUE/CLUE-H use hidden/projector
dimensions 256, 512 anchors, RBF lengths 0.01, smoothing 0.0001 and unit priors.
Fixed noise has standard deviation 0.15. CLUE-H uses 512 likelihood samples.
EDR/EDR-ELBO use weight **0.001**. MC Dropout uses probability 0.1 and 30
predictive passes. BBB uses prior standard deviation 1, rho −5, one training
draw, 30 predictive passes and KL weight 1. Weight decay is 0.0001 except
BBB, which uses **0**. These saved settings differ from some manuscript prose;
the original implementation and actual run settings are retained.

## Historical results and checkpoints

`reference.csv` is an unchanged copy of the 18 completed 20-epoch test records
from `IEDL/utkface/CorrEDL/outputs/utkface_all_seeds_backup/test_results.csv`.
`python reproduce.py reference` exports these records to `results/reference/`.
All 24 printed performance means and SDs are recovered from these records.
The original paper description calls the table errors SE, but the printed
values are SD. This release labels them SD explicitly.

Checkpoint binaries are not duplicated in this small source folder. To replay
the recovered checkpoints, pass their root explicitly:

```bash
python reproduce.py evaluate \
  --checkpoint-root /path/to/utkface_all_seeds_backup \
  --data-root /path/to/UTKFace --packed-data /path/to/sq \
  --output results/backup_evaluation --device cuda
```

The backup contains the main 20-epoch runs. Its serialized paths still name
the current `utkface_all_seeds` tree, which mostly contains two-epoch runs.
The launcher uses the supplied checkpoint root and overrides data/split paths;
it does not follow those stale result paths. Supply `--packed-data` when
replaying historical checkpoints because the original evaluator can fall back
to the packed-data path embedded in their configs.

## Prediction timing

```bash
# Benchmark seed 0 of a new 18-run training tree.
python reproduce.py timing --data-root /path/to/UTKFace --packed-data /path/to/sq --device cuda

# Replay the historical benchmark's six separate single-run checkpoints.
python reproduce.py timing --standalone --checkpoint-root /path/to/original/outputs \
  --data-root /path/to/UTKFace --packed-data /path/to/sq \
  --output results/historical_timing --device cuda
```

The unchanged benchmark caches test batches on the device, uses three warmup
passes and ten timed repeats, and reports forward-only timing. BBB/MC Dropout
use 30 predictive passes; other methods use one. `reference_timing.csv` is the
original six-row benchmark CSV. Its medians match the six printed test times
but come from separate single-run checkpoints, not a three-seed average.
Timing values are kept separate from performance SDs.

The original training loops report ordinary epoch wall-clock times. The
manuscript's cached-GPU training-time protocol is not added to this minimal
release. Exact numerical retraining and fresh replay of historical checkpoints
remain unverified. Original seeding and stochastic evaluation behavior are
preserved; no deterministic-execution hooks or scientific corrections were added.

Packaging verification confirmed source-byte equality, Python syntax, local
import completeness, all 18 training commands against 450 saved configuration
fields, the fixed raw/packed filenames, and all 54 reference mean/SD summaries.
An isolated copy exports reference tables without the surrounding repository.
Launcher orchestration was checked with stub executables; this is not an ML
training or checkpoint-replay validation.
