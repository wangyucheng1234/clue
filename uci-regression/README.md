#  UCI regression

## Setup and commands

Use Python 3.11 and run these commands inside this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Retrain the full main experiment; use cpu instead of cuda if needed.
bash reproduce.sh train cuda
```