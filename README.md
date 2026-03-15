# MolHIV (OGBG-MOLHIV)

Local, reproducible baselines for **`ogbg-molhiv`** (binary graph classification).

- Metric: **ROC-AUC** (OGB official evaluator)
- Framework: PyTorch + PyG
- Hardware target: consumer GPU (8 GB VRAM is fine)

## Project Structure

```
MolHIV/
  README.md
  src/        # train/eval scripts (each run writes JSON + checkpoints)
  scripts/    # dataset download / utilities
  docs/       # reproducibility + workflow notes (for recruiters & future you)
  reports/    # small committed summaries (results tables, run registry)
  data/       # (gitignored) dataset cache
  runs/       # (gitignored) logs + per-run JSON history
  models/     # (gitignored) checkpoints (best/latest)
```

## Setup

This repo is designed to run in the existing `GraphLink` conda/mamba environment.

Optional: capture the current environment as an explicit spec (for debugging/repro):

```bash
mamba run -n GraphLink mamba list --explicit > env/conda-explicit-GraphLink.txt
```

## Data

Download + preprocess into this project:

```bash
mamba run -n GraphLink python scripts/download_ogbg_molhiv.py \
  --root data/ogbg-molhiv
```

## Training

All trainers follow the same convention:
- `--run_tag` controls file names so runs don’t overwrite each other.
- `models/*_best.pt`: best by **valid ROC-AUC**
- `models/*_latest.pt`: latest checkpoint
- `runs/*.log`: console log (via `tee`)
- `runs/*.json`: machine-readable history + meta

### Baseline: GraphMLP (no message passing)

```bash
mamba run -n GraphLink python src/graphmlp_baseline.py \
  --root data/ogbg-molhiv \
  --epochs 50 --batch_size 256 \
  --hidden 256 --num_layers 3 --head_layers 2 \
  --dropout 0.1 --pool mean --lr 0.001 --weight_decay 0.0 \
  --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
  --run_tag molhiv_graphmlp_base \
  | tee runs/molhiv_graphmlp_base.log
```

### Baseline: GIN/GINE (Linear encoders)

```bash
mamba run -n GraphLink python src/gin_baseline.py \
  --root data/ogbg-molhiv \
  --epochs 50 --batch_size 256 \
  --hidden 128 --num_layers 5 --head_layers 2 \
  --dropout 0.1 --pool mean --lr 0.001 --weight_decay 0.0 \
  --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
  --run_tag molhiv_gin_base_h128_l5 \
  | tee runs/molhiv_gin_base_h128_l5.log
```

### Stronger: GINv2 (Atom/Bond categorical encoders + Virtual Node)

```bash
mamba run -n GraphLink python src/gin_v2_atom_bond_vn.py \
  --root data/ogbg-molhiv \
  --epochs 50 --batch_size 256 \
  --hidden 256 --num_layers 5 --head_layers 2 \
  --dropout 0.1 --pool mean --lr 0.001 --weight_decay 0.0001 \
  --virtual_node \
  --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
  --run_tag molhiv_ginv2_cat_vn_h256_l5_scanall \
  | tee runs/molhiv_ginv2_cat_vn_h256_l5_scanall.log
```

## Results (Local Runs)

See `reports/RESULTS.md` for a compact table of tracked runs.

### Final v1 (Leaderboard-Style Summary)

Configuration: **GINv2 (Atom/Bond categorical encoders) + Virtual Node + pool=mean + OneCycleLR**.

Over **N=10 seeds** (`40..49`), epochs=40, selecting best checkpoint by **valid ROC-AUC**:
- **VALID** mean±std: `0.810100 ± 0.010479`
- **TEST@best-valid** mean±std: `0.762974 ± 0.022686`

Optional: ensemble the same 10 checkpoints (avg logits):
- Ensemble **VALID** ROC-AUC: `0.832424`
- Ensemble **TEST** ROC-AUC: `0.795141`

```bash
mamba run -n GraphLink python scripts/ensemble_eval.py \
  --root data/ogbg-molhiv \
  --ckpt_glob 'models/ginv2_ogbg-molhiv_finalv1_seed*_mean_onecycle_e40_best.pt' \
  --avg logits --split both
```

Reproduce:
```bash
cd /srv/work/MolHIV
git checkout exp/scheduler
for s in 40 41 42 43 44 45 46 47 48 49; do
  mamba run -n GraphLink python src/gin_v2_atom_bond_vn.py \
    --root data/ogbg-molhiv --epochs 40 --batch_size 256 \
    --hidden 256 --num_layers 5 --head_layers 2 --dropout 0.1 \
    --pool mean --lr 0.001 --weight_decay 0.0001 --virtual_node \
    --sched onecycle --onecycle_pct_start 0.1 --onecycle_div_factor 10 --onecycle_final_div_factor 100 \
    --seed $s --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
    --run_tag finalv1_seed${s}_mean_onecycle_e40 \
    | tee runs/finalv1_seed${s}_mean_onecycle_e40.log
done
```
