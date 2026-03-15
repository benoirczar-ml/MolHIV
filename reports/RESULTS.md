# Results (Tracked Local Runs)

Metric: **ROC-AUC** (OGB evaluator). “Best” = best **valid** ROC-AUC.

| Run tag | Model | Best epoch | Best valid ROC-AUC | Test ROC-AUC @ best | Notes |
|---|---:|---:|---:|---:|---|
| `molhiv_graphmlp_base` | GraphMLP | 43 | 0.722677 | 0.664556 | no message passing |
| `molhiv_gin_base_h128_l5` | GIN/GINE (Linear enc) | 39 | 0.802246 | 0.775293 | baseline GIN |
| `molhiv_gin_h256_l5_b512_wd1e-4` | GIN/GINE (Linear enc) | 34 | 0.797463 | 0.752935 | larger hidden/batch, small WD |
| `molhiv_ginv2_cat_vn_h256_l5_attn` | GINv2 + VN + Attn pool | 35 | 0.826570 | 0.752911 | attn pool didn’t beat champ |
| `molhiv_ginv2_cat_vn_h256_l5_scanall` | **GINv2 + VN (champ)** | 50 | **0.834062** | 0.750260 | cat encoders + VN; best at last epoch |

## Current Champion (After Scheduler/Seed Sweep)

We treat this as the current best configuration for this repo:

| Run tag | Model | Best epoch | Best valid ROC-AUC | Test ROC-AUC @ best | Notes |
|---|---:|---:|---:|---:|---|
| `seed43_mean_onecycle_e60` | **GINv2 + VN + OneCycle (champ)** | 26 | **0.839782** | 0.784946 | pool=mean, lr=1e-3, OneCycle; peak early |

Seed variability (same config, 60 epochs):
- `seed42`: best val `0.812950` (epoch 19)
- `seed44`: best val `0.807347` (epoch 29)

## Final v1 (Leaderboard-Style)

Protocol:
- Configuration: **GINv2 + Virtual Node + pool=mean + OneCycle**
- Selection: best checkpoint by **valid ROC-AUC**
- Seeds: `40..49`
- Epochs: `40`

Aggregate over N=10:
- **VALID (best-valid ROC-AUC)**: mean `0.810100`, std `0.010479`
- **TEST@best-valid**: mean `0.762974`, std `0.022686`

Top-by-valid (from this final v1 batch):
- `seed48`: best val `0.821802` (epoch 21), test@best `0.772717`
- `seed40`: best val `0.820308` (epoch 24), test@best `0.784739`
- `seed46`: best val `0.818808` (epoch 17), test@best `0.753754`

### Ensemble (Same 10 Checkpoints)

If we ensemble the **10 best checkpoints** from `finalv1_seed40..49` by averaging logits:
- Ensemble VALID ROC-AUC: `0.832424`
- Ensemble TEST ROC-AUC: `0.795141`

Command:
`mamba run -n GraphLink python scripts/ensemble_eval.py --ckpt_glob 'models/ginv2_ogbg-molhiv_finalv1_seed*_mean_onecycle_e40_best.pt' --avg logits`
