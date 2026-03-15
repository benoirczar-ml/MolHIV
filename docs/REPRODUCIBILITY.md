# Reproducibility Notes

This project is intentionally “script-first”:
- each model is a standalone Python entrypoint in `src/`
- no hidden notebook state required for training/eval
- outputs are deterministic given `--seed` and the same environment

## Outputs (Standardized)

All trainers write:
- `models/*_best.pt`: checkpoint with the best **valid ROC-AUC**
- `models/*_latest.pt`: latest checkpoint
- `runs/*.log`: stdout log (use `tee`)
- `runs/*.json`: run meta + full per-epoch history (loss/time/rocauc/vram)

`data/`, `runs/`, `models/` are **gitignored** (local artifacts).

## Run Tags

Always pass `--run_tag` so artifacts are uniquely named and never overwrite.

Example:
- `--run_tag molhiv_ginv2_cat_vn_h256_l5_scanall`

## Metric (Official)

We use OGB’s evaluator:
- dataset: `ogbg-molhiv`
- metric: `rocauc`

## What Counts As “Best”

We select the best checkpoint by:
- maximizing **valid ROC-AUC**

We still log test ROC-AUC each eval for visibility, but we do not “select by test”.

