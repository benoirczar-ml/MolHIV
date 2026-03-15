# Results (Tracked Local Runs)

Metric: **ROC-AUC** (OGB evaluator). “Best” = best **valid** ROC-AUC.

| Run tag | Model | Best epoch | Best valid ROC-AUC | Test ROC-AUC @ best | Notes |
|---|---:|---:|---:|---:|---|
| `molhiv_graphmlp_base` | GraphMLP | 43 | 0.722677 | 0.664556 | no message passing |
| `molhiv_gin_base_h128_l5` | GIN/GINE (Linear enc) | 39 | 0.802246 | 0.775293 | baseline GIN |
| `molhiv_gin_h256_l5_b512_wd1e-4` | GIN/GINE (Linear enc) | 34 | 0.797463 | 0.752935 | larger hidden/batch, small WD |
| `molhiv_ginv2_cat_vn_h256_l5_attn` | GINv2 + VN + Attn pool | 35 | 0.826570 | 0.752911 | attn pool didn’t beat champ |
| `molhiv_ginv2_cat_vn_h256_l5_scanall` | **GINv2 + VN (champ)** | 50 | **0.834062** | 0.750260 | cat encoders + VN; best at last epoch |

