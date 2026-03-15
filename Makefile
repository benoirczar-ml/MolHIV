.PHONY: help download results graphmlp gin ginv2_champ

help:
	@echo "Targets:"
	@echo "  download       - download/preprocess ogbg-molhiv into data/"
	@echo "  graphmlp       - run GraphMLP baseline (writes runs/ + models/)"
	@echo "  gin            - run GIN baseline (Linear encoders)"
	@echo "  ginv2_champ    - run current champ (cat encoders + virtual node)"
	@echo "  results        - generate a small results table from runs/*.json"

download:
	mamba run -n GraphLink python scripts/download_ogbg_molhiv.py --root data/ogbg-molhiv

graphmlp:
	mamba run -n GraphLink python src/graphmlp_baseline.py \
	  --root data/ogbg-molhiv \
	  --epochs 50 --batch_size 256 \
	  --hidden 256 --num_layers 3 --head_layers 2 \
	  --dropout 0.1 --pool mean --lr 0.001 --weight_decay 0.0 \
	  --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
	  --run_tag molhiv_graphmlp_base \
	  | tee runs/molhiv_graphmlp_base.log

gin:
	mamba run -n GraphLink python src/gin_baseline.py \
	  --root data/ogbg-molhiv \
	  --epochs 50 --batch_size 256 \
	  --hidden 128 --num_layers 5 --head_layers 2 \
	  --dropout 0.1 --pool mean --lr 0.001 --weight_decay 0.0 \
	  --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
	  --run_tag molhiv_gin_base_h128_l5 \
	  | tee runs/molhiv_gin_base_h128_l5.log

ginv2_champ:
	mamba run -n GraphLink python src/gin_v2_atom_bond_vn.py \
	  --root data/ogbg-molhiv \
	  --epochs 50 --batch_size 256 \
	  --hidden 256 --num_layers 5 --head_layers 2 \
	  --dropout 0.1 --pool mean --lr 0.001 --weight_decay 0.0001 \
	  --virtual_node \
	  --eval_every 1 --save_every 1 --num_workers 4 --log_vram \
	  --run_tag molhiv_ginv2_cat_vn_h256_l5_scanall \
	  | tee runs/molhiv_ginv2_cat_vn_h256_l5_scanall.log

results:
	mamba run -n GraphLink python scripts/summarize_runs.py --runs_dir runs --out_md reports/RESULTS_GENERATED.md

