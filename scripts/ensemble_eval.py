#!/usr/bin/env python3
"""
Ensemble evaluation for OGBG-MOLHIV checkpoints.

Loads multiple checkpoints (typically from different seeds) and evaluates:
- each single model ROC-AUC (valid/test)
- ensemble ROC-AUC by averaging logits (default) or probabilities

This is intentionally a small script so it's easy to audit and run locally.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.loader import DataLoader


@dataclass
class ModelConfig:
    hidden: int
    num_layers: int
    head_layers: int
    dropout: float
    pool: str
    virtual_node: bool


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        assert layers >= 1
        mods = []
        dim = in_dim
        for _ in range(layers - 1):
            mods += [torch.nn.Linear(dim, hidden), torch.nn.ReLU()]
            if dropout and dropout > 0:
                mods += [torch.nn.Dropout(dropout)]
            dim = hidden
        mods.append(torch.nn.Linear(dim, out_dim))
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CatSumEncoder(torch.nn.Module):
    def __init__(self, cardinals: list[int], hidden: int):
        super().__init__()
        self.embs = torch.nn.ModuleList([torch.nn.Embedding(int(c), hidden) for c in cardinals])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        out = 0
        for j, emb in enumerate(self.embs):
            xj = x[:, j].clamp(min=0, max=emb.num_embeddings - 1)
            out = out + emb(xj)
        return out


class GINv2Model(torch.nn.Module):
    """
    Minimal GINv2 model compatible with checkpoints from src/gin_v2_atom_bond_vn.py
    (cat encoders + optional virtual node + mean/sum pooling; head MLP).
    """

    def __init__(
        self,
        node_cardinals: list[int],
        edge_cardinals: list[int] | None,
        cfg: ModelConfig,
    ):
        super().__init__()
        from torch_geometric.nn import GINConv, GINEConv, GlobalAttention, global_add_pool, global_mean_pool

        self.node_enc = CatSumEncoder(node_cardinals, cfg.hidden)
        self.use_edge_attr = edge_cardinals is not None and len(edge_cardinals) > 0
        self.edge_enc = CatSumEncoder(edge_cardinals, cfg.hidden) if self.use_edge_attr else None

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(cfg.num_layers):
            nn = torch.nn.Sequential(
                torch.nn.Linear(cfg.hidden, cfg.hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.hidden, cfg.hidden),
            )
            if self.use_edge_attr:
                self.convs.append(GINEConv(nn, train_eps=True, edge_dim=cfg.hidden))
            else:
                self.convs.append(GINConv(nn, train_eps=True))
            self.bns.append(torch.nn.BatchNorm1d(cfg.hidden))

        self.pool = cfg.pool
        self.dropout = float(cfg.dropout)
        self.head = MLP(cfg.hidden, cfg.hidden, 1, layers=cfg.head_layers, dropout=self.dropout)

        self.attn_pool = None
        if self.pool == "attn":
            gate_nn = torch.nn.Sequential(
                torch.nn.Linear(cfg.hidden, cfg.hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.hidden, 1),
            )
            self.attn_pool = GlobalAttention(gate_nn=gate_nn)

        self._global_mean_pool = global_mean_pool
        self._global_add_pool = global_add_pool

        self.virtual_node = bool(cfg.virtual_node)
        if self.virtual_node:
            self.vn_emb = torch.nn.Embedding(1, cfg.hidden)
            self.vn_mlp = torch.nn.Sequential(
                torch.nn.Linear(cfg.hidden, cfg.hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.hidden, cfg.hidden),
            )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return self._global_mean_pool(x, batch)
        if self.pool == "sum":
            return self._global_add_pool(x, batch)
        if self.pool == "attn":
            assert self.attn_pool is not None
            return self.attn_pool(x, batch)
        raise ValueError(self.pool)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, batch: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        x = self.node_enc(x)
        if self.edge_enc is not None and edge_attr is not None:
            edge_attr = self.edge_enc(edge_attr)
        else:
            edge_attr = None

        if self.virtual_node:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            vn = self.vn_emb.weight[0].expand(num_graphs, -1)

        for conv, bn in zip(self.convs, self.bns):
            if self.virtual_node:
                x = x + vn[batch]

            if self.use_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.virtual_node:
                g = self._pool(x, batch)
                vn = vn + self.vn_mlp(g)
                if self.dropout > 0:
                    vn = F.dropout(vn, p=self.dropout, training=self.training)

        g = self._pool(x, batch)
        logits = self.head(g).view(-1)
        return logits


@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    y_true = []
    y_logit = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_logit.append(logits.detach().cpu().float())
        y_true.append(batch.y.view(-1).detach().cpu().float())
    return torch.cat(y_true, dim=0), torch.cat(y_logit, dim=0)


def rocauc(evaluator: Evaluator, y_true: torch.Tensor, y_prob: torch.Tensor) -> float:
    # evaluator expects shape [N, 1]
    out = evaluator.eval({"y_true": y_true.view(-1, 1), "y_pred": y_prob.view(-1, 1)})
    return float(out["rocauc"])


def load_ckpt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    ap = argparse.ArgumentParser(description="Ensemble eval for ogbg-molhiv checkpoints")
    ap.add_argument("--root", default="data/ogbg-molhiv")
    ap.add_argument("--split", choices=["valid", "test", "both"], default="both")
    ap.add_argument("--ckpts", nargs="*", default=None, help="explicit list of ckpt paths")
    ap.add_argument("--ckpt_glob", default=None, help="glob pattern for ckpts")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--avg", choices=["logits", "probs"], default="logits")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.ckpts is None or len(args.ckpts) == 0:
        if not args.ckpt_glob:
            raise SystemExit("Provide --ckpts ... or --ckpt_glob 'models/*.pt'")
        ckpts = sorted(glob.glob(args.ckpt_glob))
    else:
        ckpts = list(args.ckpts)
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for this script (matches project assumptions).")

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=args.root)
    split = ds.get_idx_split()
    evaluator = Evaluator(name="ogbg-molhiv")

    def make_loader(which: str) -> DataLoader:
        idx = split[which]
        return DataLoader(ds[idx], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    loaders = {}
    if args.split in ("valid", "both"):
        loaders["valid"] = make_loader("valid")
    if args.split in ("test", "both"):
        loaders["test"] = make_loader("test")

    # Build a model per checkpoint; configs come from ckpt["args"].
    per_model = {}
    logits_sum = {k: None for k in loaders.keys()}
    probs_sum = {k: None for k in loaders.keys()}

    for p in ckpts:
        ck = load_ckpt(p)
        ck_args = ck.get("args", {})
        node_card = ck.get("node_cardinals")
        edge_card = ck.get("edge_cardinals")
        if node_card is None:
            raise SystemExit(f"Checkpoint missing node_cardinals: {p}")

        cfg = ModelConfig(
            hidden=int(ck_args.get("hidden", 256)),
            num_layers=int(ck_args.get("num_layers", 5)),
            head_layers=int(ck_args.get("head_layers", 2)),
            dropout=float(ck_args.get("dropout", 0.1)),
            pool=str(ck_args.get("pool", "mean")),
            virtual_node=bool(ck_args.get("virtual_node", True)),
        )
        model = GINv2Model(node_card, edge_card, cfg).to(device)
        model.load_state_dict(ck["model"], strict=True)

        name = os.path.basename(p)
        per_model[name] = {"path": p, "cfg": cfg.__dict__}

        for split_name, loader in loaders.items():
            y_true, y_logit = predict_logits(model, loader, device)
            y_prob = torch.sigmoid(y_logit)
            per_model[name][split_name] = {
                "rocauc": rocauc(evaluator, y_true, y_prob),
            }
            if logits_sum[split_name] is None:
                logits_sum[split_name] = y_logit.clone()
                probs_sum[split_name] = y_prob.clone()
            else:
                logits_sum[split_name] += y_logit
                probs_sum[split_name] += y_prob

    out = {"avg": args.avg, "ckpts": ckpts, "per_model": per_model, "ensemble": {}}
    for split_name, loader in loaders.items():
        # reuse y_true from last model computed for this split (identical ordering)
        # easiest: recompute from one loader pass without model, but batch.y is already in ds.
        y_true = torch.cat([b.y.view(-1).cpu().float() for b in loader], dim=0)

        if args.avg == "logits":
            y_prob_ens = torch.sigmoid(logits_sum[split_name] / len(ckpts))
        else:
            y_prob_ens = probs_sum[split_name] / len(ckpts)
        out["ensemble"][split_name] = {"rocauc": rocauc(evaluator, y_true, y_prob_ens)}

    # Print summary
    print(f"Loaded {len(ckpts)} checkpoints")
    for split_name in loaders.keys():
        print(f"Ensemble ({args.avg}) {split_name}: rocauc={out['ensemble'][split_name]['rocauc']:.6f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
