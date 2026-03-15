"""
GIN baseline for OGBG-MOLHIV (graph-level binary classification, ROC-AUC).

Encoder: stacked GINConv layers (no neighbor sampling; full message passing inside each batch)
Readout: global mean/sum pooling
Head: MLP -> logit
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, global_mean_pool


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunMeta:
    run_id: str
    run_tag: str
    best_epoch: int | None
    best_valid_rocauc: float | None
    best_path: str
    latest_path: str


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


class GINGraphClassifier(torch.nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        num_layers: int,
        dropout: float,
        pool: str,
        mlp_layers: int,
    ):
        super().__init__()
        self.node_enc = torch.nn.Linear(node_in, hidden)
        self.edge_enc = torch.nn.Linear(edge_in, hidden) if edge_in > 0 else None

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.use_edge_attr = edge_in > 0
        for _ in range(num_layers):
            # Use a plain Sequential so PyG can infer in_channels.
            nn = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
            )
            if self.use_edge_attr:
                self.convs.append(GINEConv(nn, train_eps=True, edge_dim=hidden))
            else:
                self.convs.append(GINConv(nn, train_eps=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden))

        self.pool = pool
        self.dropout = dropout
        self.head = MLP(hidden, hidden, 1, layers=mlp_layers, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, batch: torch.Tensor) -> torch.Tensor:
        x = self.node_enc(x.float())
        if self.edge_enc is not None and edge_attr is not None:
            edge_attr = self.edge_enc(edge_attr.float())
        else:
            edge_attr = None

        for conv, bn in zip(self.convs, self.bns):
            if self.use_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if self.dropout and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pool == "mean":
            g = global_mean_pool(x, batch)
        elif self.pool == "sum":
            g = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pool={self.pool!r}")

        logits = self.head(g).view(-1)
        return logits


@torch.no_grad()
def eval_split(model: torch.nn.Module, loader: DataLoader, evaluator: Evaluator, device: torch.device) -> float:
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(logits).view(-1, 1)
        y_pred.append(probs.detach().cpu())
        y_true.append(batch.y.view(-1, 1).detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    out = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return float(out["rocauc"])


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    ap = argparse.ArgumentParser(description="GIN baseline for ogbg-molhiv (ROC-AUC)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", choices=["mean", "sum"], default="mean")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--log_vram", action="store_true", help="Print peak VRAM per epoch.")
    args = ap.parse_args()

    proj = Path(__file__).resolve().parents[1]
    if args.root is None:
        args.root = str(proj / "data" / "ogbg-molhiv")
    if args.out_dir is None:
        args.out_dir = str(proj / "runs")
    if args.model_dir is None:
        args.model_dir = str(proj / "models")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    print("Loading dataset...", flush=True)
    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=args.root)
    split = ds.get_idx_split()
    evaluator = Evaluator(name="ogbg-molhiv")

    # DataLoader accepts lists of indices directly.
    train_loader = DataLoader(ds[split["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(ds[split["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds[split["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    node_in = int(ds[0].x.size(1))
    edge_in = 0 if ds[0].edge_attr is None else int(ds[0].edge_attr.size(1))

    model = GINGraphClassifier(
        node_in=node_in,
        edge_in=edge_in,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pool=args.pool,
        mlp_layers=args.head_layers,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag or run_id
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.model_dir, f"gin_ogbg-molhiv_{tag}_best.pt")
    latest_path = os.path.join(args.model_dir, f"gin_ogbg-molhiv_{tag}_latest.pt")

    best_valid = None
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if args.log_vram:
            torch.cuda.reset_peak_memory_stats()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            steps += 1

        dt = time.time() - t0
        avg_loss = total_loss / max(1, steps)
        vram_mb = None
        if args.log_vram:
            vram_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        valid_roc = None
        test_roc = None
        if do_eval:
            valid_roc = eval_split(model, valid_loader, evaluator, device)
            test_roc = eval_split(model, test_loader, evaluator, device)

        line = f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | {dt:.1f}s"
        if vram_mb is not None:
            line += f" | vram_mb={vram_mb}"
        if valid_roc is not None:
            line += f" | val_rocauc={valid_roc:.6f} | test_rocauc={test_roc:.6f}"
        print(line, flush=True)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "time_s": dt,
            "vram_mb": vram_mb,
            "valid": {"rocauc": valid_roc} if valid_roc is not None else None,
            "test": {"rocauc": test_roc} if test_roc is not None else None,
        })

        if do_eval and valid_roc is not None:
            if best_valid is None or valid_roc > best_valid:
                best_valid = valid_roc
                best_epoch = epoch
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_valid_rocauc": best_valid,
                }, best_path)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_valid_rocauc": best_valid,
            }, latest_path)

    meta = RunMeta(
        run_id=run_id,
        run_tag=tag,
        best_epoch=best_epoch,
        best_valid_rocauc=best_valid,
        best_path=best_path,
        latest_path=latest_path,
    )
    out = {"meta": asdict(meta), "args": vars(args), "history": history}
    with open(os.path.join(args.out_dir, f"gin_ogbg-molhiv_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
