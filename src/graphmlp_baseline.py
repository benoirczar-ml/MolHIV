"""
Graph-level binary classification baseline for OGBG-MOLHIV.

Baseline: node MLP -> global pooling -> graph MLP -> logit

Goal:
- fast, stable, minimal moving parts
- validates data pipeline + ROC-AUC evaluation
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
from torch_geometric.nn import global_add_pool, global_mean_pool


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
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        assert layers >= 1
        mods = []
        dim = in_dim
        for _ in range(layers - 1):
            mods += [torch.nn.Linear(dim, hidden), torch.nn.ReLU()]
            if dropout and dropout > 0:
                mods += [torch.nn.Dropout(dropout)]
            dim = hidden
        mods.append(torch.nn.Linear(dim, hidden))
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMLP(torch.nn.Module):
    def __init__(
        self,
        node_in: int,
        node_hidden: int,
        node_layers: int,
        graph_hidden: int,
        graph_layers: int,
        dropout: float,
        pool: str,
    ):
        super().__init__()
        self.node_mlp = MLP(node_in, node_hidden, node_layers, dropout)
        self.graph_mlp = MLP(node_hidden, graph_hidden, graph_layers, dropout)
        self.out = torch.nn.Linear(graph_hidden, 1)
        self.pool = pool

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.node_mlp(x)
        if self.pool == "mean":
            g = global_mean_pool(x, batch)
        elif self.pool == "sum":
            g = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pool={self.pool!r}")
        g = self.graph_mlp(g)
        return self.out(g).view(-1)  # logits


@torch.no_grad()
def eval_split(model: torch.nn.Module, loader: DataLoader, evaluator: Evaluator, device: torch.device) -> float:
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x.float(), batch.batch)
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

    ap = argparse.ArgumentParser(description="GraphMLP baseline for ogbg-molhiv (ROC-AUC)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--pos_weight", type=str, default="none",
                    help="Class imbalance handling for BCEWithLogitsLoss. "
                         "Use 'auto' to set pos_weight = (#neg/#pos) from train split, "
                         "or 'none' to disable, or a float value like '25.7'.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", choices=["mean", "sum"], default="mean")
    ap.add_argument("--node_hidden", type=int, default=128)
    ap.add_argument("--node_layers", type=int, default=2)
    ap.add_argument("--graph_hidden", type=int, default=128)
    ap.add_argument("--graph_layers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
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

    train_loader = DataLoader(ds[split["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(ds[split["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds[split["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    node_in = int(ds[0].x.size(1))
    model = GraphMLP(
        node_in=node_in,
        node_hidden=args.node_hidden,
        node_layers=args.node_layers,
        graph_hidden=args.graph_hidden,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # pos_weight for imbalance (MolHIV is highly imbalanced)
    if args.pos_weight == "none":
        pos_weight = None
    elif args.pos_weight == "auto":
        y_train = torch.cat([ds[i].y.view(-1) for i in split["train"].tolist()], dim=0)
        pos = float((y_train == 1).sum().item())
        neg = float((y_train == 0).sum().item())
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
        print(f"pos_weight=auto -> {pos_weight.item():.4f} (neg={neg:.0f}, pos={pos:.0f})", flush=True)
    else:
        pos_weight = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)
        print(f"pos_weight={pos_weight.item():.4f}", flush=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag or run_id
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.model_dir, f"graphmlp_ogbg-molhiv_{tag}_best.pt")
    latest_path = os.path.join(args.model_dir, f"graphmlp_ogbg-molhiv_{tag}_latest.pt")

    best_valid = None
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(batch.x.float(), batch.batch)
            y = batch.y.view(-1).float()
            if pos_weight is None:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            steps += 1

        dt = time.time() - t0
        avg_loss = total_loss / max(1, steps)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        valid_roc = None
        test_roc = None
        if do_eval:
            valid_roc = eval_split(model, valid_loader, evaluator, device)
            test_roc = eval_split(model, test_loader, evaluator, device)

        line = f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | {dt:.1f}s"
        if valid_roc is not None:
            line += f" | val_rocauc={valid_roc:.6f} | test_rocauc={test_roc:.6f}"
        print(line, flush=True)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "time_s": dt,
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
    with open(os.path.join(args.out_dir, f"graphmlp_ogbg-molhiv_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
