"""
GIN/GINE for OGBG-MOLHIV with categorical Atom/Bond encoders (+ optional Virtual Node).

Why this exists:
- OGB molecular datasets store atom/bond features as small categorical integers.
- Treating them as floats (Linear) works but often leaves performance on the table.
- This script builds per-feature Embeddings and sums them into a hidden vector.

Constraints:
- No neighbor sampling (dataset is small; full-batch message passing per mini-batch).
- Reproducible: logs + JSON history + best/latest checkpoints.
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
from torch_geometric.nn import GINConv, GINEConv, GlobalAttention, global_add_pool, global_mean_pool


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


class CatSumEncoder(torch.nn.Module):
    """
    Per-column categorical embeddings summed into a single vector.

    Input: x int64 of shape [N, F]
    Output: [N, hidden]
    """

    def __init__(self, cardinals: list[int], hidden: int):
        super().__init__()
        self.embs = torch.nn.ModuleList([torch.nn.Embedding(int(c), hidden) for c in cardinals])
        for emb in self.embs:
            torch.nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        out = 0
        for j, emb in enumerate(self.embs):
            # Be defensive: some categories may appear only in val/test if we scanned too narrowly.
            # Clamping avoids CUDA device-side asserts and keeps the run reproducible.
            xj = x[:, j].clamp(min=0, max=emb.num_embeddings - 1)
            out = out + emb(xj)
        return out


class GINAtomBondClassifier(torch.nn.Module):
    def __init__(
        self,
        node_cardinals: list[int],
        edge_cardinals: list[int] | None,
        hidden: int,
        num_layers: int,
        dropout: float,
        pool: str,
        head_layers: int,
        virtual_node: bool,
    ):
        super().__init__()
        self.node_enc = CatSumEncoder(node_cardinals, hidden)
        self.use_edge_attr = edge_cardinals is not None and len(edge_cardinals) > 0
        self.edge_enc = CatSumEncoder(edge_cardinals, hidden) if self.use_edge_attr else None

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
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
        self.dropout = float(dropout)
        self.head = MLP(hidden, hidden, 1, layers=head_layers, dropout=self.dropout)

        self.attn_pool = None
        if self.pool == "attn":
            gate_nn = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1),
            )
            self.attn_pool = GlobalAttention(gate_nn=gate_nn)

        self.virtual_node = bool(virtual_node)
        if self.virtual_node:
            # One learnable "template" VN vector; expanded per graph in the batch.
            self.vn_emb = torch.nn.Embedding(1, hidden)
            torch.nn.init.xavier_uniform_(self.vn_emb.weight)
            self.vn_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
            )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        if self.pool == "sum":
            return global_add_pool(x, batch)
        if self.pool == "attn":
            assert self.attn_pool is not None
            return self.attn_pool(x, batch)
        raise ValueError(f"Unknown pool={self.pool!r}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, batch: torch.Tensor) -> torch.Tensor:
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


def scan_cardinals(ds: PygGraphPropPredDataset) -> tuple[list[int], list[int] | None]:
    """
    Infer per-column cardinalities from the full dataset (max + 1).
    We intentionally scan *all* graphs so we don't miss categories that appear
    only in valid/test (no label leakage risk: this uses only feature ranges).
    """
    first = ds[0]
    node_f = int(first.x.size(1))
    edge_f = 0 if first.edge_attr is None else int(first.edge_attr.size(1))
    node_max = torch.full((node_f,), -1, dtype=torch.long)
    edge_max = torch.full((edge_f,), -1, dtype=torch.long) if edge_f > 0 else None

    for i in range(len(ds)):
        g = ds[i]
        x = g.x
        if x.dtype != torch.long:
            x = x.long()
        node_max = torch.maximum(node_max, x.max(dim=0).values)
        if edge_f > 0 and g.edge_attr is not None:
            ea = g.edge_attr
            if ea.dtype != torch.long:
                ea = ea.long()
            edge_max = torch.maximum(edge_max, ea.max(dim=0).values)

    # +1 for max index, +1 extra as a safety margin (rare but cheap).
    node_card = (node_max + 2).clamp_min(2).tolist()
    edge_card = None
    if edge_max is not None:
        edge_card = (edge_max + 2).clamp_min(2).tolist()
    return node_card, edge_card


def build_scheduler(
    args: argparse.Namespace,
    optim: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler._LRScheduler | None, str]:
    """
    Return (scheduler, step_mode).
    step_mode: "batch" | "epoch" | "epoch_metric"
    """
    if args.sched == "none":
        return None, "epoch"

    if args.sched == "cosine":
        warm = int(args.warmup_epochs)
        total = int(args.epochs)
        if warm <= 0:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total, eta_min=float(args.min_lr))
            return sch, "epoch"

        # Warmup then cosine.
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lambda e: (e + 1) / max(1, warm),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=max(1, total - warm),
            eta_min=float(args.min_lr),
        )
        sch = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warm])
        return sch, "epoch"

    if args.sched == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=float(args.lr),
            total_steps=int(args.epochs) * int(steps_per_epoch),
            pct_start=float(args.onecycle_pct_start),
            anneal_strategy="cos",
            div_factor=float(args.onecycle_div_factor),
            final_div_factor=float(args.onecycle_final_div_factor),
        )
        return sch, "batch"

    if args.sched == "step":
        sch = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=int(args.step_size),
            gamma=float(args.gamma),
        )
        return sch, "epoch"

    if args.sched == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.min_lr),
            verbose=False,
        )
        return sch, "epoch_metric"

    raise ValueError(f"Unknown --sched {args.sched!r}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    ap = argparse.ArgumentParser(description="GIN v2 (Atom/Bond categorical encoders + optional Virtual Node) for ogbg-molhiv (ROC-AUC)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", choices=["mean", "sum", "attn"], default="mean")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--virtual_node", action="store_true")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--log_vram", action="store_true")
    ap.add_argument("--pos_weight", default="none", help="BCE pos_weight: none|auto|<float>")
    ap.add_argument("--sched", choices=["none", "cosine", "onecycle", "step", "plateau"], default="none")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="cosine warmup epochs (epoch-based warmup)")
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--onecycle_pct_start", type=float, default=0.1)
    ap.add_argument("--onecycle_div_factor", type=float, default=10.0)
    ap.add_argument("--onecycle_final_div_factor", type=float, default=100.0)
    ap.add_argument("--step_size", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=10)
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

    print("Scanning feature cardinalities (full dataset)...", flush=True)
    node_card, edge_card = scan_cardinals(ds)
    print(f"Node cardinals: {node_card}", flush=True)
    print(f"Edge cardinals: {edge_card}", flush=True)

    train_loader = DataLoader(ds[split["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(ds[split["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds[split["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = GINAtomBondClassifier(
        node_cardinals=node_card,
        edge_cardinals=edge_card,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pool=args.pool,
        head_layers=args.head_layers,
        virtual_node=args.virtual_node,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, sched_mode = build_scheduler(args, optim, steps_per_epoch=len(train_loader))

    pos_weight = None
    if args.pos_weight != "none":
        if args.pos_weight == "auto":
            ys = []
            for i in split["train"].tolist():
                ys.append(int(ds[int(i)].y.view(-1).item()))
            pos = sum(1 for y in ys if y == 1)
            neg = len(ys) - pos
            pw = (neg / max(1, pos))
            pos_weight = torch.tensor([pw], device=device)
            print(f"pos_weight=auto -> {pw:.6f}", flush=True)
        else:
            pw = float(args.pos_weight)
            pos_weight = torch.tensor([pw], device=device)
            print(f"pos_weight=float -> {pw:.6f}", flush=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag or run_id
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.model_dir, f"ginv2_ogbg-molhiv_{tag}_best.pt")
    latest_path = os.path.join(args.model_dir, f"ginv2_ogbg-molhiv_{tag}_latest.pt")

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
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward()
            optim.step()
            if scheduler is not None and sched_mode == "batch":
                scheduler.step()
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

        if scheduler is not None and sched_mode == "epoch":
            scheduler.step()
        if scheduler is not None and sched_mode == "epoch_metric" and valid_roc is not None:
            scheduler.step(valid_roc)

        lr_now = float(optim.param_groups[0]["lr"])
        line = f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | {dt:.1f}s"
        line += f" | lr={lr_now:.2e}"
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
                    "node_cardinals": node_card,
                    "edge_cardinals": edge_card,
                }, best_path)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_valid_rocauc": best_valid,
                "node_cardinals": node_card,
                "edge_cardinals": edge_card,
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
    with open(os.path.join(args.out_dir, f"ginv2_ogbg-molhiv_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
