#!/usr/bin/env python3
"""
Summarize OGBG-MOLHIV run JSON files into a small committed report.

Usage:
  python scripts/summarize_runs.py --runs_dir runs --out_md reports/RESULTS_GENERATED.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass


@dataclass
class Row:
    run_tag: str
    model: str
    best_epoch: int | None
    best_valid: float | None
    best_test: float | None


def load_best(path: str) -> Row:
    with open(path, "r") as f:
        d = json.load(f)
    tag = d.get("meta", {}).get("run_tag") or os.path.splitext(os.path.basename(path))[0]
    best_epoch = d.get("meta", {}).get("best_epoch")
    best_valid = d.get("meta", {}).get("best_valid_rocauc")

    best_test = None
    for h in d.get("history", []):
        if h.get("epoch") == best_epoch and h.get("test") is not None:
            best_test = h["test"].get("rocauc")
            break

    # Infer a short model label from the JSON filename prefix.
    base = os.path.basename(path)
    if base.startswith("graphmlp_"):
        model = "GraphMLP"
    elif base.startswith("ginv2_"):
        model = "GINv2"
    elif base.startswith("gin_"):
        model = "GIN/GINE"
    else:
        model = "Unknown"

    return Row(run_tag=str(tag), model=model, best_epoch=best_epoch, best_valid=best_valid, best_test=best_test)


def fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_md", default="reports/RESULTS_GENERATED.md")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    rows = [load_best(p) for p in paths]

    lines = []
    lines.append("# Results (Generated)\n")
    lines.append("| Run tag | Model | Best epoch | Best valid ROC-AUC | Test ROC-AUC @ best |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| `{r.run_tag}` | {r.model} | {r.best_epoch or ''} | {fmt(r.best_valid)} | {fmt(r.best_test)} |")

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {args.out_md} ({len(rows)} runs)")


if __name__ == "__main__":
    main()

