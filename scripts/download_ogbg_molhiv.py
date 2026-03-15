import argparse

import torch
from ogb.graphproppred import PygGraphPropPredDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data/ogbg-molhiv')
    args = ap.parse_args()

    print('Loading/downloading ogbg-molhiv...', flush=True)
    ds = PygGraphPropPredDataset(name='ogbg-molhiv', root=args.root)
    print('OK', flush=True)
    print(ds, flush=True)

    split = ds.get_idx_split()
    print('Split sizes:', {k: int(v.numel()) for k, v in split.items()}, flush=True)

    g0 = ds[0]
    print('Example graph 0:', flush=True)
    print('  num_nodes:', int(g0.num_nodes), flush=True)
    print('  num_edges:', int(g0.edge_index.size(1)), flush=True)
    print('  x shape:', None if g0.x is None else tuple(g0.x.shape), flush=True)
    print('  edge_attr shape:', None if g0.edge_attr is None else tuple(g0.edge_attr.shape), flush=True)
    print('  y:', g0.y.view(-1).tolist(), flush=True)

    # Sanity: label distribution on train split (first pass, streaming)
    y = torch.cat([ds[i].y.view(-1) for i in split['train'].tolist()], dim=0)
    # y is {0,1} with possible NaNs? For molhiv it should be 0/1.
    n = int(y.numel())
    pos = int((y == 1).sum().item())
    neg = int((y == 0).sum().item())
    print('Train label counts:', {'n': n, 'pos': pos, 'neg': neg}, flush=True)


if __name__ == '__main__':
    main()
