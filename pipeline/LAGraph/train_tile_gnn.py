
"""
train_tile_gnn.py

Train a shared GNN model across sliding-window tiles.
Each tile directory must follow the convention:
  tile_{tid:04d}/
    - edges_b_p.csv   (cols: b_idx, poi_idx, dist, weight)
    - edges_b_b.csv   (cols: src, dst, dist, weight)
    - node_building.npz  (contains 'feats' array)
    - node_poi.npz       (contains 'feats' array)

Inputs:
 - --buildings : global buildings file (GeoPackage/shp) with labels:
      - `prob_vec` (JSON) preferred -> multi-label BCE loss
      - or `dominant_class` -> cross-entropy
 - --tiles_dir : directory that contains tile_* subfolders
 - tile_size / stride must match the tile generation used in step3 tiled script
Outputs:
 - saved model .pt
 - optional per-tile predictions saved under tiles_dir/tile_xxxx/predictions.npy
"""

import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder

# PyG imports
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False
    raise RuntimeError("This script requires torch_geometric. Install it before running.")

# -----------------------
# Utility helpers
# -----------------------
def tile_generator(total_bounds, tile_size, stride):
    xmin, ymin, xmax, ymax = total_bounds
    x = xmin
    tid = 0
    while x < xmax:
        y = ymin
        while y < ymax:
            tid += 1
            yield tid, (x, y, x + tile_size, y + tile_size)
            y += stride
        x += stride

def load_npz_feats(npz_fp):
    arr = np.load(npz_fp)
    if 'feats' in arr:
        return arr['feats']
    # support legacy: single array saved unnamed
    keys = [k for k in arr.files]
    if len(keys) > 0:
        return arr[keys[0]]
    raise RuntimeError(f"No feats key found in {npz_fp}")

def aggregate_poi_to_building(node_poi_feats, edges_b_p_df):
    """
    node_poi_feats: (n_poi, f_p)
    edges_b_p_df: DataFrame with columns ['b_idx','p_idx','dist','weight']
    returns: (n_build, f_p) aggregated features per building = sum(weight * poi_feat)
    """
    if edges_b_p_df.shape[0] == 0:
        return np.zeros((0, node_poi_feats.shape[1]), dtype=float)
    n_build = int(edges_b_p_df['b_idx'].max()) + 1
    fdim = node_poi_feats.shape[1]
    agg = np.zeros((n_build, fdim), dtype=float)
    for row in edges_b_p_df.itertuples(index=False):
        bidx = int(row.b_idx)
        pidx = int(row.p_idx)
        w = float(row.weight)
        if pidx >= node_poi_feats.shape[0]:
            continue
        agg[bidx] += w * node_poi_feats[pidx]
    return agg

# -----------------------
# Model
# -----------------------
class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ReLU()
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers-2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs)-1:
                x = self.act(x)
                x = self.dropout(x)
        return x  # raw logits (no activation)

# -----------------------
# Training loop across tiles
# -----------------------
def train(args):
    # load global buildings to fetch labels; ensure projected coords same as tiles
    buildings = gpd.read_file(args.buildings)
    # detect labels
    has_prob_vec = 'prob_vec' in buildings.columns
    has_dom = 'dominant_class' in buildings.columns
    if not has_prob_vec and not has_dom:
        print("[WARN] No 'prob_vec' or 'dominant_class' in buildings. Training will be unsupervised/won't run supervised loss.")
    # if dominants exist, create encoder
    label_encoder = None
    num_classes = None
    if has_dom:
        label_encoder = LabelEncoder()
        label_encoder.fit(buildings['dominant_class'].astype(str).values)
        num_classes = len(label_encoder.classes_)
    if has_prob_vec:
        # get C from one example
        for idx, row in buildings.iterrows():
            pv = row['prob_vec']
            if pv is None or (isinstance(pv,float) and math.isnan(pv)):
                continue
            if isinstance(pv, str):
                try:
                    d = json.loads(pv)
                except Exception:
                    d = eval(pv)
            else:
                d = pv
            num_classes = len(d)
            break

    if num_classes is None:
        raise RuntimeError("Could not infer number of classes. Make sure buildings has 'prob_vec' or 'dominant_class'.")

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using device:", device)

    # instantiate model
    # determine input dim from a sample tile
    sample_tile = sorted(glob(os.path.join(args.tiles_dir, 'tile_*')))[0]
    b_npz = os.path.join(sample_tile, 'node_building.npz')
    p_npz = os.path.join(sample_tile, 'node_poi.npz')
    if not os.path.exists(b_npz) or not os.path.exists(p_npz):
        raise RuntimeError("Sample tile missing node_building.npz or node_poi.npz; ensure tiles_dir correct.")
    b_feats_sample = load_npz_feats(b_npz)
    p_feats_sample = load_npz_feats(p_npz)
    # aggregated poi dim
    edges_bp_sample = pd.read_csv(os.path.join(sample_tile, 'edges_b_p.csv')) if os.path.exists(os.path.join(sample_tile, 'edges_b_p.csv')) else pd.DataFrame(columns=['b_idx','p_idx','dist','weight'])
    agg_p_sample = aggregate_poi_to_building(p_feats_sample, edges_bp_sample)
    in_dim = b_feats_sample.shape[1] + (agg_p_sample.shape[1] if agg_p_sample.shape[0]>0 else 0)
    print("Model in_dim:", in_dim, "num_classes:", num_classes)

    model = GraphSAGENet(in_channels=in_dim, hidden_channels=args.hidden, out_channels=num_classes, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # loss selection
    if has_prob_vec:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # generate tiles grid to iterate in deterministic order
    # We need tile grid bounds. We'll infer from existing tile folders names by counting them and recomputing using the same tile_size & stride
    # Simpler: list tile folders in sorted order and attempt to load each.
    tile_dirs = sorted([d for d in glob(os.path.join(args.tiles_dir, 'tile_*')) if os.path.isdir(d)])
    if len(tile_dirs) == 0:
        raise RuntimeError("No tile directories found under tiles_dir.")

    print(f"Found {len(tile_dirs)} tiles. Training on {args.epochs} epochs...")

    # main epochs
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        cnt = 0
        # shuffle tile order each epoch if desired
        if args.shuffle_tiles:
            import random
            random.shuffle(tile_dirs)
        for tdir in tqdm(tile_dirs, desc=f"Epoch {epoch}"):
            # load tile arrays
            b_npz = os.path.join(tdir, 'node_building.npz')
            p_npz = os.path.join(tdir, 'node_poi.npz')
            edges_bp_fp = os.path.join(tdir, 'edges_b_p.csv')
            edges_bb_fp = os.path.join(tdir, 'edges_b_b.csv')
            if not (os.path.exists(b_npz) and os.path.exists(p_npz) and os.path.exists(edges_bb_fp)):
                # skip incomplete tile
                continue
            b_feats = load_npz_feats(b_npz)  # (n_build, fb)
            p_feats = load_npz_feats(p_npz)  # (n_poi, fp)
            edges_bb = pd.read_csv(edges_bb_fp)
            edges_bp = pd.read_csv(edges_bp_fp) if os.path.exists(edges_bp_fp) else pd.DataFrame(columns=['b_idx','p_idx','dist','weight'])

            # aggregate poi features to building
            agg_p = aggregate_poi_to_building(p_feats, edges_bp)  # (n_build, fp)
            # if some buildings have no agg entry, make zeros with correct shape
            if agg_p.shape[0] < b_feats.shape[0]:
                pad = np.zeros((b_feats.shape[0]-agg_p.shape[0], p_feats.shape[1]))
                agg_p = np.vstack([agg_p, pad])

            # combine features
            X = np.hstack([b_feats, agg_p])  # (n_build, in_dim)
            X = torch.tensor(X, dtype=torch.float, device=device)

            # edge_index for buildings
            # edges_bb columns: src,dst
            if len(edges_bb) == 0:
                # no edges in tile -> skip (not meaningful for GNN)
                continue
            edge_index = torch.tensor([edges_bb['src'].values.astype(int), edges_bb['dst'].values.astype(int)], dtype=torch.long, device=device)

            # labels: derive from global buildings GeoDataFrame by spatial intersection
            # We need to fetch the buildings in this tile in the same order as node_building.npz was created.
            # Approach: derive tile bounding box from files: we assume tile dirs were built in deterministic order
            # So we will reconstruct which global buildings intersect tile bbox by reading edges_b_b length and b_feats shape:
            # Simpler: expect that the node order corresponds to the subset of buildings whose centroids lie within the tile bbox.
            # So compute tile bbox by reading any geometry file if present or parse tile id -> We can't rely on that robustly.
            # Instead, we ask user to provide global 'buildings' gdf and we filter by bounding box matching tile derived from filenames.
            # We'll compute bbox from buildings centroids matching X.shape[0] nearest buildings to tile center as a fallback.

            # fetch tile centroid from building coords (approx)
            # Use global buildings to extract labels by finding n_build nearest buildings to tile centroid.
            # This is heuristic but should match earlier ordering if node_building was derived from buildings.intersects(tile_geom).
            # compute approximate tile center from mean of building centroids in X: but we don't have geometry here.
            # Therefore, require a mapping file is best. If not available, we attempt best-effort by spatial matching:
            try:
                # Reconstruct tile bbox by checking tile folder name order and using global tiling function is fragile.
                # Fallback approach: find buildings in global whose centroids fall within tile derived from edges' coordinate ranges.
                # We use the coordinates stored in global buildings gdf centroids to find the subset matching X.shape[0].
                b_global_centroids = buildings.geometry.centroid
                # compute pairwise candidate by finding buildings whose centroids fall inside the min/max of building feature coords?
                # This is complex and brittle - instead, try to match by nearest: find n_build nearest global buildings to the mean coordinates of all buildings in the tile from edges.
                # We'll estimate tile centroid by using building indices present in edges_bb (src/dst) if they are global indices (they are local), so can't directly map.
                # Best here: user should provide global mapping file per tile. If not provided, we try the following heuristic:
                # compute bounding box of POI coords saved in node_poi? We can't get coords from npz (only feats). So cannot robustly map.
                # Therefore, we require that user provide argument --buildings that contains tile assignment column 'tile_id' or that node_building.npz contains meta ids.
                # Check if node_building.npz contains 'ids' key:
                arr = np.load(b_npz)
                if 'ids' in arr.files:
                    ids = arr['ids']
                    # fetch labels from buildings gdf by index
                    targets = []
                    for bid in ids:
                        row = buildings.loc[int(bid)]
                        if 'prob_vec' in buildings.columns:
                            pv = row['prob_vec']
                            if isinstance(pv, str):
                                tvec = np.array(list(json.loads(pv).values()), dtype=float)
                            elif isinstance(pv, dict):
                                tvec = np.array(list(pv.values()), dtype=float)
                            else:
                                tvec = np.ones(num_classes)/num_classes
                            targets.append(tvec)
                        else:
                            dom = row.get('dominant_class', None)
                            if dom is None:
                                targets.append(np.zeros(num_classes))
                            else:
                                targets.append(label_encoder.transform([str(dom)])[0])
                    targets = np.array(targets)
                else:
                    # try 'idxs.npy' mapping or model cannot find mapping
                    raise RuntimeError("node_building.npz does not contain 'ids' key mapping to global building indices. "
                                       "Please regenerate tiles including ids in node_building.npz or provide mapping.")
            except Exception as e:
                print("ERROR: Unable to map tile nodes to global building labels automatically.")
                print("Detail:", str(e))
                print("You must ensure each tile's node_building.npz contains an 'ids' array (global building index). Exiting.")
                return

            # form labels tensor
            if has_prob_vec:
                y = torch.tensor(targets, dtype=torch.float, device=device)  # (n_build, C)
            else:
                y = torch.tensor(targets, dtype=torch.long, device=device)   # (n_build,) int labels

            # forward
            optimizer.zero_grad()
            logits = model(X, edge_index)  # (n_build, C)
            if has_prob_vec:
                loss = loss_fn(logits, y)
            else:
                loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            cnt += 1

        avg_loss = total_loss / max(1, cnt)
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")

        # optional: save intermediate model
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), args.out_model.replace('.pt', f'.epoch{epoch}.pt'))
            print("Saved model checkpoint at epoch", epoch)

    # final save
    torch.save(model.state_dict(), args.out_model)
    print("Training complete. Model saved to:", args.out_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tile GNN (shared model across tiles)")
    parser.add_argument("--buildings", required=True, help="Global buildings file (with labels: prob_vec or dominant_class).")
    parser.add_argument("--tiles_dir", required=True, help="Directory with tile_* subfolders produced by step3 tiled script.")
    parser.add_argument("--tile_size", type=float, default=2000.0, help="Tile size (meters) - must match tile generation.")
    parser.add_argument("--stride", type=float, default=1500.0, help="Tile stride (meters) - must match tile generation.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--out_model", default="tile_gnn_model.pt")
    parser.add_argument("--shuffle_tiles", action='store_true', help="Shuffle tile order each epoch")
    parser.add_argument("--cpu", action='store_true', help="Force CPU")
    args = parser.parse_args()
    train(args)
