#!/usr/bin/env python3
"""
Step 3 (Tiled Version): Sliding-window Graph Construction
for large-scale automatic building function prediction.

Each tile builds its own local graph (building–POI–building)
and saves independent edge/node files.
"""

import os, math, json, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from sklearn.neighbors import BallTree, NearestNeighbors

# ---------- 参数设置 ----------
def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window graph construction (Step3-Tiled)")
    p.add_argument("--buildings", required=True, help="Path to building shapefile or GeoPackage")  #
    p.add_argument("--pois", required=True, help="Path to POI shapefile or GeoPackage")  #
    p.add_argument("--out_dir", default="graph_tiles", help="Output directory")
    p.add_argument("--tile_size", type=float, default=2000, help="Tile size (meters)")
    p.add_argument("--stride", type=float, default=1500, help="Stride for sliding window (meters)")
    p.add_argument("--radius_bp", type=float, default=300, help="Search radius for building->POI (meters)")
    p.add_argument("--sigma", type=float, default=200, help="Sigma for Gaussian distance weight")
    p.add_argument("--k_b2b", type=int, default=8, help="K for building->building edges")
    return p.parse_args()

# ---------- 辅助函数 ----------
def coords_from_gdf(gdf):
    return np.vstack([ (geom.centroid.x, geom.centroid.y) for geom in gdf.geometry ])

def distance_kernel(d, sigma=200.0):
    return math.exp(-d*d/(2*sigma*sigma))

def compute_local_poi_weights(pois_tile):
    """计算局部POI类别权重"""
    counts = pois_tile['dominant_c'].value_counts()
    N_total = counts.sum()
    weights = {}
    for cat, n in counts.items():
        weights[cat] = math.log((N_total + 1e-6) / (n + 1e-6))
    return weights

def build_edges_bp(buildings, pois, radius, sigma):
    """构建建筑→POI边"""
    b_coords = coords_from_gdf(buildings)
    p_coords = coords_from_gdf(pois)
    p_tree = BallTree(p_coords, metric='euclidean')
    idx_list = p_tree.query_radius(b_coords, r=radius)
    edges = []
    for bi, idxs in enumerate(idx_list):
        bx, by = b_coords[bi]
        for pj in idxs:
            px, py = p_coords[pj]
            d = math.hypot(px-bx, py-by)
            w = distance_kernel(d, sigma=sigma)
            edges.append((bi, pj, d, w))
    return pd.DataFrame(edges, columns=['b_idx','p_idx','dist','weight'])

def build_edges_bb(buildings, k=8, sigma=200):
    """构建建筑→建筑边"""
    b_coords = coords_from_gdf(buildings)
    nbrs = NearestNeighbors(n_neighbors=min(k+1,len(b_coords))).fit(b_coords)
    dists, idxs = nbrs.kneighbors(b_coords)
    edges = []
    for i in range(len(b_coords)):
        for j, d in zip(idxs[i,1:], dists[i,1:]):
            w = distance_kernel(d, sigma=sigma)
            edges.append((i, j, d, w))
    return pd.DataFrame(edges, columns=['src','dst','dist','weight'])

def tile_generator(total_bounds, tile_size, stride):
    xmin, ymin, xmax, ymax = total_bounds
    x = xmin
    tile_id = 0
    while x < xmax:
        y = ymin
        while y < ymax:
            tile_id += 1
            yield tile_id, box(x, y, x+tile_size, y+tile_size)
            y += stride
        x += stride

# ---------- 主流程 ----------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data ...")
    buildings = gpd.read_file(args.buildings)
    pois = gpd.read_file(args.pois)

    assert buildings.crs == pois.crs, "CRS mismatch! Reproject both to the same projected CRS (e.g. EPSG:3857)"

    print("Generating sliding tiles ...")
    tiles = list(tile_generator(buildings.total_bounds, args.tile_size, args.stride))
    print(f"Total tiles: {len(tiles)}")

    for tid, tile_geom in tqdm(tiles, desc="Processing tiles"):
        tile_dir = os.path.join(args.out_dir, f"tile_{tid:04d}")
        os.makedirs(tile_dir, exist_ok=True)

        b_tile = buildings[buildings.intersects(tile_geom)]
        p_tile = pois[pois.intersects(tile_geom)]

        if len(b_tile) == 0 or len(p_tile) == 0:
            continue

        # Step 1: 局部权重计算
        local_weights = compute_local_poi_weights(p_tile)

        # Step 2: 生成POI特征
        classes = sorted(p_tile['dominant_c'].unique())
        p_feats = []
        for _, r in p_tile.iterrows():
            vec = np.zeros(len(classes))
            c = r['dominant_c']
            if c in classes:
                idx = classes.index(c)
                vec[idx] = local_weights.get(c,1.0)
            p_feats.append(vec)
        p_feats = np.array(p_feats)

        # Step 3: 建筑物特征（如果有dominant_class或prob_vec可扩展）
        b_feats = np.ones((len(b_tile), 1))  # 暂时统一占位

        # Step 4: 构建边
        edges_bp = build_edges_bp(b_tile, p_tile, args.radius_bp, args.sigma)
        edges_bb = build_edges_bb(b_tile, args.k_b2b, args.sigma)

        # Step 5: 保存
        edges_bp.to_csv(os.path.join(tile_dir, "edges_b_p.csv"), index=False)
        edges_bb.to_csv(os.path.join(tile_dir, "edges_b_b.csv"), index=False)
        np.savez_compressed(os.path.join(tile_dir, "node_poi.npz"), feats=p_feats)
        np.savez_compressed(os.path.join(tile_dir, "node_building.npz"), feats=b_feats)
    
    print("✅ All tiles processed. Local graphs saved to", args.out_dir)

if __name__ == "__main__":
    main()
