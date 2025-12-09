#!/usr/bin/env python3
"""
building_poi_pipeline.py

End-to-end framework:
 - road -> blocks (street blocks)
 - compute block-level POI weights (dynamic scanning)
 - for each building compute class contributions -> normalize -> p_vec
 - spatial smoothing
 - build graph for optional GNN training/inference

"""

import argparse
import os
import json
from collections import Counter, defaultdict
import math
import multiprocessing as mp

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, polygonize
from sklearn.neighbors import BallTree, NearestNeighbors
from tqdm import tqdm

# Optional imports for GNN - these are optional; wrap in try/except
try:
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborSampler
    from torch_geometric.nn import SAGEConv, to_hetero
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -------------------------
# CONFIG & DEFAULT PARAMS
# -------------------------
DEFAULTS = {
    "proj": None,  # if None, will auto-calc UTM based on centroid
    "max_radius": 1000.0,      # meters fallback search radius
    "min_poi_per_block": 8,
    "kappa": 1.0,              # sigma scaling
    "alpha": 0.5,              # pow-decay for weight w_c = 1/(freq^alpha)
    "smooth_lambda": 0.6,
    "smooth_iters": 2,
    "smooth_radius": 50.0,     # meters for building smoothing neighbors
    "min_sigma": 20.0,         # avoid zero sigma
    "eps": 1e-9,
    "grid_downsample_threshold": 500,  # #poi in block to trigger downsample
    "grid_size_downsample": 50.0,      # meters grid for downsampling dense POI
    "conf_th": 0.75,
}

# -------------------------
# Utility functions
# -------------------------

# -------------------------
# Robust shapefile reader (handle DBF encoding / pyogrio issues)
# -------------------------
def safe_read_shp(path):
    """
    Robustly read a shapefile (.shp) using geopandas.
    Tries: pyogrio with utf-8 -> gbk, then fiona engine -> fallback to reading geometries only.
    Returns a GeoDataFrame.
    """
    import warnings
    # try pyogrio / default with utf-8
    try:
        gdf = gpd.read_file(path, encoding='utf-8')
        return gdf
    except Exception as e1:
        print(f"[safe_read_shp] utf-8 read failed: {e1}")
    # try gbk (common for Chinese DBF)
    try:
        gdf = gpd.read_file(path, encoding='gbk')
        print("[safe_read_shp] read with encoding='gbk' succeeded.")
        return gdf
    except Exception as e2:
        print(f"[safe_read_shp] gbk read failed: {e2}")
    # try fiona engine explicitly
    try:
        gdf = gpd.read_file(path, engine="fiona")
        print("[safe_read_shp] read with engine='fiona' succeeded.")
        return gdf
    except Exception as e3:
        print(f"[safe_read_shp] fiona read failed: {e3}")
    # fallback: read only geometries using fiona and build GeoDataFrame (lose attributes)
    try:
        import fiona
        from shapely.geometry import shape
        geoms = []
        props = []
        with fiona.open(path) as src:
            crs = src.crs
            for feat in src:
                try:
                    geoms.append(shape(feat['geometry']))
                    props.append(feat.get('properties', {}))
                except Exception as ge:
                    # skip invalid features
                    print(f"[safe_read_shp] skipping feature due to geometry error: {ge}")
        if len(geoms) == 0:
            raise RuntimeError("No valid geometries read from shapefile.")
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
        print("[safe_read_shp] fallback geometry-only read succeeded (attributes may be partial).")
        return gdf
    except Exception as e4:
        print(f"[safe_read_shp] fallback geometry-only read failed: {e4}")
        raise RuntimeError(f"Failed to read shapefile {path} with multiple fallbacks.")


def determine_project_crs(gdf_list):
    """Determine a suitable projected CRS (UTM-like) based on data centroid.
       Return EPSG code string like 'EPSG:32650' or a proj4 string.
    """
    # if user provided projection, they can override. Here compute mean centroid lon/lat
    concat = pd.concat([gpd.GeoSeries(gdf.geometry.centroid) for gdf in gdf_list])
    centroid = concat.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    # compute UTM zone EPSG
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return f"EPSG:{epsg}"

def reproject_to(gdf, crs):
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Please set CRS first.")
    return gdf.to_crs(crs)

def build_blocks_from_roads(roads_gdf, min_area=10.0):
    """
    Given road line geometries, polygonize to blocks.
    Returns GeoDataFrame of blocks with block_id.
    """
    # unify lines
    merged = unary_union(roads_gdf.geometry)
    # polygonize -> produce polygons
    polys = list(polygonize(merged))
    blocks = gpd.GeoDataFrame(geometry=polys)
    blocks = blocks.set_crs(roads_gdf.crs)
    # filter tiny or degenerate
    blocks['area'] = blocks.geometry.area
    blocks = blocks[blocks['area'] > min_area].reset_index(drop=True)
    blocks['block_id'] = blocks.index.astype(int)
    return blocks[['block_id', 'geometry', 'area']]

def compute_block_assignments(pois, buildings, blocks):
    """
    Spatial join to assign block_id to POIs and buildings (buildings by centroid).
    """
    pois_s = gpd.sjoin(pois, blocks[['block_id', 'geometry']], how='left', predicate='within')
    pois_s = pois_s.drop(columns=['index_right'], errors='ignore')
    # building centroid to block
    b = buildings.copy()
    b['centroid'] = b.geometry.centroid
    b_cent = gpd.GeoDataFrame(b[['centroid']], geometry='centroid', crs=buildings.crs)
    b_cent = b_cent.set_geometry('centroid')
    b_cent = gpd.sjoin(b_cent, blocks[['block_id','geometry']], how='left', predicate='within')
    b['block_id'] = b_cent['block_id'].values
    # cleanup
    pois_s = pois_s.reset_index(drop=True)
    b = b.reset_index(drop=True)
    return pois_s, b

def compute_block_poi_stats(pois_s, classes):
    """
    For every block compute counts per class and total.
    Returns dict: block -> { 'N': total, 'freq': {class:count}, 'w': {class:weight} }
    Weight uses IDF-style fallback or power-law.
    """
    block_groups = pois_s.groupby('block_id')
    block_stats = {}
    for block_id, g in block_groups:
        freq = g['poi_class'].value_counts().to_dict()
        N = len(g)
        # compute weight - use log(idf) fallback: w = log(N/(1+freq))
        w = {}
        for c in classes:
            cnt = freq.get(c, 0)
            w[c] = math.log((N + 1) / (cnt + 1))  # +1 smoothing
        block_stats[int(block_id)] = {'N': int(N), 'freq': freq, 'w': w}
    return block_stats

def compute_global_class_stats(pois, classes, alpha=0.5):
    M = len(pois)
    freq = pois['poi_class'].value_counts().to_dict()
    w_global = {}
    for c in classes:
        cnt = freq.get(c, 0)
        # power decay
        w_global[c] = 1.0 / ((cnt + 1) ** alpha)
    # sigma global estimation per class using 1-NN median
    sigma = {}
    for c in classes:
        coords = pois.loc[pois['poi_class']==c, ['x','y']].values
        if len(coords) < 2:
            sigma[c] = DEFAULTS['min_sigma']
            continue
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(coords)
        dists, _ = nbrs.kneighbors(coords)
        med = np.median(dists[:,1])
        sigma[c] = max(med * DEFAULTS['kappa'], DEFAULTS['min_sigma'])
    return w_global, sigma

def build_balltree(coords):
    """coords: Nx2 array in projected meters"""
    return BallTree(coords, leaf_size=40, metric='euclidean')

def grid_downsample_pois_in_block(pois_block, grid_size):
    """
    If very dense, aggregate by grid: return representative points with weight.
    pois_block: GeoDataFrame with geometry in projected coords
    returns: DataFrame with columns x,y,poi_class,importance,count_in_grid
    """
    xs = pois_block.geometry.x.values
    ys = pois_block.geometry.y.values
    ix = np.floor(xs / grid_size).astype(int)
    iy = np.floor(ys / grid_size).astype(int)
    grp_keys = list(zip(ix, iy, pois_block['poi_class'].values))
    df = pd.DataFrame({
        'ix': ix, 'iy': iy, 'poi_class': pois_block['poi_class'].values,
        'x': xs, 'y': ys
    })
    agg = df.groupby(['ix','iy','poi_class']).agg({'x':'mean','y':'mean','poi_class':'count'}).rename(columns={'poi_class':'count'}).reset_index()
    agg['importance'] = agg['count']  # keep count as importance/weight
    return agg[['x','y','poi_class','importance']]

# -------------------------
# Core pipeline functions
# -------------------------
def compute_building_contributions_parallel(buildings_df, pois_df, block_stats,
                                             sigma_global, w_global, classes,
                                             max_radius=1000.0, min_poi_per_block=8,
                                             downsample_threshold=500, grid_size=50.0,
                                             n_jobs=4):
    """
    Compute per-building class score vector S_{i,c}, then normalize to p_vec.
    Returns dict building_index -> np.array(p_vec)
    """
    # prepare POI arrays for BallTree (global)
    poi_coords = pois_df[['x','y']].values
    poi_tree = build_balltree(poi_coords)
    # mapping for fast access
    poi_class_arr = pois_df['poi_class'].values
    poi_importance_arr = pois_df.get('importance', pd.Series([1]*len(pois_df))).values

    # build block->poi idx mapping for quick selection
    pois_by_block = defaultdict(list)
    for idx, row in pois_df.reset_index().iterrows():
        b = row.get('block_id', None)
        if pd.isna(b):
            continue
        pois_by_block[int(b)].append(idx)

    def worker(idx_row):
        i, b_row = idx_row
        centroid_x = b_row['centroid'].x if 'centroid' in b_row else b_row.geometry.centroid.x
        centroid_y = b_row['centroid'].y if 'centroid' in b_row else b_row.geometry.centroid.y
        block_id = b_row.get('block_id', None)
        # candidate poi idxs
        cand_idxs = []
        if block_id is not None and int(block_id) in pois_by_block:
            idxs_in_block = pois_by_block[int(block_id)]
            if len(idxs_in_block) >= min_poi_per_block:
                cand_idxs = idxs_in_block
        if len(cand_idxs) < min_poi_per_block:
            # fall back to radius query
            cand_idxs = poi_tree.query_radius([[centroid_x, centroid_y]], r=max_radius)[0].tolist()
        if len(cand_idxs) == 0:
            # no POI nearby, return uniform small probs
            p = np.ones(len(classes)) / len(classes)
            return i, p

        S = np.zeros(len(classes), dtype=float)
        for j in cand_idxs:
            # poi j info
            xj, yj = poi_coords[j]
            d = math.hypot(xj - centroid_x, yj - centroid_y)
            c = poi_class_arr[j]
            importance = poi_importance_arr[j] if j < len(poi_importance_arr) else 1.0
            if c not in classes:
                continue
            cidx = classes.index(c)
            # sigma choose global; optionally could pick block local sigma if computed
            sigma = sigma_global.get(c, DEFAULTS['min_sigma'])
            K = math.exp(- (d*d) / (2.0 * sigma * sigma)) if sigma > 0 else 0.0
            # choose block-local weight if available else global
            w_block = w_global.get(c, 1.0)
            if block_id is not None and int(block_id) in block_stats:
                w_block = block_stats[int(block_id)]['w'].get(c, w_block)
            contrib = w_block * importance * K
            S[cidx] += contrib
        total = S.sum() + DEFAULTS['eps']
        p_vec = S / total
        return i, p_vec

    # parallel map
    inputs = list(buildings_df.iterrows())
    results = {}
    if n_jobs == 1:
        for item in tqdm(inputs, desc="computing building contributions"):
            i, p = worker(item)
            results[i] = p
    else:
        with mp.Pool(processes=n_jobs) as pool:
            for i, p in tqdm(pool.imap_unordered(worker, inputs), total=len(inputs), desc="computing building contributions (parallel)"):
                results[i] = p
    return results

def spatial_smoothing(buildings_df, building_probs, smooth_radius, lam, iters=2):
    """
    Simple smoothing: for each building, average neighbor p_vec then combine with lambda
    building_probs: dict idx->np.array
    """
    coords = np.array([[b.geometry.centroid.x, b.geometry.centroid.y] for _, b in buildings_df.iterrows()])
    btree = build_balltree(coords)
    for _ in range(iters):
        new_probs = {}
        for i, b in buildings_df.iterrows():
            neigh_idxs = btree.query_radius([[b.geometry.centroid.x, b.geometry.centroid.y]], r=smooth_radius)[0]
            neigh_probs = [building_probs[int(k)] for k in neigh_idxs if int(k) in building_probs]
            if len(neigh_probs) == 0:
                new_probs[i] = building_probs[i]
                continue
            neigh_mean = np.mean(neigh_probs, axis=0)
            new_probs[i] = lam * building_probs[i] + (1 - lam) * neigh_mean
        building_probs = new_probs
    return building_probs

def write_output(buildings_gdf, building_probs, classes, out_path):
    # attach columns
    prob_list = []
    primary = []
    prim_p = []
    mixed = []
    for i, _ in buildings_gdf.iterrows():
        pvec = building_probs.get(i, np.ones(len(classes))/len(classes))
        prob_list.append(json.dumps({c: float(pvec[idx]) for idx, c in enumerate(classes)}))
        max_idx = int(np.argmax(pvec))
        primary.append(classes[max_idx])
        prim_p.append(float(pvec[max_idx]))
        mixed_classes = [classes[idx] for idx,v in enumerate(pvec) if v >= 0.2 and idx != max_idx]
        mixed.append(json.dumps(mixed_classes))
    buildings_gdf['prob_vec'] = prob_list
    buildings_gdf['primary_class'] = primary
    buildings_gdf['primary_prob'] = prim_p
    buildings_gdf['mixed_classes'] = mixed
    # save GeoPackage
    buildings_gdf.to_file(out_path, driver='GPKG')
    print(f"Wrote results to: {out_path}")

# -------------------------
# Optional: simple GNN example (sketch)
# -------------------------
def build_hetero_data_for_pyg(buildings_gdf, pois_gdf, building_probs, classes):
    """
    Build a small HeteroData object for PyG, using buildings nodes and poi nodes,
    and edges building->building (knn) and building->poi (radius). This is a sketch.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch/PyG not available. Install torch and torch_geometric to use GNN portion.")
    # NOTE: for very large graphs, need partitioning / neighbor sampling - this is a simple example
    # Build node features
    num_build = len(buildings_gdf)
    num_poi = len(pois_gdf)
    b_feats = []
    for i, b in buildings_gdf.iterrows():
        pvec = building_probs.get(i, np.ones(len(classes))/len(classes))
        feat = np.array([b.geometry.area])  # add area etc.
        feat = np.concatenate([feat, pvec])
        b_feats.append(feat)
    b_feats = torch.tensor(np.vstack(b_feats), dtype=torch.float)

    # poi features - simple one-hot for class
    poi_class_map = {c: idx for idx, c in enumerate(classes)}
    p_feats = []
    for _, p in pois_gdf.iterrows():
        vec = np.zeros(len(classes), dtype=float)
        ci = poi_class_map.get(p['poi_class'], None)
        if ci is not None:
            vec[ci] = 1.0
        p_feats.append(vec)
    p_feats = torch.tensor(np.vstack(p_feats), dtype=torch.float)

    data = HeteroData()
    data['building'].x = b_feats
    data['poi'].x = p_feats

    # edges: building->poi using radius queries (careful for large graphs)
    # Here we make a small subset for demo (not production)
    b_coords = np.array([[b.geometry.centroid.x, b.geometry.centroid.y] for _, b in buildings_gdf.iterrows()])
    p_coords = np.array([[p.geometry.x, p.geometry.y] for _, p in pois_gdf.iterrows()])
    # use KDTree logic in numpy for demo
    from sklearn.neighbors import RadiusNeighborsClassifier, NearestNeighbors
    p_tree = BallTree(p_coords)
    rows, cols = [], []
    for i, coord in enumerate(b_coords):
        idxs = p_tree.query_radius([coord], r=DEFAULTS['max_radius'])[0]
        for j in idxs:
            rows.append(i)
            cols.append(j)
    if len(rows) == 0:
        raise RuntimeError("No building-poi edges found. Check radii or data.")
    data['building', 'near', 'poi'].edge_index = torch.tensor([rows, cols], dtype=torch.long)
    # optionally poi->building reverse edge
    data['poi', 'rev_near', 'building'].edge_index = torch.tensor([cols, rows], dtype=torch.long)

    # building->building edges via knn
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='auto').fit(b_coords)
    dists, idxs = nbrs.kneighbors(b_coords)
    b_rows, b_cols = [], []
    for i in range(len(b_coords)):
        for j in idxs[i]:
            b_rows.append(i)
            b_cols.append(int(j))
    data['building', 'adj', 'building'].edge_index = torch.tensor([b_rows, b_cols], dtype=torch.long)
    return data

# -------------------------
# Main entry
# -------------------------
def main(args):
    # read
    POIS_PATH = r"E:\University\guangzhou_dataset_full\guangzhou_POI.shp"
    BUILDINGS_PATH = r"E:\University\guangzhou_dataset_full\guangzhou_shp.shp"
    ROADS_PATH = r"E:\University\guangzhou_dataset_full\guangzhou_road.shp"

    # Use robust reader
    print(f"Reading POIs from: {POIS_PATH}")
    pois = safe_read_shp(POIS_PATH)
    print(f"Reading buildings from: {BUILDINGS_PATH}")
    buildings = safe_read_shp(BUILDINGS_PATH)
    print(f"Reading roads from: {ROADS_PATH}")
    roads = safe_read_shp(ROADS_PATH)
    # ensure CRS
    # if not set, assume WGS84 (user must ensure)
    if pois.crs is None or buildings.crs is None or roads.crs is None:
        raise RuntimeError("Please ensure all inputs have a valid CRS set (e.g., EPSG:4326).")

    # choose projected CRS if not supplied
    if args.proj is None:
        proj = determine_project_crs([pois, buildings, roads])
        print("Auto-chosen projection:", proj)
    else:
        proj = args.proj

    pois = pois.to_crs(proj)
    buildings = buildings.to_crs(proj)
    roads = roads.to_crs(proj)

    # normalize poi columns: ensure poi_class field exists; simplify names
    if 'poi_class' not in pois.columns:
        # try common column names
        candidates = ['class','category','type','poi_type']
        for c in candidates:
            if c in pois.columns:
                pois = pois.rename(columns={c: 'poi_class'})
                break
        if 'poi_class' not in pois.columns:
            raise RuntimeError("POI GeoDataFrame must have a 'poi_class' column (or rename one).")
    if 'importance' not in pois.columns:
        pois['importance'] = 1.0

    # add projected x,y for fast access
    pois['x'] = pois.geometry.x
    pois['y'] = pois.geometry.y
    buildings['centroid'] = buildings.geometry.centroid
    buildings['cx'] = buildings.centroid.x
    buildings['cy'] = buildings.centroid.y

    # 1. build blocks from roads
    print("Building street-block polygons from road network...")
    blocks = build_blocks_from_roads(roads)
    print(f"Num blocks: {len(blocks)}")

    # 2. assign POI & buildings to blocks
    print("Assigning POIs and buildings to blocks...")
    pois_s, buildings_s = compute_block_assignments(pois, buildings, blocks)

    # 3. classes set
    classes = sorted(pois_s['poi_class'].unique().tolist())
    print("Detected POI classes:", classes)

    # 4. compute block-level stats
    print("Computing block-level POI stats (local weights)...")
    block_stats = compute_block_poi_stats(pois_s, classes)

    # 5. global class stats & sigma
    print("Computing global class-level sigma and global weights...")
    w_global, sigma_global = compute_global_class_stats(pois_s, classes, alpha=args.alpha)

    # 6. optionally downsample dense blocks (not implemented automatically for all)
    # left as extension - user can call grid_downsample_pois_in_block for specific blocks

    # 7. compute per-building contributions (parallel)
    print("Computing building-level contributions and probability vectors...")
    building_probs = compute_building_contributions_parallel(
        buildings_s, pois_s, block_stats, sigma_global, w_global, classes,
        max_radius=args.max_radius, min_poi_per_block=args.min_poi_per_block,
        downsample_threshold=args.grid_downsample_threshold, grid_size=args.grid_size_downsample,
        n_jobs=args.n_jobs
    )

    # 8. spatial smoothing
    print("Applying spatial smoothing...")
    building_probs = spatial_smoothing(buildings_s, building_probs, args.smooth_radius, args.smooth_lambda, iters=args.smooth_iters)

    # 9. write output
    print("Writing outputs...")
    write_output(buildings_s, building_probs, classes, args.out)

    # 10. optional: construct hetero data & train GNN
    if args.train_gnn:
        if not TORCH_AVAILABLE:
            print("Torch / PyG not available - cannot train GNN. Skipping.")
        else:
            print("Constructing hetero data for PyG (NOTE: for demo; for large scale use sampling/partitioning).")
            data = build_hetero_data_for_pyg(buildings_s, pois_s, building_probs, classes)
            # Training loop omitted - user can implement task-specific training (BCE loss etc.)
            print("HeteroData constructed. Implement training loop as needed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building Function Prediction Pipeline (POI -> Building)")
    parser.add_argument("--pois", required=True, help="POI GeoJSON/GPKG with columns 'poi_class' (and optional 'importance')")
    parser.add_argument("--buildings", required=True, help="Buildings GeoJSON/GPKG (polygons).")
    parser.add_argument("--roads", required=True, help="Road line GeoJSON/GPKG for block generation.")
    parser.add_argument("--out", required=True, help="Output GeoPackage path for buildings with probs.")
    parser.add_argument("--proj", default=None, help="Projected CRS (EPSG string). If None, auto-select UTM.")
    parser.add_argument("--max_radius", type=float, default=DEFAULTS['max_radius'])
    parser.add_argument("--min_poi_per_block", type=int, default=DEFAULTS['min_poi_per_block'])
    parser.add_argument("--kappa", type=float, default=DEFAULTS['kappa'])
    parser.add_argument("--alpha", type=float, default=DEFAULTS['alpha'])
    parser.add_argument("--smooth_lambda", type=float, default=DEFAULTS['smooth_lambda'])
    parser.add_argument("--smooth_iters", type=int, default=DEFAULTS['smooth_iters'])
    parser.add_argument("--smooth_radius", type=float, default=DEFAULTS['smooth_radius'])
    parser.add_argument("--grid_downsample_threshold", type=int, default=DEFAULTS['grid_downsample_threshold'])
    parser.add_argument("--grid_size_downsample", type=float, default=DEFAULTS['grid_size_downsample'])
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel workers for contribution computation")
    parser.add_argument("--train_gnn", action='store_true', help="If set, attempt to create HeteroData for GNN training")
    args = parser.parse_args()

    main(args)
