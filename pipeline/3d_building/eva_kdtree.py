import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
import rasterio.features
from tqdm import tqdm
from sklearn.neighbors import KDTree   # ★ 替换 KDE

# ---------- 1) 输入 ----------
bld_files = {
    1: r"E:/University/zhuhai_dataset_full/show_reslut/商业.shp",
    2: r"E:/University/zhuhai_dataset_full/show_reslut/住宅.shp",
    3: r"E:/University/zhuhai_dataset_full/show_reslut/公共服务.shp",
    4: r"E:/University/zhuhai_dataset_full/show_reslut/科技与工业.shp",
    5: r"E:/University/zhuhai_dataset_full/show_reslut/教育文化.shp", 
}

poi_files = {
    1: r"E:/University/zhuhai_dataset_full/show_reslut/商业POI.shp",
    2: r"E:/University/zhuhai_dataset_full/show_reslut/住宅POI.shp",
    3: r"E:/University/zhuhai_dataset_full/show_reslut/公共服务POI.shp",
    4: r"E:/University/zhuhai_dataset_full/show_reslut/科技与工业POI.shp",
    5: r"E:/University/zhuhai_dataset_full/show_reslut/教育文化POI.shp",
}

# ---------- 2) 合并建筑 ----------
blds = []
for c, f in tqdm(list(bld_files.items()), desc="读取建筑"):
    gdf = gpd.read_file(f)
    gdf["cls"] = c
    blds.append(gdf[["geometry", "cls"]])
bld = gpd.GeoDataFrame(pd.concat(blds, ignore_index=True), crs=blds[0].crs)

# ---------- 3) 读 POI ----------
target_crs = "EPSG:3857"
if bld.crs != target_crs:
    bld = bld.to_crs(target_crs)

poi_dict = {}
for c, f in tqdm(list(poi_files.items()), desc="读取POI"):
    gdf = gpd.read_file(f)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    poi_dict[c] = gdf

# ---------- 4) 栅格范围 ----------
total_bounds = bld.total_bounds
for gdf in poi_dict.values():
    tb = gdf.total_bounds
    total_bounds = [
        min(total_bounds[0], tb[0]),
        min(total_bounds[1], tb[1]),
        max(total_bounds[2], tb[2]),
        max(total_bounds[3], tb[3]),
    ]

minx, miny, maxx, maxy = total_bounds
res = 15.0
width  = int(np.ceil((maxx - minx) / res))
height = int(np.ceil((maxy - miny) / res))
transform = from_origin(minx, maxy, res, res)

# ---------- 5) 构建网格中心点 ----------
xs = minx + res * (np.arange(width) + 0.5)
ys = maxy - res * (np.arange(height) + 0.5)
grid_x, grid_y = np.meshgrid(xs, ys)
centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])   # (H*W,2)
Ngrid = centers.shape[0]

# ---------- 6) KD-Tree 密度计算（替换 KDE） ----------
bandwidth = 100.0
search_radius = 3 * bandwidth                 # 3σ 范围（99.7% 能量）

def kde_tree_density(points_xy, bw, sample_xy):
    """使用 KD-Tree + Gaussian 权重替代 KDE。

    优化策略：根据 POI 数量与栅格数量选择对较小集合进行迭代。
    - 如果 POI 数量 <= 栅格数量：在栅格点上建树，按 POI 逐个累加贡献（适合 POI 较少的情况）。
    - 否则：在 POI 上建树，对栅格分块查询并逐点计算（适合栅格较大、POI 较多的情况）；分块避免一次性占用过多内存。
    保持原始高斯权重计算不变。
    """
    if len(points_xy) == 0:
        return np.zeros(sample_xy.shape[0])

    n_points = points_xy.shape[0]
    n_samples = sample_xy.shape[0]
    bw2 = 2 * (bw ** 2)
    dens = np.zeros(n_samples, dtype=float)

    # 若 POI 数量较少，按 POI 累加到栅格（避免为每个栅格保存邻居列表）
    if n_points <= n_samples:
        tree_s = KDTree(sample_xy, leaf_size=40)
        for p in tqdm(points_xy, desc="按 POI 累加密度", leave=False):
            idxs = tree_s.query_radius([p], r=search_radius)[0]
            if idxs.size == 0:
                continue
            d2 = np.sum((sample_xy[idxs] - p) ** 2, axis=1)
            dens[idxs] += np.exp(-d2 / bw2)
        return dens

    # 否则按栅格分块查询 POI，并计算每个栅格点的密度（节省内存）
    tree_p = KDTree(points_xy, leaf_size=40)
    chunk_size = 200000
    for start in tqdm(range(0, n_samples, chunk_size), desc="按栅格分块查询密度", leave=False):
        end = min(start + chunk_size, n_samples)
        sub = sample_xy[start:end]
        ind_lists = tree_p.query_radius(sub, r=search_radius)
        for i, idxs in enumerate(ind_lists):
            if len(idxs) == 0:
                continue
            d2 = np.sum((points_xy[idxs] - sub[i]) ** 2, axis=1)
            dens[start + i] = np.sum(np.exp(-d2 / bw2))

    return dens

# ---------- 7) 逐类计算栅格密度 ----------
dens_stack = []

for c in tqdm([1,2,3,4,5], desc="KDTree 密度计算"):
    g = poi_dict[c].geometry

    # Fast numpy xy
    pts_xy = np.column_stack([g.x.values, g.y.values])

    dens = kde_tree_density(pts_xy, bandwidth, centers)
    dens = dens.reshape(height, width)
    dens_stack.append(dens)

dens_stack = np.stack(dens_stack, axis=0)   # (5, H, W)

# ---------- 8) 转成概率 ----------
eps = 1e-9
sum_d = dens_stack.sum(axis=0, keepdims=True) + eps
prob_stack = dens_stack / sum_d

# ---------- 9) Zonal 统计 ----------
def zonal_mean_probs(poly):
    mask_arr = rasterio.features.rasterize(
        [(poly, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0, all_touched=False,
        dtype="uint8"
    ).astype(bool)

    if mask_arr.sum() == 0:
        return np.full(5, np.nan)

    vals = prob_stack[:, mask_arr]  # (5, Npix)
    m = np.nanmean(vals, axis=1)
    s = m.sum()
    if s <= 0:
        return np.full(5, 1/5)
    return m / s

poi_probs = []
for geom in tqdm(bld.geometry, desc="建筑分区概率"):
    poi_probs.append(zonal_mean_probs(geom))
poi_probs = np.vstack(poi_probs)

# ---------- 10) 指标 ----------
y = bld["cls"].to_numpy()
N = len(y)

top1_pred = np.argmax(poi_probs, axis=1) + 1
top1_match = (top1_pred == y).astype(float)
top1_acc = np.nanmean(top1_match)

den = np.sqrt((poi_probs**2).sum(axis=1))
cos_sim = poi_probs[np.arange(N), y-1] / (den + 1e-12)
cos_mean = np.nanmean(cos_sim)

w = 0.5
bfvi = np.nanmean(w*top1_match + (1-w)*cos_sim)

print(f"Top1 Accuracy = {top1_acc:.4f}")
print(f"Cosine Similarity (mean) = {cos_mean:.4f}")
print(f"BFVI (w={w}) = {bfvi:.4f}")
