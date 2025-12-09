import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from sklearn.neighbors import KernelDensity
import rasterio.features
from tqdm import tqdm
# ---------- 1) 输入数据 ----------

"""
    1: r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/商业.shp",
    2: r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/住宅.shp",
    3: r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/公共服务.shp",
    4: r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/科技与工业.shp",
    5: r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/教育文化.shp",
    """
# 建筑物五类
bld_files = {
    1: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\商业.shp",
    2: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\住宅.shp",
    3: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\公共服务.shp",
    4: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\科技与工业.shp",
    5: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\教育文化.shp", 
}

# POI 五类
poi_files = {
    1: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\商业_POI.shp",
    2: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\住宅_POI.shp",
    3: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\公共服务_POI.shp",
    4: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\科技与工业_POI.shp",
    5: r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai2_subset\教育文化_POI.shp",
}

# ---------- 2) 合并建筑数据 ----------
blds = []
for c, f in tqdm(list(bld_files.items()), desc="读取建筑", total=len(bld_files)):
    gdf = gpd.read_file(f)
    gdf["cls"] = c
    blds.append(gdf[["geometry", "cls"]])
bld = gpd.GeoDataFrame(pd.concat(blds, ignore_index=True), crs=blds[0].crs)

# ---------- 3) 读 POI 并投影 ----------
target_crs = "EPSG:3857"  # 统一投影，按需要修改
if bld.crs != target_crs:
    bld = bld.to_crs(target_crs)

poi_dict = {}
for c, f in tqdm(list(poi_files.items()), desc="读取POI", total=len(poi_files)):
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
res = 20  # 栅格分辨率（米）  
width  = int(np.ceil((maxx - minx) / res))
height = int(np.ceil((maxy - miny) / res))
transform = from_origin(minx, maxy, res, res)

# ---------- 5) KDE 生成密度栅格 ----------
def kde_density(points_xy, bw, sample_xy):
    if len(points_xy) == 0:
        return np.zeros(len(sample_xy))

    kde = KernelDensity(bandwidth=bw, kernel="gaussian", metric="euclidean")
    kde.fit(points_xy)

    # 对 sample_xy 分块计算 score_samples，避免一次性占用过多内存/时间
    n = sample_xy.shape[0]
    # 每块大小可调整；200k 对大多数机器较为稳妥
    chunk_size = 200000
    res = np.empty(n, dtype=float)
    if n <= chunk_size:
        log_d = kde.score_samples(sample_xy)
        res[:] = np.exp(log_d)
        return res

    # 分块并显示进度
    for start in tqdm(range(0, n, chunk_size), desc="KDE 分块", leave=False):
        end = min(start + chunk_size, n)
        log_d = kde.score_samples(sample_xy[start:end])
        res[start:end] = np.exp(log_d)

    return res

xs = minx + res * (np.arange(width) + 0.5)
ys = maxy - res * (np.arange(height) + 0.5)
grid_x, grid_y = np.meshgrid(xs, ys)
centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])

bandwidth = 100.0  # KDE 带宽
dens_stack = []

for c in tqdm([1,2,3,4,5], desc="KDE 生成密度"):
    pts = poi_dict[c].geometry
    pts_xy = np.array([[p.centroid.x, p.centroid.y] for p in pts if p is not None and not p.is_empty])
    dens = kde_density(pts_xy, bandwidth, centers).reshape(height, width)
    dens_stack.append(dens)

dens_stack = np.stack(dens_stack, axis=0)  # (5, H, W)

# ---------- 6) 转为概率 ----------
eps = 1e-9
sum_d = dens_stack.sum(axis=0, keepdims=True) + eps
prob_stack = dens_stack / sum_d

# ---------- 7) 建筑分区概率 ----------
def zonal_mean_probs(poly_geom):
    mask_arr = rasterio.features.rasterize(
        [(poly_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0, all_touched=False, dtype="uint8"
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
for geom in tqdm(bld.geometry, desc="建筑分区概率计算"):
    poi_probs.append(zonal_mean_probs(geom))
poi_probs = np.vstack(poi_probs)

# ---------- 8) 指标 ----------
y = bld["cls"].to_numpy()
N = len(y)

# Top-1
top1_pred = np.argmax(poi_probs, axis=1) + 1
top1_match = (top1_pred == y).astype(float)
top1_acc = np.nanmean(top1_match)

# Cosine
den = np.sqrt((poi_probs**2).sum(axis=1))
cos_sim = poi_probs[np.arange(N), y-1] / (den + 1e-12)
cos_mean = np.nanmean(cos_sim)

# BFVI
w = 0.5
bfvi = np.nanmean(w*top1_match + (1-w)*cos_sim)

print(f"Top1 Accuracy = {top1_acc:.4f}")
print(f"Cosine Similarity (mean) = {cos_mean:.4f}")
print(f"BFVI (w={w}) = {bfvi:.4f}")
