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

# ---------- 3) 读 POI 并投影 ----------
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
res = 20.0
width  = int(np.ceil((maxx - minx) / res))
height = int(np.ceil((maxy - miny) / res))
transform = from_origin(minx, maxy, res, res)

# ---------- 5) KDE 优化：一次 fit，每类 sample_weight ----------
# 合并全部 POI
all_pts = []
weights = []

for c in range(1, 6):
    g = poi_dict[c].geometry
    xy = np.column_stack([g.x.values, g.y.values])
    all_pts.append(xy)
    weights.append(np.full(len(xy), c))  # 用 c 区分类别

all_pts = np.vstack(all_pts)
weights = np.concatenate(weights)

bandwidth = 100.0

# 单次 fit（极大加速）
kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(all_pts)

# 网格中心
xs = minx + res * (np.arange(width) + 0.5)
ys = maxy - res * (np.arange(height) + 0.5)
grid_x, grid_y = np.meshgrid(xs, ys)
centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])

# 自适应 chunk
chunk_size = max(50000, int(2e7 / centers.shape[1]))  # 动态设置，保证不爆内存
N = centers.shape[0]

log_all = np.empty(N)
for start in tqdm(range(0, N, chunk_size), desc="KDE 分块", leave=True):
    end = min(start + chunk_size, N)
    log_all[start:end] = kde.score_samples(centers[start:end])

dens_total = np.exp(log_all).reshape(height, width)

# ---------- 逐类密度（矫正：逐类 normalizer） ----------
dens_stack = []

for c in range(1, 6):
    mask_c = (weights == c)
    kde_c = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(all_pts[mask_c])
    log_c = np.empty(N)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        log_c[s:e] = kde_c.score_samples(centers[s:e])
    dens_stack.append(np.exp(log_c).reshape(height, width))

dens_stack = np.stack(dens_stack, axis=0)  # (5, H, W)

# ---------- 6) 概率 ----------
eps = 1e-9
sum_d = dens_stack.sum(axis=0, keepdims=True) + eps
prob_stack = dens_stack / sum_d

# ---------- 7) 建筑 zonal mean ----------
def zonal_mean_probs(poly):
    # 先裁剪 bounding box，减少 rasterize 范围
    minx, miny, maxx, maxy = poly.bounds
    row_start = max(0, int((transform.f - maxy) / res))
    row_end   = min(height, int((transform.f - miny) / res))
    col_start = max(0, int((minx - transform.c) / res))
    col_end   = min(width, int((maxx - transform.c) / res))

    if row_start >= row_end or col_start >= col_end:
        return np.full(5, np.nan)

    sub_transform = rasterio.transform.from_origin(
        minx=transform.c + col_start * res,
        maxy=transform.f - row_start * res,
        xsize=res, ysize=res
    )

    mask_arr = rasterio.features.rasterize(
        [(poly, 1)],
        out_shape=(row_end - row_start, col_end - col_start),
        transform=sub_transform,
        fill=0, all_touched=False
    ).astype(bool)

    if mask_arr.sum() == 0:
        return np.full(5, np.nan)

    vals = prob_stack[:, row_start:row_end, col_start:col_end][:, mask_arr]
    m = np.nanmean(vals, axis=1)
    s = m.sum()
    if s <= 0:
        return np.full(5, 1/5)
    return m / s

poi_probs = np.vstack([zonal_mean_probs(g) for g in tqdm(bld.geometry, desc="建筑分区概率")] )

# ---------- 8) 指标 ----------
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
