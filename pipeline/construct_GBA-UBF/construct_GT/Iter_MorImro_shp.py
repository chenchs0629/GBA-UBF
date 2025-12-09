import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
import numpy as np
from shapely import wkt
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x

from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. 读取 shp 文件
# =========================================================
input_shp = r"E:/University/zhuhai_dataset_full/output/building_class_ini/building_with_class2.shp"
output_shp = r"E:/University/zhuhai_dataset_full/output/building_class_ini/building_with_class3.shp"

gdf = gpd.read_file(input_shp)

# 要求 label 字段必须存在
if "dominant_c" not in gdf.columns:
    raise ValueError("输入的 SHP 必须包含字段 'dominant_c'！")
if "dominant_c" not in gdf.columns:
    raise ValueError("输入的 SHP 必须包含字段 'dominant_c'！")

# 原始 CRS 若未知，建议你补充一个
if gdf.crs is None:
    print("⚠ 输入 shp 没有 CRS，默认使用 EPSG:4326")
    gdf = gdf.set_crs(epsg=4326)

# 转换为米制
gdf = gdf.to_crs(epsg=3857)


# =========================================================
# 2. 删除空几何并计算质心
# =========================================================
gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)]
gdf["centroid"] = gdf.geometry.centroid


# =========================================================
# 3. 计算形态学特征
# =========================================================
def calc_shape_features(geom):
    if geom is None or geom.is_empty:
        return pd.Series([0, 0, 0, 0, 0, 0],
                         index=['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index'])

    if isinstance(geom, MultiPolygon):
        area = sum(p.area for p in geom.geoms)
        perimeter = sum(p.length for p in geom.geoms)
        largest = max(geom.geoms, key=lambda p: p.area)
        vertices = len(largest.exterior.coords)
        minx, miny, maxx, maxy = geom.bounds
    else:
        area = geom.area
        perimeter = geom.length
        vertices = len(geom.exterior.coords)
        minx, miny, maxx, maxy = geom.bounds

    compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    dx = maxx - minx + 1e-6
    dy = maxy - miny + 1e-6
    elongation = max(dx, dy) / min(dx, dy)
    shape_index = perimeter / (2 * np.sqrt(np.pi * area + 1e-6))

    return pd.Series(
        [area, perimeter, compactness, vertices, elongation, shape_index],
        index=['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']
    )


if TQDM_AVAILABLE:
    from tqdm import tqdm as _tqdm
    _tqdm.pandas()
    gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']] = \
        gdf['geometry'].progress_apply(calc_shape_features)
else:
    gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']] = \
        gdf['geometry'].apply(calc_shape_features)


# =========================================================
# 4. 构造空间邻近索引
# =========================================================
coords = np.array([[pt.y, pt.x] for pt in gdf["centroid"]])
tree = BallTree(coords, metric='euclidean')


# =========================================================
# 5. 构造联合特征：空间 + 形态
# =========================================================
beta = 0.001  # 坐标缩放避免主导权重

features = np.column_stack([
    gdf['centroid'].x.values * beta,
    gdf['centroid'].y.values * beta,
    gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']].values
])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# =========================================================
# 6. 标签优化迭代
# =========================================================
# =========================================================
# 6. 标签优化迭代
# =========================================================
# 从输入的 `dominant_c` 列读取初始标签，遇到缺失值时用 -1 填充并转为 int
labels = gdf["dominant_c"].fillna(-1).astype(int).tolist()
new_labels = labels.copy()

num_iterations = 3
neighbor_k = 15
vote_threshold = 0.5

for _ in (range(num_iterations) if not TQDM_AVAILABLE else tqdm(range(num_iterations), desc="iterations")):

    updated_labels = []

    iterator = enumerate(coords)
    if TQDM_AVAILABLE:
        iterator = enumerate(tqdm(coords, desc="buildings"))

    for i, center in iterator:

        # 空间邻居
        dist, ind = tree.query([center], k=neighbor_k + 1)
        neighbor_ids = ind[0][1:]

        # 相似度
        F_i = features_scaled[i].reshape(1, -1)
        F_neighbors = features_scaled[neighbor_ids]
        sims = cosine_similarity(F_i, F_neighbors)[0]

        # 邻居投票
        neighbor_labels = [new_labels[j] for j in neighbor_ids]
        vote_df = pd.DataFrame({'label': neighbor_labels, 'sim': sims})
        vote_weights = vote_df.groupby('label')['sim'].sum()

        most_common = vote_weights.idxmax()
        majority_ratio = vote_weights.max() / vote_weights.sum()

        current = new_labels[i]
        if current != most_common and majority_ratio >= vote_threshold:
            updated_labels.append(most_common)
        else:
            updated_labels.append(current)

    new_labels = updated_labels


# =========================================================
# 7. 输出为 Shapefile
# =========================================================
# =========================================================
# 7. 输出为 Shapefile（将优化结果写回 `dominant_c` 列）
# =========================================================
gdf["dominant_c"] = new_labels
gdf = gdf.drop(columns=["centroid"], errors="ignore")

gdf.to_file(output_shp, driver="ESRI Shapefile")
print("👉 已成功输出优化后的建筑类别 Shapefile：", output_shp)
