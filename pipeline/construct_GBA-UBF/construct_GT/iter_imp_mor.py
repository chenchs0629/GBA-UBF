import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    # 回退：简单的无进度包装
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x
from shapely.geometry import Point, MultiPolygon
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 读取数据 ---
buildings = pd.read_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/labels_full_2_融合优先标签.csv")
buildings['geometry'] = buildings['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(buildings, geometry='geometry', crs="EPSG:4326")

# --- 2. 投影为平面坐标系（米） ---
gdf = gdf.to_crs("EPSG:3857")

# --- 3. 删除空几何 & 计算质心 ---
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
gdf['centroid'] = gdf.geometry.centroid
gdf = gdf[~gdf['centroid'].is_empty & gdf['centroid'].notnull()]

# --- 4. 计算形态学特征 ---
def calc_shape_features(geom):
    # 兼容 Polygon 与 MultiPolygon
    if geom is None or geom.is_empty:
        # 返回缺省值
        return pd.Series([0.0, 0.0, 0.0, 0, 1.0, 0.0], index=['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index'])

    if isinstance(geom, MultiPolygon):
        # 总面积与总周长使用所有子多边形的和
        area = sum(p.area for p in geom.geoms)
        perimeter = sum(p.length for p in geom.geoms)
        # 对于顶点数和外边界相关的特征，使用最大子多边形作为代表
        largest = max(geom.geoms, key=lambda p: p.area)
        vertices = len(largest.exterior.coords)
        minx, miny, maxx, maxy = geom.bounds
    else:
        area = geom.area
        perimeter = geom.length
        vertices = len(geom.exterior.coords)
        minx, miny, maxx, maxy = geom.bounds

    compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)  # 避免除零
    # elongation 使用边界盒的长宽比，避免除零
    dx = maxx - minx + 1e-6
    dy = maxy - miny + 1e-6
    elongation = max(dx, dy) / min(dx, dy)
    shape_index = perimeter / (2 * np.sqrt(np.pi * area + 1e-6))

    return pd.Series([area, perimeter, compactness, vertices, elongation, shape_index],
                     index=['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index'])

# 计算形态学特征时显示进度（若安装了 tqdm 则启用进度条）
if TQDM_AVAILABLE:
    # tqdm.pandas() 可以与 pandas apply 配合显示进度
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.pandas()
        gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']] = \
            gdf['geometry'].progress_apply(calc_shape_features)
    except Exception:
        gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']] = \
            gdf['geometry'].apply(calc_shape_features)
else:
    gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']] = \
        gdf['geometry'].apply(calc_shape_features)

# --- 5. 建立空间邻居索引 ---
coords = np.array([[pt.y, pt.x] for pt in gdf['centroid']])  # BallTree: (lat, lon)
tree = BallTree(coords, metric='euclidean')

# --- 6. 构造联合特征向量（空间 + 形态） ---
# 空间坐标缩放因子 beta，避免主导相似度
beta = 0.001
features = np.column_stack([
    gdf['centroid'].x.values * beta,
    gdf['centroid'].y.values * beta,
    gdf[['area', 'perimeter', 'compactness', 'vertices', 'elongation', 'shape_index']].values
])

# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- 7. 初始化标签 ---
labels = gdf['label'].astype(int).tolist()
new_labels = labels.copy()

# --- 8. 迭代优化 ---
num_iterations = 3
neighbor_k = 15
vote_threshold = 0.5  # 占比阈值

for _ in (range(num_iterations) if not TQDM_AVAILABLE else tqdm(range(num_iterations), desc='iterations')):
    updated_labels = []
    iterator = enumerate(coords)
    if TQDM_AVAILABLE:
        iterator = enumerate(tqdm(coords, desc='buildings'))
    for i, center in iterator:
        # 先取空间邻居
        dist, ind = tree.query([center], k=neighbor_k + 1)
        neighbor_ids = ind[0][1:]  # 排除自己

        # 取特征向量
        F_i = features_scaled[i].reshape(1, -1)
        F_neighbors = features_scaled[neighbor_ids]

        # 计算余弦相似度
        sims = cosine_similarity(F_i, F_neighbors)[0]

        # 按相似度加权投票
        neighbor_labels = [new_labels[j] for j in neighbor_ids]
        vote_df = pd.DataFrame({'label': neighbor_labels, 'sim': sims})
        vote_weights = vote_df.groupby('label')['sim'].sum()

        most_common_label = vote_weights.idxmax()
        majority_ratio = vote_weights.max() / vote_weights.sum()

        current_label = new_labels[i]
        if current_label != most_common_label and majority_ratio >= vote_threshold:
            updated_labels.append(most_common_label)
        else:
            updated_labels.append(current_label)

    new_labels = updated_labels

# --- 9. 结果保存 ---
gdf['label'] = new_labels
gdf.drop(columns=['centroid']).to_csv(
    "E:/University/dataset_guangzhou/广州研究区/研究区二/labels_spatial_morph_cos优化版.csv",
    index=False
)
