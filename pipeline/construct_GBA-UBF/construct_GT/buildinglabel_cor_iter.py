import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree

####循环迭代优化建筑物标签，增加不同区域建筑的集聚程度

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

# --- 4. 建立 BallTree ---
coords = np.array([[pt.y, pt.x] for pt in gdf['centroid']])
tree = BallTree(coords, metric='euclidean')

# --- 5. 初始化标签 ---
labels = gdf['label'].astype(int).tolist()
new_labels = labels.copy()

# --- 6. 空间标签聚集优化：迭代 N 轮 ---
num_iterations = 3
neighbor_k = 15
vote_threshold = 0.5  # 超过一半邻居标签相同才替换

for _ in range(num_iterations):
    updated_labels = []
    for i, center in enumerate(coords):
        dist, ind = tree.query([center], k=neighbor_k + 1)
        neighbor_ids = ind[0][1:]  # 排除自己
        neighbor_labels = [new_labels[j] for j in neighbor_ids]

        label_counts = pd.Series(neighbor_labels).value_counts()
        most_common_label = label_counts.idxmax()
        majority_ratio = label_counts.max() / len(neighbor_labels)

        current_label = new_labels[i]
        if current_label != most_common_label and majority_ratio >= vote_threshold:
            updated_labels.append(most_common_label)
        else:
            updated_labels.append(current_label)

    new_labels = updated_labels

# --- 7. 结果保存 ---
gdf['label'] = new_labels
gdf.drop(columns=['centroid']).to_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/labels_spatial_融合优先标签_聚集优化_迭代版.csv", index=False)
