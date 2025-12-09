import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm

#####用POI对建筑物的属性进行赋值

# --- 1. 读取数据 ---
buildings = pd.read_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/建筑/haizhu_2.csv")
pois = pd.read_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/百度POI_2/haizhu_2_POI_label.csv")

# --- 2. 转换geometry为shapely对象 ---
buildings['geometry'] = buildings['geometry'].apply(wkt.loads)
pois['geometry'] = pois['geometry'].apply(wkt.loads)

# --- 3. 转换为GeoDataFrame，设置WGS84投影 ---
gdf_buildings = gpd.GeoDataFrame(buildings, geometry='geometry', crs="EPSG:4326")
gdf_pois = gpd.GeoDataFrame(pois, geometry='geometry', crs="EPSG:4326")

# --- 4. 投影转换为平面坐标系（EPSG:3857） ---
gdf_buildings = gdf_buildings.to_crs("EPSG:3857")
gdf_pois = gdf_pois.to_crs("EPSG:3857")

# --- 5. 预处理：查找落入建筑内的POI ---
target_labels = {2, 3, 5}
matched_building_labels = pd.Series(index=gdf_buildings.index, dtype="float")

# 执行空间连接，找到POI落入哪些建筑
joined = gpd.sjoin(gdf_pois[gdf_pois['label'].isin(target_labels)], gdf_buildings, predicate='within')

# 为这些建筑赋标签（如果有多个，默认取第一个POI标签）
for idx, group in joined.groupby('index_right'):
    label = group.iloc[0]['label']  # 默认取第一条落入的POI的标签
    matched_building_labels.at[idx] = label

# --- 6. 分离已赋值 & 未赋值建筑 ---
assigned_idx = matched_building_labels.dropna().index
unassigned_idx = gdf_buildings.index.difference(assigned_idx)

# --- 7. 计算未赋值建筑的质心 ---
gdf_buildings['centroid'] = gdf_buildings.geometry.centroid
building_centroids = np.vstack(gdf_buildings.loc[unassigned_idx, 'centroid'].apply(lambda pt: (pt.y, pt.x)))

# --- 8. 提取POI坐标和标签 ---
poi_coords = np.vstack(gdf_pois.geometry.apply(lambda pt: (pt.y, pt.x)))
poi_labels = gdf_pois['label'].apply(lambda x: [int(i) for i in str(x)]).to_list()

# --- 9. 构建BallTree ---
ball_tree = BallTree(poi_coords, metric='euclidean')

# --- 10. 标签权重设置 ---
#  ！！！
label_weights = {1: 0.2, 2: 2, 3: 1.5, 4: 1.5, 5: 2.2}  ######需要实验并手动调整权重
all_labels = [1, 2, 3, 4, 5]
#  ！！！

final_labels = matched_building_labels.tolist()  # 初始化为已有赋值的标签

# --- 11. 对未赋值建筑使用最近POI + 周边POI融合判定 ---
print("开始处理未匹配的建筑标签...")
for i, b_idx in enumerate(tqdm(unassigned_idx)):
    center = building_centroids[i]
    dist, ind = ball_tree.query([center], k=10)

    # 最近POI标签打分
    near_labels = poi_labels[ind[0][0]]
    near_score = {l: 0 for l in all_labels}
    for l in near_labels:
        near_score[l] += 1.0

    # 周边10个POI权重得分
    weight_score = {l: 0 for l in all_labels}
    for j in ind[0]:
        for l in poi_labels[j]:
            weight_score[l] += label_weights.get(l, 0)

    # 综合打分（最近标签60%，周边得分40%）     
    #  ！！！
    # ######需要实验并手动调整权重
    total_score = {l: 0.6 * near_score[l] + 0.4 * weight_score[l] for l in all_labels}
    final_label = max(total_score.items(), key=lambda x: x[1])[0]
    final_labels[b_idx] = final_label  # 赋值
    #  ！！！

# --- 12. 添加最终标签并保存 ---
gdf_buildings['label'] = final_labels
output_df = gdf_buildings.drop(columns=['centroid'])
output_df.to_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/labels_full_1_融合优先标签.csv", index=False)
print("✅ 处理完成，结果已保存。")
