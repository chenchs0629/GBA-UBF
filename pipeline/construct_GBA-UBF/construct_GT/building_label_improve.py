import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from tqdm import tqdm

# --- 工具函数：安全读取CSV ---
def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='gbk')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='utf-8')

# --- 1. 准备：定义POI信息（顺序是 商场→小区→高校） ---
poi_info = [
    {"file": "E:/University/dataset_guangzhou/广州研究区/研究区二/大POI/广州商场_2.csv", "buffer": 50, "label_value": 2},
    {"file": "E:/University/dataset_guangzhou/广州研究区/研究区二/大POI/广州小区_2.csv", "buffer": 30, "label_value": 1},
    {"file": "E:/University/dataset_guangzhou/广州研究区/研究区二/大POI/广州高校_2.csv", "buffer": 50, "label_value": 5},
]

# --- 2. 读取建筑物数据 ---
building_df = safe_read_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/labels_spatial_融合优先标签_聚集优化_迭代版.csv")
building_df['geometry'] = building_df['geometry'].apply(wkt.loads)
gdf_building = gpd.GeoDataFrame(building_df, geometry='geometry', crs="EPSG:4326")
gdf_building = gdf_building.to_crs("EPSG:3857")

# --- 3. 依次处理POI数据 ---
for poi in poi_info:
    print(f"正在处理：{poi['file']} ...")
    # 安全读取POI数据
    poi_df = safe_read_csv(poi["file"])
    
    # 判断POI数据格式
    if 'geometry' in poi_df.columns:
        poi_df['geometry'] = poi_df['geometry'].apply(wkt.loads)
    else:
        poi_df['geometry'] = poi_df.apply(lambda row: Point(row['lon_84'], row['lat_84']), axis=1)
    
    gdf_poi = gpd.GeoDataFrame(poi_df, geometry='geometry', crs="EPSG:4326")
    gdf_poi = gdf_poi.to_crs("EPSG:3857")
    
    # 创建缓冲区
    gdf_poi['buffer'] = gdf_poi.geometry.buffer(poi["buffer"])
    gdf_buffer = gdf_poi[['buffer']].set_geometry('buffer')
    
    # 对建筑物逐个判断是否在当前POI缓冲区内
    for idx, building in tqdm(gdf_building.iterrows(), total=len(gdf_building)):
        if gdf_buffer.intersects(building.geometry).any():
            gdf_building.at[idx, 'label'] = poi["label_value"]  # 直接覆盖label字段

# --- 4. 保存最终结果 ---
gdf_building.to_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/建筑物_POI缓冲_label最终版.csv", index=False, encoding='gbk')
print("✅ 全部处理完成，最终结果已保存到：建筑物_POI缓冲_label最终版.csv")
