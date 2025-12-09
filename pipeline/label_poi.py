import geopandas as gpd
import pandas as pd

# 1. 读取shp文件
shp = gpd.read_file("C:/Users/chenchs/Desktop/POI_ori/中山POI数据11.1/POI_clip.shp")

# 2. 读取csv文件
csv = pd.read_csv("C:/Users/chenchs/Desktop/POI_ori/中山POI数据11.1/POI_clip_Label.csv")

# 3. 检查两者共有的对应字段，比如"id"
print("Shapefile字段：", shp.columns)
print("CSV字段：", csv.columns)

# 假设共有字段名都叫"id"
# 4. 合并（类似SQL左连接）
merged = shp.merge(csv[['uid', 'label']], on='uid', how='left')

# 5. 检查结果
print(merged.head())

# 6. 保存新的shapefile
merged.to_file("C:/Users/chenchs/Desktop/POI_ori/中山POI数据11.1/中山POI_Label.shp", encoding='utf-8')
