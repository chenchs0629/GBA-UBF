import geopandas as gpd

#shp_path = "E:/University/dataset_guangzhou/广州研究区/研究区二/百度POI_2/haizhu_2_POI.shp"
shp_path = "E:/University/dataset_guangzhou/广州研究区/研究区一/大POI/广州高校_1.shp"
csv_path = "E:/University/dataset_guangzhou/广州研究区/研究区一/大POI/广州高校_1.csv"

# 读取 SHP 文件python
gdf = gpd.read_file(shp_path)

# 检查是否读取成功
print("✅ 读取记录数：", len(gdf))
print("✅ 字段名：", gdf.columns.tolist())
print(gdf.head())  # 预览前几行数据

# 转换 geometry 并导出
gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.wkt if geom else None)
gdf.to_csv(csv_path, index=False, encoding='utf-8-sig')
