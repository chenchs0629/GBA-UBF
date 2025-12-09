import geopandas as gpd

#shp_path = "E:/University/dataset_guangzhou/广州研究区/研究区二/百度POI_2/haizhu_2_POI.shp"
shp_path = "E:/University/guangzhou_dataset_full/eva/graph_guanzhou1.shp"
csv_path = "E:/University/guangzhou_dataset_full/eva/graph_guanzhou1_field.csv"

# 读取 SHP 文件python
try:
    gdf = gpd.read_file(shp_path, encoding='GBK')
    print("✅ 使用 GBK 编码读取成功！")
    
except UnicodeDecodeError:
    # 如果 GBK 失败，尝试更通用的中文编码 GB18030
    print("尝试 GBK 失败，尝试 GB18030 编码...")
    gdf = gpd.read_file(shp_path, encoding='GB18030')
    print("✅ 使用 GB18030 编码读取成功！")

# 检查是否读取成功
print("✅ 读取记录数：", len(gdf))
print("✅ 字段名：", gdf.columns.tolist())
print(gdf.head())  # 预览前几行数据

# 转换 geometry 并导出
gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.wkt if geom else None)
# 导出 CSV 时，使用 'utf-8-sig' 确保中文在 Excel 中显示正确
gdf.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 数据已成功处理并导出到 {csv_path}")
