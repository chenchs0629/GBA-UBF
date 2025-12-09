import geopandas as gpd
import json

# ====== 1. 读取建筑物 Shapefile ======
input_shp = r"E:\University\zhuhai_dataset_full\show_reslut\科技与工业.shp"   # 修改为你的 shp 路径
gdf = gpd.read_file(input_shp)

# ====== 2. 确保坐标系为 WGS84（Kepler 必须用经纬度） ======
if gdf.crs is None:
    raise ValueError("输入 Shapefile 没有坐标系信息，请检查 .prj 文件！")

gdf = gdf.to_crs(epsg=4326)

# ====== 3. 自动识别建筑高度字段 ======
height_field = None
for name in ["height", "Height", "height_m", "建筑高度", "H", "HEIGHT"]:
    if name in gdf.columns:
        height_field = name
        break

if height_field is None:
    print("⚠ 未找到高度字段，将不包含高度信息。")
else:
    print(f"✔ 使用高度字段: {height_field}")

# ====== 4. 导出为 GeoJSON ======
output_geojson = r"E:\University\zhuhai_dataset_full\show_reslut\科技与工业.geojson"   # 输出路径

# 保留所有字段（Kepler.gl 会自动识别高度）
gdf.to_file(output_geojson, driver="GeoJSON")

print("🎉 转换完成！已导出为:", output_geojson)
