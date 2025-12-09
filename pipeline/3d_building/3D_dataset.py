import os
import geopandas as gpd
import pydeck as pdk

# 五个文件路径（替换成你的实际文件名）
files = {
    "商业": "E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/商业.shp",
    "住宅": "E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/住宅.shp",
    "公共服务": "E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/公共服务.shp",
    "科技与工业": "E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/科技与工业区.shp",
    "文化教育用地": "E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/教育文化.shp"
}

# 自定义颜色（RGB）
color_map = {
    "商业": [255, 0, 0],
    "住宅": [0, 255, 0],
    "公共服务": [0, 0, 255],
    "科技与工业": [255, 165, 0],
    "文化教育用地": [128, 0, 128]
}

layers = []
mid_lon_list = []
mid_lat_list = []

for category, filepath in files.items():
    gdf = gpd.read_file(filepath)

    # 统一坐标系到 WGS84
    gdf = gdf.to_crs(epsg=4326)
    gdf["category"] = category
    gdf["color"] = [color_map[category]] * len(gdf)

    # 自动选择高度字段
    height_field = None
    for candidate in ["Height", "Height_1", "高度", "建筑高度"]:
        if candidate in gdf.columns:
            height_field = candidate
            break
    if height_field is None:
        raise ValueError(f"{filepath} 没有高度字段，请检查字段名！字段列表: {gdf.columns}")

    # 计算质心（需要投影到米制再转回去）
    gdf_proj = gdf.to_crs(epsg=3857)
    centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    mid_lon_list.extend(centroids.x)
    mid_lat_list.extend(centroids.y)

    # 构造 Pydeck Layer
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf,
        extruded=True,
        get_elevation=height_field,  # ✅ 自动选择高度字段
        get_fill_color="color",
        elevation_scale=1,
        pickable=True,
    )
    layers.append(layer)

# 计算地图中心
mid_lon = sum(mid_lon_list) / len(mid_lon_list)
mid_lat = sum(mid_lat_list) / len(mid_lat_list)

view_state = pdk.ViewState(
    latitude=mid_lat,
    longitude=mid_lon,
    zoom=15,
    pitch=45,
    bearing=0
)

# 生成 Deck 对象
r = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v9",
    tooltip={"text": "{category}\n高度: {" + height_field + "}"}
)

# 保存到脚本所在路径
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "3d_buildings.html")
r.to_html(output_path, notebook_display=False)

print(f"✅ 3D 可视化结果已保存到: {output_path}")
