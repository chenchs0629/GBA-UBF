import geopandas as gpd
import pydeck as pdk
import folium
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import box
import pandas as pd

# ========== 1. 读取所有建筑物 ========== 
filepaths = [
    r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/商业.shp",
    r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/住宅.shp",
    r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/公共服务.shp",
    r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/科技与工业区.shp",
    r"E:/University/dataset_guangzhou/广州研究区/研究区一/final图层/教育文化.shp"
]

gdfs = []
for filepath in filepaths:
    gdf = gpd.read_file(filepath)
    # 兼容不同字段名，统一高度列
    height_field = None
    for candidate in ["建筑高度", "Height", "Height_1"]:
        if candidate in gdf.columns:
            height_field = candidate
            break
    if not height_field:
        raise ValueError(f"{filepath} 没有找到高度字段，请检查！")
    gdf = gdf.rename(columns={height_field: "height"})
    gdfs.append(gdf[["geometry", "height"]])

all_buildings = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

# ========== 2. 计算研究区范围 ==========
bounds = all_buildings.total_bounds  # [minx, miny, maxx, maxy]
print("研究区范围：", bounds)
study_area = box(*bounds)
study_gdf = gpd.GeoDataFrame(geometry=[study_area], crs=all_buildings.crs)

# ========== 3. 构造 pydeck 图层 ==========
all_buildings = all_buildings.to_crs(epsg=4326)  # 转换为WGS84坐标系
study_gdf = study_gdf.to_crs(epsg=4326)

layer = pdk.Layer(
    "PolygonLayer",
    all_buildings,  
    get_polygon="geometry.coordinates",
    get_elevation="height",
    get_fill_color="[180, 0, 200, 140]",
    extruded=True,
    pickable=True,
    wireframe=True
)

# ========== 4. 设置视角：缩放到建筑物范围 ==========
minx, miny, maxx, maxy = all_buildings.total_bounds
center = [(miny + maxy) / 2, (minx + maxx) / 2]  # 中心点

view_state = pdk.ViewState(
    latitude=center[0],
    longitude=center[1],
    zoom=16,
    pitch=45,
    bearing=0
)

# ========== 5. 生成地图 (只显示研究区范围的 OSM) ==========
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v10",
    tooltip={"text": "高度: {height}m"}
)

# 输出 html
r.to_html("E:/University/dataset_paper/3d_buildings_clipped.html", notebook_display=False)
print("✅ 已生成 3d_buildings_clipped.html，可以打开查看。")
