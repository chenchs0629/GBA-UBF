import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from pyproj import CRS
import trimesh
import os

# ============================
# 修改参数
# ============================
input_path = r"E:\University\zhuhai_dataset_full\output\building_class_ini\building_with_class3.shp"
height_field = "Height"
output_path = r"E:\University\课件讲义作业_大三上\CG计算机图形学\CG大作业\buildings2.glb"

# ↓↓↓ 请注意：这是关键。模型会被移动到 Blender 原点附近
COORD_SCALE = 0.001   # 把米缩小1000倍，让城市模型在Blender里更好查看
# ============================


def ensure_projected(gdf):
    """确保是米为单位的投影坐标系"""
    crs = CRS.from_user_input(gdf.crs)
    if crs.is_projected:
        return gdf

    centroid = gdf.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone

    print("自动投影到 UTM:", epsg)
    return gdf.to_crs(epsg)


def extrude_polygon_simple(poly: Polygon, height: float, offset):
    """ 手动挤出，不依赖 triangulation """
    if poly.is_empty:
        return None

    exterior = np.array(poly.exterior.coords)

    # 坐标平移并缩放
    xy = (exterior - offset) * COORD_SCALE

    bottom = np.column_stack([xy, np.zeros(len(exterior))])
    top    = np.column_stack([xy, np.ones(len(exterior)) * height])

    vertices = np.vstack([bottom, top])
    faces = []
    n = len(exterior)

    # 顶面
    for i in range(1, n - 1):
        faces.append([n + 0, n + i, n + i + 1])

    # 底面
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])

    # 侧面
    for i in range(n - 1):
        a, b = i, i + 1
        a_top, b_top = a + n, b + n
        faces.append([a, b, a_top])
        faces.append([b, b_top, a_top])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def extrude_multipolygon(mp: MultiPolygon, height: float, offset):
    meshes = []
    for poly in mp.geoms:
        mesh = extrude_polygon_simple(poly, height, offset)
        if mesh is not None:
            meshes.append(mesh)
    if meshes:
        return trimesh.util.concatenate(meshes)
    return None


# ============================
# 主流程
# ============================

print("读取数据:", input_path)
gdf = gpd.read_file(input_path)
gdf = gdf[~gdf.geometry.is_empty].copy()

if height_field not in gdf.columns:
    raise ValueError(f"字段 {height_field} 不存在！")

gdf = ensure_projected(gdf)

heights = gdf[height_field].astype(float).fillna(0)

# 关键：获取所有建筑的最小 XY（用于整体平移）
minx, miny, _, _ = gdf.total_bounds
offset = np.array([minx, miny])

meshes = []
for geom, h in zip(gdf.geometry, heights):
    if h <= 0:
        continue

    if isinstance(geom, Polygon):
        mesh = extrude_polygon_simple(geom, h, offset)
    elif isinstance(geom, MultiPolygon):
        mesh = extrude_multipolygon(geom, h, offset)
    else:
        continue

    if mesh:
        meshes.append(mesh)

print("生成 mesh:", len(meshes))

scene = trimesh.Scene(trimesh.util.concatenate(meshes))

# 输出 glb
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print("导出:", output_path)
with open(output_path, "wb") as f:
    f.write(scene.export("glb"))

print("完成！Blender 中一定能看到建筑物。")
