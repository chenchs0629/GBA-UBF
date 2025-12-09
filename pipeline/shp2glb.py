import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
import pyproj
from rasterio.sample import sample
from tqdm import tqdm


# -------------------------------------------------------
# 1) 读取建筑物和 DEM
# -------------------------------------------------------

shp_path = r"E:\University\zhuhai_dataset_full\output\building_class_ini\building_with_class3.shp"
dem_path = r"E:/your_folder/dem.tif"

gdf = gpd.read_file(shp_path)
dem = rasterio.open(dem_path)

# 高度字段
HEIGHT_FIELD = "Height"

# 坐标系统一：建筑物 CRS → DEM CRS
if gdf.crs != dem.crs:
    gdf = gdf.to_crs(dem.crs)


# -------------------------------------------------------
# 2) 从 DEM 采样得到建筑物底部高度
# -------------------------------------------------------

def get_dem_height(point):
    """ 使用 DEM 在点位采样得到高程 """
    for val in dem.sample([(point.x, point.y)]):
        return float(val)
    return 0.0


gdf["base_z"] = gdf.geometry.centroid.apply(get_dem_height)


# -------------------------------------------------------
# 3) 建筑物转 3D Mesh（高度 ×0.001，并加 DEM 高度）
# -------------------------------------------------------

def polygon_to_mesh(poly: Polygon, height: float, base_z: float):
    """
    将 Polygon extrude 成 mesh（带 DEM 高度抬升）
    """

    # 修正高度：例如毫米 → 米
    height_m = height * 0.001

    # trimesh 需要 polygon 必须简单化
    poly = poly.simplify(0.1, preserve_topology=True)

    try:
        mesh = trimesh.creation.extrude_polygon(poly, height_m)
    except:
        # 再试一次更强简化
        poly = poly.simplify(1, preserve_topology=True)
        mesh = trimesh.creation.extrude_polygon(poly, height_m)

    # 平移到 DEM 高度
    mesh.apply_translation([0, 0, base_z])

    return mesh


building_meshes = []

for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
    geom = row.geometry
    h = float(row[HEIGHT_FIELD])
    base_z = float(row["base_z"])

    if isinstance(geom, Polygon):
        building_meshes.append(polygon_to_mesh(geom, h, base_z))
    elif isinstance(geom, MultiPolygon):
        for poly in geom:
            building_meshes.append(polygon_to_mesh(poly, h, base_z))


# 合并所有建筑物 mesh
scene_buildings = trimesh.util.concatenate(building_meshes)

# 导出 GLB
scene_buildings.export("buildings.glb")
print("✔ buildings.glb 已生成")


# -------------------------------------------------------
# 4) 生成 DEM 网格 → terrain.glb
# -------------------------------------------------------

def dem_to_mesh(dem, downsample=4):
    """
    将 DEM 转换为 trimesh 的地形模型
    downsample：降低分辨率以减少面数
    """

    data = dem.read(1)

    # 下采样
    data_ds = data[::downsample, ::downsample]
    h, w = data_ds.shape

    # 网格坐标
    xs = np.arange(w)
    ys = np.arange(h)
    xx, yy = np.meshgrid(xs, ys)

    # 像素转真实坐标
    transform = dem.transform
    xs_real = transform.c + xx * transform.a
    ys_real = transform.f + yy * transform.e

    # 构造顶点
    vertices = np.vstack([
        xs_real.flatten(),
        ys_real.flatten(),
        data_ds.flatten()
    ]).T

    # 生成 faces（三角网格）
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


terrain_mesh = dem_to_mesh(dem, downsample=5)
terrain_mesh.export("terrain.glb")
print("✔ terrain.glb 已生成")
