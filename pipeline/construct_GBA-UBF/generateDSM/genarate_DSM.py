import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

# 文件路径
dem_path = r"E:\University\数据集汇总\佛山\aera2\foshan_dem2_filled.tif"
shp_path = r"E:\University\数据集汇总\佛山\aera2\bui2\bui2export.shp"
output_dsm_path = r"E:\University\数据集汇总\佛山\aera2\foshan_dsm.tif"

# 读取 DEM
with rasterio.open(dem_path) as dem_src:
    dem_data = dem_src.read(1)
    dem_meta = dem_src.meta.copy()
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs
    dem_shape = dem_data.shape

# 读取建筑物矢量数据（含高度）
buildings = gpd.read_file(shp_path)

# 确保矢量和栅格坐标系一致
if buildings.crs != dem_crs:
    buildings = buildings.to_crs(dem_crs)

# 提取(几何, 高度值)对
shapes = ((geom, height) for geom, height in zip(buildings.geometry, buildings["Height_1"]))

# 栅格化建筑物高度
building_height_raster = rasterize(
    shapes=shapes,
    out_shape=dem_shape,
    transform=dem_transform,
    fill=0,
    all_touched=True,
    dtype='float32'
)

# 生成 DSM：DEM + 建筑物高度（仅在建筑物处有值）
dsm_data = dem_data + building_height_raster

# 保存 DSM
dem_meta.update({
    "dtype": "float32",
    "compress": "lzw"
})

with rasterio.open(output_dsm_path, "w", **dem_meta) as dst:
    dst.write(dsm_data, 1)

print("DSM 生成完毕，保存至：", output_dsm_path)
