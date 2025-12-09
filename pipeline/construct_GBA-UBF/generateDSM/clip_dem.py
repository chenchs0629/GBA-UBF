import geopandas as gpd
import rasterio
from rasterio.mask import mask

# ------------------------------
# 输入路径（请自行修改）
# ------------------------------
dem_path = r"C:\Users\chenchs\Desktop\广东\广东.tif"          # DEM 文件
shp_path = r"E:\University\zhuhai_dataset_full\zhihai_distract\zhuhai_dis.shp" # 市范围 shp
out_tif = r"C:\Users\chenchs\Desktop\广东\zhuhai.tif"    # 输出裁切后的 DEM

# ------------------------------
# 1. 读取 SHP
# ------------------------------
city = gpd.read_file(shp_path)

# 投影转换: shp CRS 若与 DEM 不一致，需要转换
with rasterio.open(dem_path) as src:
    dem_crs = src.crs
city = city.to_crs(dem_crs)    # 转成 DEM 的 CRS

# 获取 shapely geometry
geoms = [geom for geom in city.geometry]

# ------------------------------
# 2. 使用 rasterio.mask 裁切 DEM
# ------------------------------
with rasterio.open(dem_path) as src:
    out_image, out_transform = mask(src, geoms, crop=True)
    out_meta = src.meta.copy()

# 更新元数据
out_meta.update({
    "height": out_image.shape[1],
    "width":  out_image.shape[2],
    "transform": out_transform
})

# ------------------------------
# 3. 保存裁切后的 DEM
# ------------------------------
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_image)

print("裁切完成！输出文件：", out_tif)
