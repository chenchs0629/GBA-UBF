import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from scipy.interpolate import griddata
from shapely.geometry import Point
from tqdm import tqdm

# ----------------------------
# 输入路径（请自行修改）
# ----------------------------
dem_path = r"C:\Users\chenchs\Desktop\广东\zhuhai.tif"
shp_path = r"E:\University\zhuhai_dataset_full\zhihai_distract\zhuhai_dis.shp"
out_path = r"C:\Users\chenchs\Desktop\广东\zhuhai_fill.tif"

# ----------------------------
# 1. 读取城市范围（作为插值 mask）
# ----------------------------
city = gpd.read_file(shp_path)

with rasterio.open(dem_path) as src:
    dem_crs = src.crs

city = city.to_crs(dem_crs)

# ----------------------------
# 2. 读取 DEM
# ----------------------------
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(float)
    nodata = src.nodata
    transform = src.transform
    meta = src.meta.copy()

# 将 nodata 替换为 np.nan
dem[dem == nodata] = np.nan

# ----------------------------
# 3. 判断每个像元是否在城市范围内部
# ----------------------------
rows, cols = dem.shape
xs = np.arange(cols)
ys = np.arange(rows)

xx, yy = np.meshgrid(xs, ys)

# 像元坐标 -> 地理坐标
xs_geo, ys_geo = rasterio.transform.xy(transform, yy, xx)

# 构建点 GeoSeries（带进度条）
# 将地理坐标数组展平以便显示进度
xs_flat = np.array(xs_geo).ravel()
ys_flat = np.array(ys_geo).ravel()

print("构建像元点并检查是否在城市范围内...")
pts = gpd.GeoSeries(
    [Point(x, y) for x, y in tqdm(zip(xs_flat, ys_flat), total=xs_flat.size, desc="构建像元点")],
    crs=dem_crs,
)

inside_mask = pts.within(city.unary_union).to_numpy().reshape(rows, cols)

# ----------------------------
# 4. 仅在城市区域内部进行插值补洞
# ----------------------------
# 原始有效点（用于插值）
valid_mask = (~np.isnan(dem)) & inside_mask

coords_valid = np.column_stack(np.where(valid_mask))
values_valid = dem[valid_mask]

# 待插值点（城市内且 nan）
interp_mask = np.isnan(dem) & inside_mask
coords_interp = np.column_stack(np.where(interp_mask))

if len(coords_interp) > 0:
    print("正在对城市内部缺失值进行插值...")
    # 如果待插值点很多，分块调用 griddata 以提供进度反馈
    chunk_size = 50000
    interpolated_vals = np.empty(len(coords_interp), dtype=float)
    for i in tqdm(range(0, len(coords_interp), chunk_size), desc="插值进度"):
        i2 = min(i + chunk_size, len(coords_interp))
        chunk = coords_interp[i:i2]
        vals = griddata(
            coords_valid,
            values_valid,
            chunk,
            method='cubic',
            fill_value=np.nan,
        )
        interpolated_vals[i:i2] = vals

    # 将插值结果写回 DEM
    dem[interp_mask] = interpolated_vals

# ----------------------------
# 5. 保留城市范围外的原始 NaN，不进行填补
# ----------------------------

# 将 NaN 重新设置为 nodata
dem[np.isnan(dem)] = nodata

# ----------------------------
# 6. 保存结果
# ----------------------------
meta.update(dtype=rasterio.float32)

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(dem.astype(np.float32), 1)

print("插值完成！输出 DEM:", out_path)
