import numpy as np
import rarfile
import rasterio
from rasterio.plot import show
import cv2
import matplotlib.pyplot as plt

# 可选进度条：尝试导入 tqdm，若不可用则提供一个回退实现
try:
    from tqdm import tqdm
except Exception:
    class _DummyTqdm:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)
        def update(self, n=1):
            pass
        def close(self):
            pass
    def tqdm(iterable=None, **kwargs):
        # 支持两种用法：作为包装器(iterable)或作为进度条对象(total=...)
        if iterable is None:
            return _DummyTqdm()
        return _DummyTqdm(iterable)
# 路径
input_path = r"C:\Users\chenchs\Desktop\广东\zhuhai_DEM1.tif"

# 读取 DEM 数据
with rasterio.open(input_path) as src:
    dem = src.read(1).astype(np.float32)   # 读取第一波段
    profile = src.profile                  # 存储元信息
    nodata = src.nodata if src.nodata is not None else -9999  # 获取无效值

# 创建掩码
mask = (dem == nodata) | np.isnan(dem)

# 粗略填补（使用均值）
dem_temp = dem.copy()
dem_temp[mask] = np.nanmean(dem[~mask])

# 双三次插值
scale = 2
resized_up = cv2.resize(dem_temp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
resized_down = cv2.resize(resized_up, (dem.shape[1], dem.shape[0]), interpolation=cv2.INTER_CUBIC)

# 填补缺失值
dem_filled = dem.copy()
dem_filled[mask] = resized_down[mask]

# 可视化
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(dem, cmap='terrain')
axs[0].set_title("Original DEM (with missing)")
axs[1].imshow(dem_filled, cmap='terrain')
axs[1].set_title("DEM after Bicubic Interpolation")
plt.show()

# 保存为新 TIF（可选）
output_path = r"C:\Users\chenchs\Desktop\广东\zhuhai_filled.tif"
profile.update(dtype=rasterio.float32, nodata=None)

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(dem_filled, 1)

print("插值填补完成，已保存为：", output_path)
