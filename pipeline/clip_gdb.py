import fiona
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import os
from shutil import rmtree

# -----------------------------
# 参数设置
# -----------------------------
input_gdb = r"E:\University\guangzhou_dataset_full\广州百度poi\广州_label.gdb"       # 输入GDB路径
layer_name = "广州"              # 需要裁剪的图层名
clip_shp = r"E:\University\guangzhou_dataset_full\GuangZhou_map\GZ_downtown.shp"         # 裁剪范围shp路径
output_gdb = r"E:\University\guangzhou_dataset_full\广州百度poi\广州_label_clip.gdb"     # 输出GDB路径

# 若输出GDB已存在则删除
if os.path.exists(output_gdb):
    rmtree(output_gdb)

# -----------------------------
# 读取裁剪范围 shapefile
# -----------------------------
with fiona.open(clip_shp, "r") as clip_src:
    # 合并所有几何为一个范围
    clip_geoms = [shape(feat["geometry"]) for feat in clip_src]
    clip_union = unary_union(clip_geoms)
    clip_crs = clip_src.crs

# -----------------------------
# 打开GDB文件并执行裁剪
# -----------------------------
with fiona.open(input_gdb, layer=layer_name, driver="OpenFileGDB") as src:
    meta = src.meta

    # 输出图层
    with fiona.open(output_gdb, "w",
                    driver="OpenFileGDB",
                    layer=layer_name,
                    schema=meta["schema"],
                    crs=meta["crs"]) as dst:

        count_in = 0
        count_out = 0
        for feat in src:
            geom = shape(feat["geometry"])

            # 判断相交
            if geom.intersects(clip_union):
                dst.write({
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": feat["properties"]
                })
                count_in += 1
            else:
                count_out += 1

        print(f"✅ 裁剪完成！保留 {count_in} 个要素，剔除 {count_out} 个要素。")
        print(f"📁 输出文件：{output_gdb}")
