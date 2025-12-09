import pandas as pd
from pyproj import Transformer
import re

# ========== 参数 ==========
input_csv = "E:/University/guangzhou_dataset_full/eva/graph_guanzhou1_field.csv"                # 输入文件（包含 geometry 列）
output_csv = "E:/University/guangzhou_dataset_full/eva/graph_guanzhou1_field_con.csv"  # 输出文件
geom_col = "geometry"                  # 原始列名
out_col = "geometry_xy"                # 要写入的新列名

# 如果你确认是 Web Mercator（看起来像你给的示例），使用 EPSG:3857
# 如果确认是 UTM zone 49N，请改为 "EPSG:32649"
src_crs = "EPSG:3857"
dst_crs = "EPSG:4326"

transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

# ========== 辅助函数（未改动） ==========
def parse_coords_from_wkt(wkt):
    wkt = wkt.strip()
    wkt_upper = wkt.upper()
    if wkt_upper.startswith("POINT"):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", wkt)
        if len(nums) >= 2:
            return "POINT", [(float(nums[0]), float(nums[1]))]
        return None, None

    if wkt_upper.startswith("POLYGON"):
        inner = re.search(r"POLYGON Z\s*\(\s*\((.*)\)\s*\)", wkt, flags=re.IGNORECASE)
        if not inner:
            return None, None
        ring_text = inner.group(1).strip()
        pts = []
        for part in ring_text.split(","):
            xy = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", part)
            if len(xy) >= 2:
                pts.append((float(xy[0]), float(xy[1])))
        return "POLYGON", [pts]

    if wkt_upper.startswith("MULTIPOLYGON"):
        polys = re.findall(r"\(\s*\(([^()]+)\)\s*\)", wkt)
        result = []
        for poly_text in polys:
            pts = []
            for part in poly_text.split(","):
                xy = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", part)
                if len(xy) >= 2:
                    pts.append((float(xy[0]), float(xy[1])))
            if pts:
                result.append([pts])
        return "MULTIPOLYGON", result

    return None, None

def transform_polygon_rings(rings):
    out_rings = []
    for ring in rings:
        xs, ys = zip(*ring)
        lons, lats = transformer.transform(xs, ys)
        out_ring = [(round(lon, 6), round(lat, 6)) for lon, lat in zip(lons, lats)]
        out_rings.append(out_ring)
    return out_rings

def rings_to_wkt_polygon(rings):
    ring_texts = []
    for ring in rings:
        pts_txt = ", ".join(f"{pt[0]} {pt[1]}" for pt in ring)
        ring_texts.append(f"({pts_txt})")
    return "POLYGON (" + ", ".join(ring_texts) + ")"

def multipolys_to_wkt(mpolys):
    poly_texts = []
    for rings in mpolys:
        ring_texts = []
        for ring in rings:
            pts_txt = ", ".join(f"{pt[0]} {pt[1]}" for pt in ring)
            ring_texts.append(f"({pts_txt})")
        poly_texts.append("(" + ", ".join(ring_texts) + ")")
    return "MULTIPOLYGON (" + ", ".join(poly_texts) + ")"

# ========== 主流程（此处是唯一修改！ iteritems -> items） ==========
df = pd.read_csv(input_csv, dtype=str)

out_list = []
for i, row in df[geom_col].fillna("").items():  # <-- 修改这里
    wkt = row.strip()
    if not wkt:
        out_list.append("")
        continue

    gtype, parsed = parse_coords_from_wkt(wkt)
    if gtype is None:
        out_list.append("")
        continue

    if gtype == "POINT":
        x, y = parsed[0]
        lon, lat = transformer.transform(x, y)
        out_list.append(f"POINT ({round(lon,6)} {round(lat,6)})")
        continue

    if gtype == "POLYGON":
        rings = parsed
        trans_rings = transform_polygon_rings(rings)
        out_list.append(rings_to_wkt_polygon(trans_rings))
        continue

    if gtype == "MULTIPOLYGON":
        trans_polys = []
        for rings in parsed:
            trans_polys.append(transform_polygon_rings(rings))
        out_list.append(multipolys_to_wkt(trans_polys))
        continue

df[out_col] = out_list
df.to_csv(output_csv, index=False)
print("转换完成，输出文件：", output_csv)