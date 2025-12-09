import geopandas as gpd
import os

def gdb_to_csv(gdb_path, layer_name, output_csv_path, geometry_to_wkt=True):
    """
    读取 GDB 文件中的一个图层，将其属性信息导出为 CSV 文件。
    
    参数:
    gdb_path (str): .gdb 文件的完整路径。
    layer_name (str): GDB 文件中包含 POI 数据的图层的名称（Feature Class Name）。
    output_csv_path (str): 导出的 CSV 文件的完整路径。
    geometry_to_wkt (bool): 是否将几何信息（点、线、面）转换为 WKT 格式的字符串并包含在 CSV 中。
                           如果只需要属性信息，可以设置为 False。
    """
    
    print(f"正在读取 GDB 文件: {gdb_path}")
    print(f"正在读取图层: {layer_name}")
    
    # 1. 使用 GeoPandas 读取 GDB 图层
    # GeoPandas 使用 fiona 驱动，可以读取 GDB 文件。
    try:
        # layer 参数指定要读取的 Feature Class 名称
        gdf = gpd.read_file(gdb_path, layer=layer_name)
    except Exception as e:
        print(f"读取 GDB 文件或图层失败: {e}")
        # 尝试列出图层名称，以便用户确认
        try:
            import fiona
            layers = fiona.listlayers(gdb_path)
            print(f"该 GDB 中可用的图层名称有: {layers}")
        except:
            pass
        return

    print(f"成功读取 {len(gdf)} 条记录。")

    # 2. 处理几何信息
    if geometry_to_wkt:
        # 将 GeoDataFrame 的 'geometry' 列转换为 WKT (Well-Known Text) 格式的字符串
        # 这样几何信息（如POI的坐标）可以保留在 CSV 中
        gdf['WKT_Geometry'] = gdf.geometry.apply(lambda geom: geom.wkt if geom else None)
        # 移除原有的 GeoPandas geometry 列（因为它包含复杂的几何对象，无法直接存入普通 CSV）
        df_export = gdf.drop(columns=['geometry'])
    else:
        # 如果不需要几何信息，只保留属性，则直接移除 geometry 列
        df_export = gdf.drop(columns=['geometry'], errors='ignore')

    # 3. 导出到 CSV 文件
    try:
        # index=False 表示不将 GeoDataFrame 的索引写入 CSV
        df_export.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"数据已成功导出到: {output_csv_path}")
        print(f"CSV 文件包含 {len(df_export.columns)} 个属性字段。")
    except Exception as e:
        print(f"导出 CSV 文件失败: {e}")

# --- 配置参数 ---

# 你的 GDB 文件路径 (替换成你的实际路径)
GDB_FILE = r"E:\University\dataset_guangzhou\广州百度poi\广州.gdb"

# GDB 中包含 POI 数据的图层名称 (Feature Class)
# 你需要知道这个名称。如果你不确定，可以尝试使用 fiona.listlayers(GDB_FILE) 查找。
LAYER_NAME = "广州"

# 导出的 CSV 文件路径
OUTPUT_CSV = r"E:\University\dataset_guangzhou\广州百度poi\guangzhou_poi.csv"

# --- 执行函数 ---

# 确保 GDB 文件存在
if os.path.isdir(GDB_FILE) and GDB_FILE.endswith(".gdb"):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    gdb_to_csv(GDB_FILE, LAYER_NAME, OUTPUT_CSV, geometry_to_wkt=True)
else:
    print(f"错误: GDB 文件路径不正确或不存在: {GDB_FILE}")