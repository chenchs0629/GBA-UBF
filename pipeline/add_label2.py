import fiona
from fiona import Env
from fiona.errors import DriverError
import os
from shutil import rmtree
from typing import Tuple, Optional, Any

# -----------------------------
# 参数设置
# -----------------------------
# 注意：请修改 input_path 为您的实际路径。
# 如果是 GDB，路径应为 GDB 文件夹：r"E:\path\to\your\data.gdb"
# 如果是 SHP，路径应为 .shp 文件：r"E:\path\to\your\data.shp"
input_path = r"E:\University\数据集汇总\珠海\研究区一_new\POI\xiangzhou_1.shp"  # <-- 假设输入路径为 GDB
output_gdb = r"E:\University\zhuhai_dataset_full\uba-gbf-nozhuhai-full\zhuhai1_subset\zhuhai1_POI.shp"  # 输出 GDB 路径

# 分类规则定义
category_rules = {
    1: ["美食", "酒店", "购物", "生活服务", "丽人", "休闲娱乐", "运动健身", "汽车服务", "金融"],
    2: ["房地产"],
    3: ["生活服务", "文体传媒", "政府机构", "交通设施", "医疗", "旅游景点"],
    4: ["公司企业"],
    5: ["教育培训"]
}

# -----------------------------
# 辅助函数：确定驱动和图层名
# -----------------------------
def get_driver_and_layer_name(input_path: str) -> Tuple[str, str, Optional[str]]:
    """
    根据文件路径确定 Fiona 驱动、数据源路径和图层名。

    Args:
        input_path: 用户提供的原始路径。

    Returns:
        (driver, data_source, layer_name)
    """
    
    # 路径规范化 (重要：确保路径格式对 GDAL 友好)
    normalized_path = os.path.normpath(input_path)
    base_name, ext = os.path.splitext(normalized_path)

    # 1. OpenFileGDB 格式
    # GDB 是一个文件夹，我们通过检查它是否是目录且以 .gdb 结尾来判断
    if os.path.isdir(normalized_path) and normalized_path.lower().endswith(".gdb"):
        # 对于 GDB，driver 是 OpenFileGDB，需要指定 layer name
        # 尝试列出图层并自动选择第一个图层（更稳健）
        try:
            if fiona is not None:
                layers = fiona.listlayers(normalized_path)
                if layers:
                    gdb_layer_name = layers[0]
                else:
                    raise ValueError(f"GDB 路径 {normalized_path} 中未找到任何图层")
            else:
                # 如果没有 fiona，退回到一个默认图层名（可能需要用户手动指定）
                gdb_layer_name = None
        except Exception:
            gdb_layer_name = None
        return "OpenFileGDB", normalized_path, gdb_layer_name
    
    # 2. ESRI Shapefile 格式
    # Shapefile 的主文件扩展名是 .shp
    elif ext.lower() == ".shp":
        # 对于 SHP，driver 是 ESRI Shapefile。
        # OGR 数据源 (DS) 是 .shp 文件路径本身。
        # 图层名 (layer) 传入 None 即可，fiona 会自动解析。
        return "ESRI Shapefile", normalized_path, None

    else:
        # 其他未支持的格式
        raise ValueError(
            f"❌ 不支持的输入格式或路径：{input_path}。请使用 .gdb 文件夹或 .shp 文件。"
        )

# -----------------------------
# 主执行逻辑
# -----------------------------
def process_data(input_path: str, output_gdb: str, category_rules: dict) -> None:
    """主函数：读取数据，分类并写入新的 GDB。"""

    # 1. 若输出路径已存在则删除（兼容 .gdb 目录 或 .shp 文件）
    out_ext = os.path.splitext(output_gdb)[1].lower()
    try:
        if out_ext == ".gdb":
            if os.path.exists(output_gdb):
                rmtree(output_gdb)
                print(f"♻️ 已删除旧的输出 GDB：{output_gdb}")
        elif out_ext == ".shp":
            # 删除 shapefile 的组件文件（.shp .shx .dbf .prj .cpg）
            base = os.path.splitext(output_gdb)[0]
            for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix'):
                f = base + ext
                if os.path.exists(f):
                    os.remove(f)
            print(f"♻️ 已删除旧的输出 Shapefile：{output_gdb}")
        else:
            # 如果未指定扩展，则尝试以目录方式处理（当作 gdb）
            if os.path.isdir(output_gdb):
                rmtree(output_gdb)
                print(f"♻️ 已删除旧的输出目录：{output_gdb}")
    except OSError as e:
        print(f"❌ 无法删除旧的输出文件/目录：{e}")
        return

    try:
        # 2. 确定输入驱动和图层
        input_driver, input_source, layer_name = get_driver_and_layer_name(input_path)
        
        # 打印信息用于调试
        print(f"ℹ️ 检测到输入格式：{input_driver}")
        if layer_name:
             print(f"ℹ️ 图层名：{layer_name}")
        
        # 3. 读取输入数据（使用 fiona.open 以便兼容 GDB 图层）
        with Env():
            # 动态设置 fiona.open() 参数，只在 layer_name 存在时传入 layer
            open_args: dict[str, Any] = {}
            if input_driver:
                open_args['driver'] = input_driver
            if layer_name is not None:
                open_args['layer'] = layer_name

            # 使用解包操作符 (**) 传入参数
            with fiona.open(input_source, **open_args) as src:
                meta = src.meta

                # 4. 添加新字段 'category' 到元数据
                # 如果字段已存在，fiona 会自动处理，但为了安全和类型统一，我们覆盖它
                meta["schema"]["properties"]["category"] = "int"

                # 5. 根据输出路径类型选择写入方式（支持 .gdb 或 .shp）
                output_layer_name = layer_name if layer_name else os.path.splitext(os.path.basename(input_source))[0]
                print(f"ℹ️ 输出图层名：{output_layer_name}")

                if out_ext == ".gdb":
                    # 写入 FileGDB（输出为目录）
                    with fiona.open(output_gdb, "w",
                                    driver="OpenFileGDB",
                                    layer=output_layer_name,
                                    schema=meta["schema"],
                                    crs=meta["crs"]) as dst:
                        for feat in src:
                            main_tag = feat["properties"].get("main_tag")
                            category_value = None
                            for cat, tags in category_rules.items():
                                if main_tag in tags:
                                    category_value = cat
                                    break
                            if category_value is None:
                                continue
                            feat["properties"]["category"] = category_value
                            dst.write(feat)
                    print(f"✅ 已保存结果到新 GDB：{output_gdb}")
                else:
                    # 写入 Shapefile（确保指定编码以防中文属性乱码）
                    shp_path = output_gdb if out_ext == ".shp" else (os.path.splitext(output_gdb)[0] + ".shp")
                    # 使用 UTF-8 编码写入，并在写入后生成 .cpg 文件以指示编码
                    with fiona.open(shp_path, "w",
                                    driver="ESRI Shapefile",
                                    schema=meta["schema"],
                                    crs=meta["crs"],
                                    encoding="utf-8") as dst:
                        for feat in src:
                            main_tag = feat["properties"].get("main_tag")
                            category_value = None
                            for cat, tags in category_rules.items():
                                if main_tag in tags:
                                    category_value = cat
                                    break
                            if category_value is None:
                                continue
                            feat["properties"]["category"] = category_value
                            dst.write(feat)
                    # 生成 .cpg 文件声明编码，许多 GIS 软件会读取该文件来确定 DBF 编码
                    try:
                        cpg_path = os.path.splitext(shp_path)[0] + ".cpg"
                        with open(cpg_path, "w", encoding="utf-8") as f:
                            f.write("UTF-8")
                    except Exception:
                        pass
                    print(f"✅ 已保存结果到 Shapefile：{shp_path} (encoding=UTF-8)")

    except DriverError as e:
        print(f"❌ 打开或创建数据出错：{e}")
    except ValueError as e:
        print(f"❌ 参数设置出错：{e}")
    except Exception as e:
        print(f"❌ 发生了意外错误：{e}")

# -----------------------------
# 运行主程序
# -----------------------------
if __name__ == "__main__":
    process_data(input_path, output_gdb, category_rules)