import fiona
from fiona import Env
from fiona.errors import DriverError
import os
from shutil import rmtree

# -----------------------------
# 参数设置
# -----------------------------
input_gdb = r"E:\University\guangzhou_dataset_full\广州百度poi\广州.gdb"       # 原始gdb路径
output_gdb = r"E:\University\guangzhou_dataset_full\广州百度poi\广州_label.gdb"     # 输出gdb路径
layer_name = "广州"              # 图层名（可通过fiona.listlayers查看）

# 分类规则定义
category_rules = {
    1: ["美食", "酒店", "购物", "生活服务", "丽人", "休闲娱乐", "运动健身", "汽车服务", "金融"],
    2: ["房地产"],
    3: ["生活服务", "文体传媒", "政府机构", "交通设施", "医疗", "旅游景点"],
    4: ["公司企业"],
    5: ["教育培训"]
    
}

# -----------------------------
# 若输出gdb存在则删除
# -----------------------------
if os.path.exists(output_gdb):
    rmtree(output_gdb)

# -----------------------------
# 读取输入数据
# -----------------------------
with Env():
    try:
        with fiona.open(input_gdb, layer=layer_name, driver="OpenFileGDB") as src:
            meta = src.meta

            # 添加新字段 category
            meta["schema"]["properties"]["category"] = "int"

            # 创建新的gdb
            with fiona.open(output_gdb, "w",
                            driver="OpenFileGDB",
                            layer=layer_name,
                            schema=meta["schema"],
                            crs=meta["crs"]) as dst:
                for feat in src:
                    main_tag = feat["properties"].get("main_tag")

                    # 确定category值
                    category_value = None
                    for cat, tags in category_rules.items():
                        if main_tag in tags:
                            category_value = cat
                            break

                    # 不符合则跳过
                    if category_value is None:
                        continue

                    # 写入属性
                    feat["properties"]["category"] = category_value
                    dst.write(feat)

            print(f"✅ 已保存结果到新GDB：{output_gdb}")

    except DriverError as e:
        print(f"❌ 打开GDB出错：{e}")
