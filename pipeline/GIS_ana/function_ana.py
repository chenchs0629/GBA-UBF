#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能性分析 - 完整版
==================
包含功能多样性、莫兰指数和热点分析
"""

import shp_utils as shp

# ==================== 配置 ====================

SHP_PATH = r"J:\\POI大创数据集汇总\\深圳1\\final_1.shp"
OUTPUT_PATH = r"J:\\POI大创数据集汇总\\深圳1\\final_1_hotspots.shp"


# ==================== 分析 ====================

# 读取数据
print("\n正在读取数据...")
data = shp.read_shapefile(SHP_PATH)
print(f"✓ 读取 {len(data)} 个要素")

# 1. 功能多样性
diversity = shp.functional_diversity_analysis(
    data, 
    label_names=shp.LABEL_NAMES
)
'''
# 2. 莫兰指数
morans = shp.compute_morans_i_by_label(
    data,
    label_names=shp.LABEL_NAMES,
    weights='knn',
    k=8,
    permutations=999
)
'''
# 3. 热点分析
hotspots, data = shp.analyze_land_use_hotspots(
    data,
    label_field='label',
    target_labels=list(shp.LABEL_NAMES.keys()),
    generate_report=True,
    generate_plots=True,
    shp_output_path=OUTPUT_PATH,
    plot_output_dir=r"J:\\POI大创数据集汇总\\深圳1\\plots.png"
)


print("\n新增字段说明:")
print("  {label}_GiStar  - Gi*统计量")
print("  {label}_Zscore  - Z分数")
print("  {label}_Pvalue  - P值")
print("  {label}_Hotspot - 热点类型")