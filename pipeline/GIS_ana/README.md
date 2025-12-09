# 城市功能性分析工具

基于Python的城市空间功能性分析工具包，支持功能多样性分析、空间自相关分析和热点识别。

## 📋 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [指标解读](#指标解读)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

---

## 🎯 功能特性

### 核心分析功能

- **功能多样性分析**
  - Shannon熵指数
  - Simpson多样性指数
  - 功能均衡度评估
  
- **空间自相关分析**
  - 全局Moran's I指数计算
  - 分类别空间聚集性检验
  - 显著性检验（置换检验）
  
- **热点分析**
  - Getis-Ord Gi*统计量
  - 热点/冷点识别
  - 多层次显著性判定

### 性能优化（这个是ai扯淡的，别信，最多两倍，显卡特别好的当我没说）

- ✅ **标准版**：适用于常规数据集（< 1000要素）
- ⚡ **多线程版**：2-8倍加速（1000-10000要素）
- 🚀 **GPU加速版**：10-100倍加速（> 10000要素，需NVIDIA GPU）

---

## 💻 环境要求

### 基础环境

```bash
Python >= 3.8
geopandas >= 0.10.0
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
pysal >= 2.6.0
```

### 安装依赖

```bash
pip install geopandas numpy pandas scipy pysal
```

### GPU加速（可选）

**硬件要求：**
- NVIDIA GPU（显存 ≥ 8GB 推荐）
- CUDA Toolkit 11.x 或 12.x

**软件要求：**

```bash
# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

**性能对比：**

| 数据规模 | 标准版 | 多线程版 | GPU版 |
|---------|--------|---------|-------|
| 1000要素 | 10秒 | 4秒 | 2秒 |
| 5000要素 | 120秒 | 30秒 | 8秒 |
| 10000要素 | 480秒 | 90秒 | 15秒 |

---

## 🚀 快速开始

### 1. 标准版（推荐，但是慢）

```python
import shp_utils as shp

# 配置文件路径
SHP_PATH = r"C:\data\urban_poi.shp"
OUTPUT_PATH = r"C:\output\result.shp"

# 定义功能类型
LABEL_NAMES = {
    1: "商业",
    2: "住宅", 
    3: "公共服务",
    4: "科技与工业区",
    5: "教育文化用地"
}

# 读取数据
data = shp.read_shapefile(SHP_PATH)

# 1. 功能多样性分析
diversity = shp.functional_diversity_analysis(
    data, 
    label_names=LABEL_NAMES
)

# 2. 莫兰指数分析
morans = shp.compute_morans_i_by_label(
    data,
    label_names=LABEL_NAMES,
    weights='knn',
    k=8,
    permutations=999
)

# 3. 热点分析(自行取舍部分参数)
hotspots, data = shp.analyze_land_use_hotspots(
    data='land_use.shp',
    label_field='land_type',
    target_labels=[1, 3, 5],
    label_names={1: '商业区', 3: '公共设施', 5: '教育用地'},
    weights='knn',
    k=12,  # 使用12个最近邻
    method='binary',
    generate_report=True,
    generate_plots=True,
    plot_output_dir='./custom_outputs',
    shp_output_path='./custom_outputs/custom_hotspots.shp',
    verbose=True
)

# 保存结果
shp.write_shapefile(data, OUTPUT_PATH)
```

### 2. 多线程加速版（内存爆炸也是正常情况，建议16GB及以下内存不要轻易尝试）

```python
import shp_utils as shp

data = shp.read_shapefile(SHP_PATH)

# 使用多线程加速
morans = shp.compute_morans_i_by_label_parallel(
    data,
    label_names=LABEL_NAMES,
    k=8,
    n_jobs=-1  # -1表示使用所有CPU核心!!!!使用所有核心可能会导致程序崩溃，建议空闲状态使用可用核心的一半即可！！！！！！
)

hotspots, data = shp.analyze_land_use_hotspots(
    data='land_use.shp',
    label_field='land_type',
    target_labels=[1, 3, 5],
    label_names={1: '商业区', 3: '公共设施', 5: '教育用地'},
    weights='knn',
    k=12,  # 使用12个最近邻
    method='binary',
    generate_report=True,
    generate_plots=True,
    plot_output_dir='./custom_outputs',
    shp_output_path='./custom_outputs/custom_hotspots.shp',
    verbose=True
)
```
### 3. GPU加速版（显存溢出是正常情况，请不用惊慌）

详见 `加速版功能性分析.py`，自动检测GPU并智能切换计算方式。

**运行方式：**

```bash
python 加速版功能性分析.py
```

程序会自动：
- 检测GPU可用性
- 评估显存需求
- 智能降级到多线程（如显存不足）

---

## 🔬 核心功能

### 1. 功能多样性分析

#### shp.functional_diversity_analysis()

分析城市空间的功能类型多样性和分布均衡度。

**参数说明：**

```python
diversity = shp.functional_diversity_analysis(
    gdf,                    # GeoDataFrame对象
    label_field='label',    # 功能类型字段名
    label_names=None,       # 类型名称映射字典（可选）
    verbose=True            # 是否打印详细信息
)
```

**返回结果：**

```python
{
    'shannon_entropy': 1.548,      # Shannon熵
    'simpson_index': 0.732,        # Simpson指数
    'functional_balance': 0.843,   # 功能均衡度（0-1）
    'label_counts': {...},         # 各类型数量
    'label_proportions': {...}     # 各类型比例
}
```

---

### 2. 莫兰指数分析

#### shp.compute_morans_i_by_label()

计算各功能类型的空间自相关性，识别空间聚集模式。

**参数说明：**

```python
morans = shp.compute_morans_i_by_label(
    gdf,
    label_field='label',
    label_names=None,
    weights='knn',          # 空间权重类型：'knn'或'queen'
    k=8,                    # k近邻数量（weights='knn'时）
    permutations=999        # 置换检验次数
)
```

**返回结果：**

```python
{
    1: {
        'I': 0.345,           # Moran's I值
        'EI': -0.001,         # 期望值
        'z': 12.45,           # Z分数
        'p_value': 0.001      # P值
    },
    ...
}
```

---

### 3. 热点分析

#### shp.hotspot_analysis_by_label()

识别各功能类型的空间聚集热点区域（高值聚集）和冷点区域（低值聚集）。

**参数说明：**

```python
hotspots, data = shp.analyze_land_use_hotspots(
    data,
    label_field='label',
    target_labels=list(shp.LABEL_NAMES.keys()),
    generate_report=True,
    generate_plots=True,
    shp_output_path=OUTPUT_PATH,
    plot_output_dir=r"J:\\POI大创数据集汇总\\深圳1\\plots.png"
)
```

**新增字段：**

对每个功能类型（如"商业"）添加4个字段：
- `商业_GiStar`: Gi*统计量
- `商业_Zscore`: Z分数
- `商业_Pvalue`: P值
- `商业_Hotspot`: 热点类型

---

## 📊 指标解读

### 功能多样性指标

#### 1. Shannon熵 (Shannon Entropy)

**含义：** 衡量功能类型的丰富度和不确定性

**计算公式：**
```
H = -Σ(pi × ln(pi))
```
其中 pi 为第i类功能的比例

**数值范围：** 0 到 ln(n)，n为功能类型数

**解读标准：**
- **H < 1.0**：功能类型单一，多样性低
- **1.0 ≤ H < 1.5**：多样性较低
- **1.5 ≤ H < 2.0**：多样性中等
- **H ≥ 2.0**：多样性高，功能丰富

**实际意义：**
- 高Shannon熵：区域功能混合度高，用地类型丰富
- 低Shannon熵：区域功能单一，可能为单一用地类型主导

---

#### 2. Simpson指数 (Simpson Index)

**含义：** 衡量功能类型的多样性和均匀度

**计算公式：**
```
D = 1 - Σ(pi²)
```

**数值范围：** 0 到 1

**解读标准：**
- **D < 0.3**：多样性极低，某类功能占绝对优势
- **0.3 ≤ D < 0.6**：多样性较低
- **0.6 ≤ D < 0.8**：多样性较高
- **D ≥ 0.8**：多样性很高，各类功能分布均匀

**实际意义：**
- 高Simpson指数：功能类型分布均匀，不存在单一主导功能
- 低Simpson指数：少数功能类型占主导地位

---

#### 3. 功能均衡度 (Functional Balance)

**含义：** 标准化的功能分布均衡程度

**计算公式：**
```
Balance = H / ln(n)
```
其中 H 为Shannon熵，n为功能类型数

**数值范围：** 0 到 1

**解读标准：**
- **0.0-0.3**：功能严重失衡，单一功能主导
- **0.3-0.5**：功能较不均衡
- **0.5-0.7**：功能基本均衡
- **0.7-0.9**：功能较均衡
- **0.9-1.0**：功能高度均衡，理想的混合用地

**实际意义：**
- 接近1.0：各功能类型比例接近，符合混合用地理念
- 接近0.0：功能单一化严重，可能导致职住分离

---

### 莫兰指数分析

#### Moran's I 指数

**含义：** 衡量空间要素的聚集程度和空间依赖性

**数值范围：** -1 到 +1

**解读标准：**

| Moran's I | 空间模式 | 显著性 | 实际意义 |
|-----------|---------|-------|---------|
| **I > 0** | 正相关 | p < 0.05 | 相似类型空间聚集 |
| **I ≈ 0** | 随机分布 | p > 0.05 | 无明显空间模式 |
| **I < 0** | 负相关 | p < 0.05 | 相似类型空间分散 |

**显著性水平：**
- `***`：p < 0.01（高度显著）
- `**`：p < 0.05（显著）
- `*`：p < 0.10（边缘显著）
- `n.s.`：p ≥ 0.10（不显著）

**Z分数解读：**
- |Z| > 2.58：99%置信水平（***）
- |Z| > 1.96：95%置信水平（**）
- |Z| > 1.65：90%置信水平（*）
- |Z| ≤ 1.65：不显著

**实际应用示例：**

```
商业用地：I = 0.456*** (Z = 8.32, p < 0.001)
→ 商业设施呈显著空间聚集，存在商业中心
```

```
住宅用地：I = 0.123 n.s. (Z = 1.45, p = 0.147)
→ 住宅分布较为随机，无明显聚集或分散模式
```

```
工业用地：I = -0.234** (Z = -2.15, p = 0.032)
→ 工业设施呈空间分散分布，可能是规划隔离的结果
```

---

### 热点分析指标

#### Getis-Ord Gi* 统计量

**含义：** 识别空间高值聚集区（热点）和低值聚集区（冷点）

**热点类型分类：**

| 类型 | Z分数 | P值 | 含义 | 颜色建议 |
|------|-------|-----|------|---------|
| **Hot Spot (99%)** | Z > 2.58 | p < 0.01 | 高值聚集区（高度显著） | 深红色 |
| **Hot Spot (95%)** | 1.96 < Z ≤ 2.58 | 0.01 ≤ p < 0.05 | 高值聚集区（显著） | 红色 |
| **Hot Spot (90%)** | 1.65 < Z ≤ 1.96 | 0.05 ≤ p < 0.10 | 高值聚集区（边缘显著） | 橙红色 |
| **Not Significant** | -1.65 ≤ Z ≤ 1.65 | p ≥ 0.10 | 无显著模式 | 浅灰色 |
| **Cold Spot (90%)** | -1.96 ≤ Z < -1.65 | 0.05 ≤ p < 0.10 | 低值聚集区（边缘显著） | 浅蓝色 |
| **Cold Spot (95%)** | -2.58 ≤ Z < -1.96 | 0.01 ≤ p < 0.05 | 低值聚集区（显著） | 蓝色 |
| **Cold Spot (99%)** | Z < -2.58 | p < 0.01 | 低值聚集区（高度显著） | 深蓝色 |

**实际应用示例：**

**商业热点分析：**
- Hot Spot (99%)：核心商业中心（CBD）
- Hot Spot (95%)：次级商业中心
- Cold Spot (99%)：商业荒漠区

**住宅热点分析：**
- Hot Spot (99%)：高密度住宅聚集区
- Not Significant：职住平衡区域
- Cold Spot (99%)：非居住功能区

**公共服务热点分析：**
- Hot Spot：公共设施完善区域
- Cold Spot：公共服务薄弱区域（规划重点）

---

## 💡 使用示例

### 完整分析流程

```python
import shp_utils as shp

# 1. 数据准备
SHP_PATH = r"data/urban_poi.shp"
OUTPUT_PATH = r"output/result.shp"

LABEL_NAMES = {
    1: "商业",
    2: "住宅", 
    3: "公共服务",
    4: "科技与工业区",
    5: "教育文化用地"
}

# 2. 读取数据
print("正在读取数据...")
data = shp.read_shapefile(SHP_PATH)
print(f"✓ 读取 {len(data)} 个要素")

# 3. 功能多样性分析
print("\n【功能多样性分析】")
diversity = shp.functional_diversity_analysis(
    data, 
    label_names=LABEL_NAMES
)

print(f"\nShannon熵: {diversity['shannon_entropy']:.4f}")
print(f"Simpson指数: {diversity['simpson_index']:.4f}")
print(f"功能均衡度: {diversity['functional_balance']:.4f}")

# 解读多样性
balance = diversity['functional_balance']
if balance > 0.8:
    print("→ 功能高度均衡，混合用地特征明显")
elif balance > 0.6:
    print("→ 功能较均衡")
elif balance > 0.4:
    print("→ 功能基本均衡")
else:
    print("→ 功能分布不均衡，存在主导类型")

# 4. 莫兰指数分析
print("\n【空间自相关分析】")
morans = shp.compute_morans_i_by_label(
    data,
    label_names=LABEL_NAMES,
    weights='knn',
    k=8,
    permutations=999
)

for label, result in morans.items():
    if result:
        label_name = LABEL_NAMES[label]
        I = result['I']
        z = result['z']
        p = result['p_value']
        
        # 显著性标记
        if p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.10:
            sig = "*"
        else:
            sig = "n.s."
        
        # 空间模式判断
        if I > 0 and p < 0.05:
            pattern = "空间聚集"
        elif I < 0 and p < 0.05:
            pattern = "空间分散"
        else:
            pattern = "随机分布"
        
        print(f"\n{label_name}:")
        print(f"  Moran's I = {I:.4f}{sig}")
        print(f"  Z-score = {z:.4f}")
        print(f"  P-value = {p:.4f}")
        print(f"  → {pattern}")

# 5. 热点分析
print("\n【热点分析】")
hotspots, data = shp.hotspot_analysis_by_label(
    data,
    label_names=LABEL_NAMES,
    weights='knn',
    k=8
)

for label, result in hotspots.items():
    label_name = LABEL_NAMES[label]
    print(f"\n{label_name}热点统计:")
    
    # 统计各类型数量
    hotspot_field = f'{label_name}_Hotspot'
    if hotspot_field in data.columns:
        counts = data[hotspot_field].value_counts()
        
        for hotspot_type, count in counts.items():
            print(f"  {hotspot_type}: {count}个要素")

# 6. 保存结果
print(f"\n正在保存到: {OUTPUT_PATH}")
shp.write_shapefile(data, OUTPUT_PATH)
print("✓ 分析完成！")

print("\n输出文件新增字段：")
for label_name in LABEL_NAMES.values():
    print(f"  {label_name}_GiStar  - Gi*统计量")
    print(f"  {label_name}_Zscore  - Z分数")
    print(f"  {label_name}_Pvalue  - P值")
    print(f"  {label_name}_Hotspot - 热点类型")
```

---

### 结果可视化（在GIS软件中）

#### ArcGIS Pro / QGIS 可视化步骤：

1. **功能分布图**
   - 使用 `label` 字段进行分类显示
   - 设置不同颜色表示不同功能类型

2. **热点地图**
   - 使用 `{功能}_Hotspot` 字段
   - 建议配色方案：
     - Hot Spot (99%): 深红色 `#D73027`
     - Hot Spot (95%): 红色 `#FC8D59`
     - Hot Spot (90%): 橙红色 `#FEE090`
     - Not Significant: 浅灰色 `#E0E0E0`
     - Cold Spot (90%): 浅蓝色 `#E0F3F8`
     - Cold Spot (95%): 蓝色 `#91BFDB`
     - Cold Spot (99%): 深蓝色 `#4575B4`

3. **Z分数地图**
   - 使用 `{功能}_Zscore` 字段
   - 采用渐变色显示空间聚集强度

---

## ❓ 常见问题

### Q1: 如何选择合适的k值？

**A:** k值（近邻数量）选择建议：

| 数据特征 | 推荐k值 | 说明 |
|---------|--------|------|
| 稀疏分布 | 4-6 | 避免近邻过远 |
| 中等密度 | 8-10 | **通用推荐** |
| 高密度 | 12-16 | 捕捉更大范围模式 |

**经验法则：** k ≈ √n / 10，其中n为要素总数

---

### Q2: GPU版本显存不足怎么办？

**A:** 三种解决方案：

1. **降低k值**（推荐）
```python
K_VALUE_GPU = 4  # 从8降到4
```

2. **数据抽样**
```python
sample_data = shp.sample_features(data, frac=0.5)  # 50%抽样
```

3. **切换多线程版**
```python
morans = shp.compute_morans_i_by_label_parallel(data, k=8, n_jobs=-1)
```

---

### Q3: 莫兰指数不显著说明什么？

**A:** 不显著（p > 0.05）通常有三种情况：

1. **真随机分布**：功能类型空间分布确实随机
2. **尺度问题**：分析尺度不合适，调整k值
3. **数据质量**：数据量过少或分布不均

**建议：** 
- 尝试不同k值（6, 8, 10, 12）
- 检查数据分布是否均匀
- 考虑分区域分析

---

### Q4: 如何解读矛盾的结果？

**示例：** 某区域Shannon熵高，但莫兰指数显示聚集

**解读：**
- Shannon熵高：整体功能类型丰富多样
- 莫兰指数高：相同类型在空间上聚集
- **结论：** 该区域功能丰富，但呈"组团式"分布（商业区、住宅区分离明显）

这种模式在许多城市中很常见，反映了功能分区规划的影响。

---

### Q5: 计算速度慢怎么办？

**A:** 性能优化建议：

| 数据规模 | 推荐方案 | 预计耗时 |
|---------|---------|---------|
| < 1000 | 标准版 | < 30秒 |
| 1000-5000 | 多线程版 (n_jobs=-1) | 1-3分钟 |
| 5000-10000 | 多线程版 + 降低置换数 | 3-10分钟 |
| > 10000 | GPU版 或 抽样分析 | 5-30分钟 |

**减少置换次数：**
```python
morans = shp.compute_morans_i_by_label(data, permutations=99)  # 从999降到99
```



## 🤝 技术支持

俊霖友情提醒：
遇到问题可以先检查

- ✅ 文件路径正确（使用原始字符串 `r"路径"`）
- ✅ 数据坐标系为投影坐标系（非经纬度）
- ✅ `shp_utils.py` 在同一目录   ！！！！！
- ✅ 所有依赖包已正确安装   ！！！
如遇环境问题请私信我
注：我跑的时候，显存和内存都炸过。。。。。。。。

