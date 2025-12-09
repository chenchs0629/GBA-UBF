# -*- coding: utf-8 -*-
"""
需要：
1. NVIDIA GPU
2. CUDA Toolkit  
3. pip install cupy-cuda12x（对于CUDA 12.x）
"""

import shp_utils as shp
import time

# ==================== 配置 ====================

SHP_PATH = r"J:\\POI大创数据集汇总\\深圳1\\final_1.shp"
OUTPUT_PATH = r"J:\\POI大创数据集汇总\\深圳1\\final_1_gpu.shp"

LABEL_NAMES = {
    1: "商业",
    2: "住宅", 
    3: "公共服务",
    4: "科技与工业区",
    5: "教育文化用地"
}

# GPU优化参数
MAX_FEATURES_GPU = 15000  # GPU处理的最大要素数
K_VALUE_GPU = 6  # GPU模式下使用较小的k值以节省显存

# ==================== 检查GPU ====================

gpu_available = False
gpu_memory_gb = 0

try:
    import cupy as cp
    
    # 检查GPU
    device = cp.cuda.Device()
    mem_info = device.mem_info
    total_mem_gb = mem_info[1] / (1024**3)
    free_mem_gb = mem_info[0] / (1024**3)
    gpu_memory_gb = free_mem_gb
    
    print("="*60)
    print("GPU信息")
    print("="*60)
    print(f"✓ GPU设备: {device.name}")
    print(f"✓ 可用显存: {free_mem_gb:.2f} GB / {total_mem_gb:.2f} GB")
    
    # 根据显存大小决定是否使用GPU
    if free_mem_gb < 1.0:
        print(f"\n⚠ 警告: 可用显存不足1GB")
        print("  将使用多线程版本代替")
        gpu_available = False
    else:
        gpu_available = True
        print("✓ 显存充足，可以使用GPU加速")
        
except ImportError:
    print("="*60)
    print("GPU检测")
    print("="*60)
    print("✗ CuPy未安装")
    print("\n安装方法（CUDA 12.x）:")
    print("  pip install cupy-cuda12x")
    print("\n将使用多线程版本代替...")
except Exception as e:
    print("="*60)
    print("GPU检测")
    print("="*60)
    print(f"✗ GPU检测失败: {e}")
    print("\n将使用多线程版本代替...")

# ==================== 分析 ====================

print("\n" + "="*60)
if gpu_available:
    print("功能性分析 - GPU加速版")
else:
    print("功能性分析 - 多线程版本（GPU不可用）")
print("="*60)

# 读取数据
print("\n正在读取数据...")
start_time = time.time()
data = shp.read_shapefile(SHP_PATH)
read_time = time.time() - start_time
n_features = len(data)
print(f"✓ 读取 {n_features} 个要素 (耗时: {read_time:.2f}秒)")

# 评估是否适合GPU处理
if gpu_available and n_features > MAX_FEATURES_GPU:
    print(f"\n⚠ 警告: 数据量({n_features})较大，可能导致显存不足")
    print(f"建议：")
    print(f"  1. 使用多线程版本")
    print(f"  2. 抽样分析（如50%）")
    print(f"  3. 减小k值（当前: {K_VALUE_GPU}）")
    
    response = input("\n是否继续使用GPU？(y/n): ")
    if response.lower() != 'y':
        gpu_available = False
        print("切换到多线程版本...")

# 显存需求估算
if gpu_available:
    estimated_mem = (n_features ** 2 * 4) / (1024**3)  # GB
    print(f"\n显存需求估算: {estimated_mem:.2f} GB")
    if estimated_mem > gpu_memory_gb * 0.8:
        print(f"⚠ 警告: 显存可能不足")
        print(f"  可用显存: {gpu_memory_gb:.2f} GB")
        print(f"  估算需求: {estimated_mem:.2f} GB")
        print("\n将使用多线程版本...")
        gpu_available = False

# 1. 功能多样性
print("\n【步骤1】功能多样性分析")
start_time = time.time()
diversity = shp.functional_diversity_analysis(
    data, 
    label_names=LABEL_NAMES
)
div_time = time.time() - start_time
print(f"耗时: {div_time:.2f}秒")

# 2. 莫兰指数
print("\n【步骤2】莫兰指数分析")

if gpu_available:
    print(f"使用GPU加速（k={K_VALUE_GPU}）...")
    print("="*60)
    
    unique_labels = sorted(data['label'].unique())
    morans_results = {}
    
    start_time = time.time()
    for label in unique_labels:
        label_name = LABEL_NAMES.get(label, str(label))
        binary_field = f'_binary_{label}'
        data[binary_field] = (data['label'] == label).astype(int)
        
        try:
            print(f"\n【{label_name}】")
            label_start = time.time()
            
            result = shp.compute_morans_i_gpu(
                data,
                field=binary_field,
                weights='knn',
                k=K_VALUE_GPU,  # 使用较小k值
                row_standardize=True
            )
            
            label_time = time.time() - label_start
            morans_results[label] = result
            
            print(f"  莫兰指数 I: {result['I']:.4f}")
            print(f"  期望值 E(I): {result['EI']:.4f}")
            print(f"  GPU耗时: {label_time:.2f}秒")
            
        except Exception as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in str(e):
                print(f"  ✗ 显存不足，切换到多线程...")
                gpu_available = False
                break
            else:
                print(f"  ✗ 计算失败: {e}")
                morans_results[label] = None
        
        # 清理临时字段
        data.drop(columns=[binary_field], inplace=True)
        
        # 清理GPU缓存
        if gpu_available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
    
    morans_time = time.time() - start_time
    print(f"\n莫兰指数总耗时: {morans_time:.2f}秒 (GPU)")

# 如果GPU失败或不可用，使用多线程
if not gpu_available:
    from multiprocessing import cpu_count
    n_jobs = max(1, cpu_count() // 4)  # 保守设置
    
    print(f"使用多线程加速（{n_jobs}个核心）...")
    start_time = time.time()
    morans_results = shp.compute_morans_i_by_label_parallel(
        data,
        label_names=LABEL_NAMES,
        weights='knn',
        k=8,  # 多线程可以用较大k值
        permutations=999,
        n_jobs=n_jobs,
        verbose=True
    )
    morans_time = time.time() - start_time
    print(f"\n莫兰指数总耗时: {morans_time:.2f}秒 (多线程)")

# 3. 热点分析（只用多线程）
print("\n【步骤3】热点分析")
print("热点分析使用多线程版本（GPU显存消耗太大）")

hotspots, data = shp.analyze_land_use_hotspots(
    data,
    label_field='label',
    target_labels=list(shp.LABEL_NAMES.keys()),
    generate_report=True,
    generate_plots=True,
    shp_output_path=OUTPUT_PATH,
    plot_output_dir=r"J:\\POI大创数据集汇总\\深圳1\\plots.png"
)
hotspot_time = time.time() - start_time
print(f"\n热点分析总耗时: {hotspot_time:.2f}秒")


# 总结
total_time = read_time + div_time + morans_time + hotspot_time
print("\n" + "="*60)
print("分析完成！")
print("="*60)
print(f"\n总耗时: {total_time:.2f}秒")
print(f"  - 数据读取: {read_time:.2f}秒")
print(f"  - 多样性分析: {div_time:.2f}秒")
if gpu_available:
    print(f"  - 莫兰指数: {morans_time:.2f}秒 (GPU)")
else:
    print(f"  - 莫兰指数: {morans_time:.2f}秒 (多线程)")
print(f"  - 热点分析: {hotspot_time:.2f}秒 (多线程)")

print("\n💡 性能提示:")
if gpu_available:
    print("  ✓ GPU加速已成功使用")
else:
    print("  ℹ GPU未使用，已使用多线程替代")
