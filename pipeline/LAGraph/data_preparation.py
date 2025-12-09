##data_preparation
import geopandas as gpd
import pandas as pd
import os

# 尝试导入 fiona ，用于读取 FileGDB (.gdb)
try:
    import fiona
except Exception:
    fiona = None
from shapely.ops import polygonize
from shapely.geometry import LineString, Polygon, Point
from shapely.strtree import STRtree
import numpy as np
from tqdm import tqdm

tqdm.pandas()

# ========== Step 1: 根据路网生成街区 ==========
def generate_city_blocks(road_shp_path, output_block_path):
    roads = gpd.read_file(road_shp_path)
    roads = roads.to_crs(epsg=3857)  # 投影为米制坐标

    # 提取线要素并生成街区多边形
    merged = roads.unary_union
    polygons = list(polygonize(merged))
    
    blocks = gpd.GeoDataFrame(geometry=polygons, crs=roads.crs)
    blocks["block_id"] = np.arange(len(blocks))
    
    blocks.to_file(output_block_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"✅ Step 1 完成：共生成 {len(blocks)} 个街区。")
    return blocks

def read_poi_from_gdb(gdb_path, layer_name=None):
    """
    使用 fiona 读取 File Geodatabase (.gdb) 中的 POI 图层并返回 GeoDataFrame。

    - gdb_path: 指向 .gdb 文件夹（或 .gdb 文件）的路径
    - layer_name: 可选，指定要读取的图层名称；如果为 None，则尝试找到包含 'poi' 的图层名，否则取第一个图层

    如果本环境没有安装 fiona，会抛出 ImportError 并给出安装建议。
    """
    if fiona is None:
        raise ImportError(
            "fiona 未安装。请在终端运行：python -m pip install fiona 或使用 conda: conda install -c conda-forge fiona"
        )

    # 列出图层
    layers = fiona.listlayers(gdb_path)
    if not layers:
        raise ValueError(f"在 GDB 路径 {gdb_path} 中未找到任何图层")

    # 如果未指定图层名，尝试猜测包含 'poi' 的图层
    if layer_name is None:
        for l in layers:
            if 'poi' in l.lower():
                layer_name = l
                break
        if layer_name is None:
            layer_name = layers[0]

    # 使用 geopandas 读取指定图层（geopandas 底层使用 fiona）
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    return gdf

def compute_dynamic_poi_weights(poi_shp_path, block_gdf, building_shp_path, output_path, radius=500):

    # 导入必要的库
    import geopandas as gpd
    import numpy as np
    import os
    from shapely.strtree import STRtree
    from tqdm import tqdm
    try:
        import fiona
    except ImportError:
        fiona = None
    
    # 假设 read_poi_from_gdb 存在
    def read_poi_from_gdb(path):
        # 实际实现应在此处，为示例，仅用gpd.read_file
        return gpd.read_file(path)

    # 读取 POI（GDB 或 SHP）
    if fiona is not None and (str(poi_shp_path).lower().endswith('.gdb') or os.path.isdir(poi_shp_path)):
        pois = read_poi_from_gdb(poi_shp_path)
    else:
        pois = gpd.read_file(poi_shp_path)

    pois = pois.to_crs(block_gdf.crs)
    buildings = gpd.read_file(building_shp_path).to_crs(block_gdf.crs)

    # POI 类别 (假设是从 1.0 到 5.0 的字符串或浮点数)
    categories = sorted(list(pois["category"].unique())) # 确保顺序一致性

    # ----------- 建立 STRtree（正确映射） -----------
    poi_geoms = list(pois.geometry.values)
    geom_to_idx = {geom: i for i, geom in enumerate(poi_geoms)}
    poi_tree = STRtree(poi_geoms)

    # ----------- 全局类别数量（用于 IDF） -----------
    global_counts = pois["category"].value_counts().to_dict()
    total_pois = len(pois)

    # 平滑 IDF（不会压制大类，也不会放大稀有类）
    idf = {}
    for cat in categories:
        frac = global_counts.get(cat, 1) / total_pois
        idf[cat] = 1 / np.sqrt(frac)    # 平滑，范围稳定 ~1–3
    # 可选：尝试使用 scipy 的 cKDTree 以加速点要素的邻域查询
    try:
        from scipy.spatial import cKDTree as KDTree  # type: ignore
    except Exception:
        KDTree = None
    # -----------------------------------------------

    # 建筑物加入街区 id
    building_with_block = gpd.sjoin(buildings, block_gdf, how="left", predicate="within")
    # 添加一个新列用于存储所有类别的得分信息，便于后续修正
    building_with_block["dominant_class"] = None
    building_with_block["_scores"] = None 
    
    # ---------------- 主循环：每个街区处理 ----------------
    for _, block in tqdm(block_gdf.iterrows(), total=len(block_gdf), desc="Processing blocks"):

        block_id = block["block_id"]
        block_geom = block.geometry

        # 街区内 POI
        local_pois = pois[pois.intersects(block_geom)]
        if len(local_pois) == 0:
            continue

        # 街区内部的类别比例（弱偏好项，增加区分度）
        local_ratio = local_pois["category"].value_counts(normalize=True).to_dict()

        # 街区内所有建筑
        local_buildings = building_with_block[building_with_block["block_id"] == block_id]
        if len(local_buildings) == 0:
            continue

        # ========== 为街区内每栋建筑赋 POI 权重（向量化实现以提速） ==========
        poi_types = local_pois.geometry.geom_type.unique()
        pois_are_points = (len(poi_types) == 1 and poi_types[0] == 'Point')
        
        # 准备类别索引
        cat_to_idx = {c: i for i, c in enumerate(categories)}

        if pois_are_points:
            # ... (省略前半部分坐标准备和距离计算，与原代码一致)
            poi_x = np.array([g.x for g in local_pois.geometry])
            poi_y = np.array([g.y for g in local_pois.geometry])
            poi_cats = np.array(local_pois['category'].values)
            poi_cat_idx = np.array([cat_to_idx.get(c, -1) for c in poi_cats])

            b_idx_list = list(local_buildings.index)
            b_centroids = np.array([[g.x, g.y] for g in local_buildings.geometry.centroid])
            if b_centroids.size == 0 or poi_x.size == 0:
                continue

            bx = b_centroids[:, 0][:, None]
            by = b_centroids[:, 1][:, None]
            px = poi_x[None, :]
            py = poi_y[None, :]

            # 优先使用 KDTree（如果可用），避免构建全矩阵导致内存/时间开销
            if KDTree is not None:
                # KDTree 支持对一组点批量返回半径邻居
                tree = KDTree(np.column_stack((poi_x, poi_y)))
                # b_centroids 是 (nb,2)
                neighbors_list = tree.query_ball_point(b_centroids, r=radius)
                beta = 0.5
                idf_arr = np.array([idf[c] for c in categories])
                local_ratio_arr = np.array([local_ratio.get(c, 0.0) for c in categories])

                for i_local, idx in enumerate(b_idx_list):
                    nbrs = neighbors_list[i_local]
                    if len(nbrs) == 0:
                        continue
                    nbrs = np.asarray(nbrs, dtype=int)
                    # 计算距离与权重
                    dists = np.hypot(poi_x[nbrs] - b_centroids[i_local, 0], poi_y[nbrs] - b_centroids[i_local, 1])
                    weights = np.exp(-dists / radius)
                    cat_indices = poi_cat_idx[nbrs]
                    valid = cat_indices >= 0
                    if not np.any(valid):
                        continue
                    cat_indices = cat_indices[valid].astype(int)
                    w_vals = weights[valid]
                    tf_arr = np.bincount(cat_indices, weights=w_vals, minlength=len(categories))

                    total_tf = tf_arr.sum() + 1e-9
                    tf_norm = tf_arr / total_tf
                    score_arr = tf_norm * (idf_arr * (1 + beta * local_ratio_arr))
                    dom_idx = int(np.argmax(score_arr))
                    building_with_block.at[idx, '_scores'] = {categories[i]: float(score_arr[i]) for i in range(len(categories))}
                    building_with_block.at[idx, 'dominant_class'] = categories[dom_idx]
            else:
                # 回退到矩阵广播（原有实现）
                dists = np.hypot(bx - px, by - py)
                mask = dists <= radius
                if not np.any(mask):
                    continue

                weights = np.exp(-dists / radius) * mask
                nb, npoi = weights.shape
                MAX_CELLS = 50_000_000

                if nb * npoi > MAX_CELLS:
                    # 对每个建筑单独计算（但仍用 numpy 计算距离）
                    for i_local, idx in enumerate(b_idx_list):
                        w_row = weights[i_local]
                        valid = w_row > 0
                        if not np.any(valid):
                            continue
                        cat_indices = poi_cat_idx[valid]
                        valid2 = cat_indices >= 0
                        if not np.any(valid2):
                            continue
                        cat_indices = cat_indices[valid2].astype(int)
                        w_vals = w_row[valid][valid2]
                        tf_arr = np.bincount(cat_indices, weights=w_vals, minlength=len(categories))

                        total_tf = tf_arr.sum() + 1e-9
                        tf_norm = tf_arr / total_tf
                        beta = 0.5
                        score_arr = tf_norm * np.array([idf[c] * (1 + beta * local_ratio.get(c, 0.0)) for c in categories])
                        dom_idx = int(np.argmax(score_arr))

                        building_with_block.at[idx, 'dominant_class'] = categories[dom_idx]
                else:
                    # 批量按类别汇总权重
                    tf_matrix = np.zeros((nb, len(categories)), dtype=float)
                    for i_cat, cat in enumerate(categories):
                        poi_mask = poi_cat_idx == i_cat
                        if not np.any(poi_mask):
                            continue
                        tf_matrix[:, i_cat] = weights[:, poi_mask].sum(axis=1)

                    # 归一化并计算 score
                    tf_sums = tf_matrix.sum(axis=1) + 1e-9
                    tf_norm = tf_matrix / tf_sums[:, None]
                    beta = 0.5
                    idf_arr = np.array([idf[c] for c in categories])
                    local_ratio_arr = np.array([local_ratio.get(c, 0.0) for c in categories])
                    score_matrix = tf_norm * (idf_arr * (1 + beta * local_ratio_arr))[None, :]
                    dom_indices = np.argmax(score_matrix, axis=1)

                    for i_local, idx in enumerate(b_idx_list):
                        building_with_block.at[idx, 'dominant_class'] = categories[int(dom_indices[i_local])]
        else:
            # 回退：逐建筑的原始逻辑（处理非点 POI）
            for idx, b in local_buildings.iterrows():
                # ... (省略中间部分，与原代码一致)
                buffer_geom = b.geometry.buffer(radius)
                nearby = poi_tree.query(buffer_geom)
                if len(nearby) == 0:
                    continue

                nearby_idx = []
                for g in nearby:
                    try:
                        if isinstance(g, (int, np.integer)):
                            nearby_idx.append(int(g))
                            continue
                    except Exception:
                        pass
                    if g in geom_to_idx:
                        nearby_idx.append(geom_to_idx[g])

                if len(nearby_idx) == 0:
                    continue

                nearby_pois = pois.iloc[nearby_idx]
                tf_raw = {cat: 0.0 for cat in categories}
                for _, p in nearby_pois.iterrows():
                    cat = p["category"]
                    d = b.geometry.centroid.distance(p.geometry)
                    w = np.exp(-d / radius)
                    tf_raw[cat] += w

                total_tf = sum(tf_raw.values()) + 1e-9
                tf = {cat: tf_raw[cat] / total_tf for cat in categories}

                beta = 0.5
                score = {cat: tf[cat] * idf[cat] * (1 + beta * local_ratio.get(cat, 0.0)) for cat in categories}
                dominant = max(score, key=score.get)
                
                # --- 记录完整得分（修正点 3：逐建筑计算） ---
                building_with_block.at[idx, "_scores"] = score
                # ----------------------------------------
                
                building_with_block.at[idx, "dominant_class"] = dominant
                

    # Step X：基于可调类权重的弱监督全局校准（替换原有修正逻辑）
    # ========================
    print("🔄 开始进行基于可调全局权重的弱监督校准...")

    # 目标先验（以字符串键为准以防数据类型不一致）
    target_proportions = {
        '1.0': 0.3,  # 商业服务
        '2.0': 0.3,  # 住宅
        '3.0': 0.15,  # 公共服务
        '4.0': 0.20,  # 科技与工业区
        '5.0': 0.05   # 教育与文化
    }

    # 将 categories 统一为字符串顺序（保证 mapping 一致）
    categories_str = [str(c) for c in categories]
    n_cats = len(categories_str)

    # 构建原始得分矩阵 S_orig (n_buildings x n_cats)
    # 对于没有 _scores 的行，用 one-hot (dominant_class) 作为近似
    scored_idx = []
    S_list = []
    for idx, row in building_with_block.iterrows():
        s = row.get('_scores', None)
        if s is None or not s:
            # 用 dominant_class 近似（若 None 则全零）
            dom = row.get('dominant_class', None)
            vec = np.zeros(n_cats, dtype=float)
            if dom is not None:
                try:
                    pos = categories_str.index(str(dom))
                    vec[pos] = 1.0
                except ValueError:
                    pass
        else:
            # 从 dict (可能 key 类型混杂) 生成向量
            vec = np.zeros(n_cats, dtype=float)
            for k, v in s.items():
                try:
                    pos = categories_str.index(str(k))
                    vec[pos] = float(v)
                except ValueError:
                    # 忽略未知类别
                    pass
            # 若全为0（极少），则以 dominant_class one-hot 作为备份
            if vec.sum() == 0:
                dom = row.get('dominant_class', None)
                if dom is not None:
                    try:
                        pos = categories_str.index(str(dom))
                        vec[pos] = 1.0
                    except ValueError:
                        pass

        S_list.append(vec)
        scored_idx.append(idx)

    S_orig = np.vstack(S_list)  # shape (n_buildings, n_cats)
    n_buildings = S_orig.shape[0]

    # 规范化每行为概率分布（防止原始未归一）
    row_sums = S_orig.sum(axis=1, keepdims=True) + 1e-12
    S_orig = S_orig / row_sums

    # 计算原始 top / second 差异（用于锁定高置信样本）
    sorted_idx = np.argsort(-S_orig, axis=1)  # 按行降序排列索引
    top_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1] if n_cats > 1 else top_idx
    top_vals = S_orig[np.arange(n_buildings), top_idx]
    second_vals = S_orig[np.arange(n_buildings), second_idx]
    score_margin = top_vals - second_vals

    # 锁定阈值：置信差 > lock_threshold 的建筑将被锁定，不允许修改标签
    lock_threshold = 0.35 # 建议 0.2–0.4，可调；越大越保守
    locked_mask = score_margin > lock_threshold

    # 初始化类权重 w_c = 1.0
    weights = np.ones(n_cats, dtype=float)

    # 更新参数
    gamma = 0.7            # 更新步长 (0,1]，越小越平滑
    max_iter = 15
    tol = 1e-3             # 当最大绝对偏差 < tol 时停止
    min_w, max_w = 0.3, 3.0  # 权重上下界，防止单次翻盘

    # 将 target_proportions 按 categories_str 排序为向量（更健壮的匹配：支持字符串或数值键）
    def resolve_prop_for_cat(cat, prop_dict):
        # 直接按原始类型或字符串匹配
        if cat in prop_dict:
            return float(prop_dict[cat])
        s = str(cat)
        if s in prop_dict:
            return float(prop_dict[s])

        # 尝试把类别与 prop_dict 的 key 做数值比较
        try:
            catf = float(cat)
        except Exception:
            catf = None

        if catf is not None:
            for k, v in prop_dict.items():
                try:
                    if float(k) == catf:
                        return float(v)
                except Exception:
                    # 忽略无法转换为 float 的 key
                    continue

        # 兼容性：尝试部分匹配（如去掉小数点）
        for k, v in prop_dict.items():
            try:
                if str(k) == s:
                    return float(v)
            except Exception:
                continue

        return 0.0

    P_list = [resolve_prop_for_cat(c, target_proportions) for c in categories]
    P_target = np.array(P_list, dtype=float)
    total_P = P_target.sum()
    if total_P <= 0:
        # 不抛异常，改为退回到均匀分布，同时打印警告以便用户检查 target_proportions 键名是否匹配
        print("⚠️ 未能从 target_proportions 中匹配到任何类别比例；将退回到对所有类别均匀分配。请检查 target_proportions 的键是否与 POI 中的类别一致。")
        P_target = np.ones(len(categories), dtype=float) / float(len(categories))
    else:
        P_target = P_target / total_P

    # 迭代权重更新
    for it in range(max_iter):
        # 应用权重到原始分数
        S_weighted = S_orig * weights[None, :]  # 广播乘法
        # 对被锁定的样本强制保持原 top 类：
        if locked_mask.any():
            # 让锁定样本在其原 top 类上具有极大优势：将整行置小值，top 类置为极大
            # 但保留数值稳定性：不破坏其他未锁样本
            locked_rows = np.where(locked_mask)[0]
            # 先把所有锁定行置为极小
            S_weighted[locked_rows, :] = 1e-12
            # 再把它们的原 top 类设置为 1.0（或保留原 top_vals 的比例）
            S_weighted[locked_rows, top_idx[locked_rows]] = 1.0

        # 按行归一化并取 argmax 得到当前标签
        row_sums = S_weighted.sum(axis=1, keepdims=True) + 1e-12
        S_norm = S_weighted / row_sums
        labels_idx = np.argmax(S_norm, axis=1)

        # 计算当前比例（在所有样本上统计；若你希望只统计未锁定样本，可改）
        counts = np.bincount(labels_idx, minlength=n_cats).astype(float)
        current_props = counts / counts.sum()

        # 计算最大偏差
        max_dev = np.max(np.abs(current_props - P_target))
        print(f"  iter {it+1}: max_dev={max_dev:.4f}, props={dict(zip(categories_str, current_props.round(4)))}")

        if max_dev < tol:
            print("  收敛：已接近目标分布。")
            break

        # 计算每类的修正因子 r_c = target / current (若 current==0，用大值)
        # 使用平滑避免除零
        smoothing = 1e-6
        ratios = (P_target + smoothing) / (current_props + smoothing)

        # 更新 weights multiplicatively: w_c *= r_c^gamma
        weights = weights * (ratios ** gamma)
        # 裁剪
        weights = np.clip(weights, min_w, max_w)

    # 最终应用 weights 并生成最终标签（再次尊重锁定）
    S_final = S_orig * weights[None, :]
    if locked_mask.any():
        locked_rows = np.where(locked_mask)[0]
        S_final[locked_rows, :] = 1e-12
        S_final[locked_rows, top_idx[locked_rows]] = 1.0

    S_final = S_final / (S_final.sum(axis=1, keepdims=True) + 1e-12)
    final_labels_idx = np.argmax(S_final, axis=1)
    final_labels = [categories_str[i] for i in final_labels_idx]

    # 将结果写回 building_with_block 的 dominant_class（注意使用索引映射）
    for i, idx in enumerate(scored_idx):
        building_with_block.at[idx, 'dominant_class'] = final_labels[i]

    # 打印最终结果对比
    final_counts = pd.Series(final_labels).value_counts().to_dict()
    final_props = {cat: final_counts.get(cat, 0) / float(len(final_labels)) for cat in categories_str}
    print(f"✅ 全局校准完成，最终占比: {final_props}")

    # 清理临时列（如果你之前希望删除_scores）
    building_with_block = building_with_block.drop(columns=['_scores'], errors='ignore')

    # 输出
    building_with_block.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")

    print("✅ Step 2 完成（使用新版权重模型）。")
    return building_with_block

# ========== 示例运行 ==========
if __name__ == "__main__":
    #road_path = "E:/University/guangzhou_dataset_full/gz_roadnet/GZ_roadnet_clip.shp"
    #poi_path = "E:/University/guangzhou_dataset_full/广州百度poi/广州_label_clip.gdb"
    #building_path = "E:/University/guangzhou_dataset_full/gz_building_shp/guangzhou_shp_clip.shp"
    #block_output = "E:/University/guangzhou_dataset_full/output/blocks/blocks_downtown.shp"
    #building_output = "E:/University/guangzhou_dataset_full/output/building_class_ini2/building_with_class.shp"
    
    road_path = "E:/University/zhuhai_dataset_full/珠海市_路网/珠海市.shp"
    poi_path = "E:/University/zhuhai_dataset_full/珠海POI/zhuhai_POI.shp"
    building_path = "E:/University/zhuhai_dataset_full/zhuhai_building/zhuhai_building_shp.shp"
    block_output = "E:/University/zhuhai_dataset_full/output/blocks/blocks_city.shp"
    building_output = "E:/University/zhuhai_dataset_full/output/building_class_ini/building_with_class2.shp"
    
    # Step1: 路网生成街区
    blocks = generate_city_blocks(road_path, block_output)

    # Step2: 动态扫描并为建筑赋值
    # 如果是 .gdb，可以先查看图层并在需要时传入 layer 名称
    if fiona is not None and poi_path.lower().endswith('.gdb'):
        try:
            layers = fiona.listlayers(poi_path)
            print("Found layers in GDB:", layers)
        except Exception as e:
            print("无法列出 GDB 图层：", e)

    result = compute_dynamic_poi_weights(poi_path, blocks, building_path, building_output)

