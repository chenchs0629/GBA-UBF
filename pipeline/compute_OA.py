import pandas as pd
import numpy as np
import shapely.wkt
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix

# =============== 参数 ===============
gt_csv = "E:/University/guangzhou_dataset_full/eva/guangzhou1_field.csv"              # 真值文件
pred_csv = "E:/University/guangzhou_dataset_full/eva/graph_guanzhou1_field_con.csv"   # 预测文件

gt_geom_col = "geometry"
pred_geom_col = "geometry_xy"

gt_label_col = "label"
pred_label_col = "dominant_c"

# =============== Step 1: 读取 CSV ===============
gt = pd.read_csv(gt_csv)
pred = pd.read_csv(pred_csv)

# 解析 WKT -> geometry 对象
gt["geom"] = gt[gt_geom_col].apply(shapely.wkt.loads)
pred["geom"] = pred[pred_geom_col].apply(shapely.wkt.loads)

# 计算质心
gt["centroid"] = gt["geom"].apply(lambda g: (g.centroid.x, g.centroid.y))
pred["centroid"] = pred["geom"].apply(lambda g: (g.centroid.x, g.centroid.y))

# 转为 numpy array
gt_xy = np.array(gt["centroid"].tolist())
pred_xy = np.array(pred["centroid"].tolist())

# =============== Step 2: 使用 KDTree 做匹配 ===============
tree = KDTree(gt_xy)
dist, idx = tree.query(pred_xy, k=1)     # 每个预测找到最近的真值

# 预测 → 真值 索引对应
pred["gt_index"] = idx[:, 0]

# =============== Step 3: 组装真值与预测 ===============
matched_gt_labels = gt.loc[pred["gt_index"], gt_label_col].values
matched_pred_labels = pred[pred_label_col].values

# =============== Step 4: 计算总体 OA ===============
oa = (matched_gt_labels == matched_pred_labels).sum() / len(matched_gt_labels)
print("Overall Accuracy (OA):", oa)

# =============== Step 5: 计算各类别精度 ===============
labels = [1, 2, 3, 4, 5]
cm = confusion_matrix(matched_gt_labels, matched_pred_labels, labels=labels)

class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\nClass Accuracy:")
for i, acc in enumerate(class_accuracy, start=1):
    print(f"  Class {i}: {acc}")

# 若需要输出混淆矩阵也可打印
print("\nConfusion Matrix:\n", cm)
