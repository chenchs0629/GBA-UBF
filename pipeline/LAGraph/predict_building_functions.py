#predict_building_functions
import os
import torch
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from torch_geometric.data import Data
from gnn_model import BuildingFunctionGNN  # 与train_tile_gnn中保持一致

# ----------------------------
# 参数设置
# ----------------------------
graph_dir = "./tile_graphs"         # 存放tile图的文件夹
model_path = "./gnn_model.pt"       # 已训练好的模型参数
building_shp = "./data/buildings.shp"  # 原始建筑物shp，用于结果写回
output_csv = "./results/building_predictions.csv"
output_shp = "./results/building_predictions.shp"

num_classes = 5  # 与训练时一致，如：居住/商业/教育/工业/公共服务

# ----------------------------
# 加载模型
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BuildingFunctionGNN(in_channels=32, hidden_channels=64, out_channels=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# 遍历tile图文件进行预测
# ----------------------------
results = []  # 存储所有预测结果 (building_id, class, probs)

for file in tqdm(os.listdir(graph_dir), desc="Predicting"):
    if not file.endswith(".pkl"):
        continue

    with open(os.path.join(graph_dir, file), "rb") as f:
        tile_data = pickle.load(f)

    # 假设 tile_data = {"building_ids": [...], "graph": Data对象}
    graph = tile_data["graph"].to(device)
    building_ids = tile_data["building_ids"]

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    for bid, pred, p in zip(building_ids, preds, probs):
        results.append({
            "building_id": bid,
            "pred_class": int(pred),
            **{f"prob_class{i}": p[i] for i in range(num_classes)}
        })

# ----------------------------
# 合并并去重（重叠区域平均）
# ----------------------------
df = pd.DataFrame(results)
df = df.groupby("building_id").agg({col: "mean" for col in df.columns if col != "building_id"}).reset_index()
df["final_class"] = df[[f"prob_class{i}" for i in range(num_classes)]].idxmax(axis=1).str.replace("prob_class", "").astype(int)

# ----------------------------
# 输出结果
# ----------------------------
df.to_csv(output_csv, index=False)
print(f"[✔] 已输出预测结果到: {output_csv}")

# 将预测结果写回建筑物shp属性表
if os.path.exists(building_shp):
    gdf = gpd.read_file(building_shp)
    gdf = gdf.merge(df, left_on="building_id", right_on="building_id", how="left")
    gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[✔] 已生成预测建筑功能矢量文件: {output_shp}")
