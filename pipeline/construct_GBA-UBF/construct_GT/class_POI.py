import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/百度POI_2/haizhu_2_POI.csv")  # 替换为你的实际路径

# 定义五个类别及其中类映射
category_dict = {

    "商业": ["美食", "酒店","购物", "生活服务", "丽人", "休闲娱乐", "运动健身","汽车服务","金融"],
    "公共服务": ["生活服务", "文体传媒", "政府机构", "交通设施", "医疗", "旅游景点"],
    "住宅": ["房地产"],
    "科技与工业区": ["公司企业"],
    "教育文化用地": ["教育培训"]
}

# 创建“功能区分类”列
def classify_category(mid_category):
    for cat_name, keywords in category_dict.items():
        if mid_category in keywords:
            return cat_name
    return "未分类"  # 如果不在列表中，标为未分类

df["功能区分类"] = df["main_tag"].apply(classify_category)

# 添加label列（1-5），按照指定顺序
label_mapping = {
    "商业": 1,
    "公共服务": 3,
    "住宅": 2,
    "科技与工业区": 4,
    "教育文化用地": 5
}

df["label"] = df["功能区分类"].map(label_mapping)

# 保存处理结果
df.to_csv("E:/University/dataset_guangzhou/广州研究区/研究区二/百度POI_2/haizhu_2_POI_label.csv", index=False, encoding='utf-8-sig')

print("处理完成，已保存至 '重新分类_添加label.csv'")
