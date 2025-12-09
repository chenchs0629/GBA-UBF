import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# ---------- 配置：确保 matplotlib 能正确显示中文和负号 ----------
# 尝试常见的 Windows 中文字体，并在找到第一个可用字体后设置为默认
preferred_fonts = ["Microsoft YaHei", "SimHei", "STHeiti", "PingFang SC", "Arial Unicode MS"]
available = {f.name for f in font_manager.fontManager.ttflist}
for fname in preferred_fonts:
    if fname in available:
        matplotlib.rcParams['font.family'] = fname
        break
# 如果上面没有找到合适字体，仍然设置常见的中文字体名作为后备
matplotlib.rcParams['font.sans-serif'] = [f for f in preferred_fonts if f in available] + ["SimHei", "Microsoft YaHei"]
# 避免负号显示为方块
matplotlib.rcParams['axes.unicode_minus'] = False

# =============================
# 1. 输入你的两个混淆矩阵
# =============================
cm1 = np.array([
    [2953, 90, 140, 295, 98],
    [268, 2807, 43, 134, 54],
    [57,  36, 261, 7, 23],
    [55,  53, 1, 275, 15],
    [17,  17, 6, 20, 130]
])

cm2 = np.array([
    [5410, 467, 127, 401, 123],
    [751, 4443, 166, 176, 112],
    [203, 272, 1641, 168, 83],
    [152, 383, 154, 3582, 28],
    [167, 98, 45, 9, 623]
])

class_names = ["商业", "住宅", "公共服务", "科技与文化", "教育文化"]

# =============================
# 2. 定义一个绘图函数（论文风格）
# =============================
def plot_confusion_matrix(cm, title, save_name):
    plt.figure(figsize=(6, 5), dpi=300)

    # 使用 Seaborn 热力图
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, fontsize=14)
    plt.xlabel("预测标签", fontsize=12)
    plt.ylabel("真值标签", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图已保存到：{save_name}")

# =============================
# 3. 绘制两个混淆矩阵
# =============================
plot_confusion_matrix(cm1, "混淆矩阵 - 研究区 1", "E:/University/zhuhai_dataset_full/eva/confusion_matrix_1.png")
plot_confusion_matrix(cm2, "混淆矩阵 - 研究区 2", "E:/University/zhuhai_dataset_full/eva/confusion_matrix_2.png")
