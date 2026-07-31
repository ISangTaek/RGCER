import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dict = np.load('./stage_rep/final_result.npy', allow_pickle=True).item()
a = dict['women_oral_TDLo']
def kde_visual(true_data, pred_data):
    # 整合为长格式DataFrame
    df = pd.DataFrame({
        "True Values": true_data,
        "Predicted Values": pred_data
    })
    df_long = df.melt(var_name="Type", value_name="Value")

    # 设置高对比度颜色（从viridis调色板取首尾颜色）
    viridis_palette = sns.color_palette("YlGnBu", n_colors=10)
    high_contrast_colors = [viridis_palette[0], viridis_palette[-1]]

    # 仅绘制KDE曲线
    plt.figure(figsize=(5, 5),dpi = 300)
    ax = sns.kdeplot(
        data=df_long,
        x="Value",
        hue="Type",
        palette=high_contrast_colors,
        lw=1,
        alpha=0.7,
        fill=True,  # 填充曲线下方区域
        common_norm=False  # 独立归一化
    )

    # 优化图例和图表
    plt.title("True vs Predicted Values", fontsize=12)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(
        title="Data Type",
        title_fontsize=12,
        fontsize=10,
        labels=["GroundTruth", "Predicted"]  # 确保图例标签正确
    )

    plt.show()
true_data, pred_data = a['gts'], a['pred']
true_data_cpu = np.array([i.cpu().numpy() for i in true_data]).reshape(-1)
pred_data_cpu = np.array([i.cpu().numpy() for i in pred_data]).reshape(-1)
kde_visual(true_data_cpu, pred_data_cpu)
