import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# 创建图像存储目录
output_dir = "graph/comparison/"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
with open("model_for_other_experiments/features/aggregated_features_powder.json", "r") as f:
    powder_data = json.load(f)

with open("model_for_other_experiments/features/aggregated_features_water.json", "r") as f:
    water_data = json.load(f)

#-----------------------------------------------对比全局特征---------------------------------------
print("-----------------------------------------------对比全局特征---------------------------------------")
# 提取全局特征
powder_global = powder_data["aggregated_global_features"]
water_global = water_data["aggregated_global_features"]

# 提取样本特征
powder_samples = powder_data["samples"]
water_samples = water_data["samples"]
print("全局特征对比：")
print(f"粉末实验最大重量: {powder_global['overall_max_weight']}, 水实验最大重量: {water_global['overall_max_weight']}")
print(f"粉末实验最小重量: {powder_global['overall_min_weight']}, 水实验最小重量: {water_global['overall_min_weight']}")
print(f"粉末实验平均增长速率: {powder_global['overall_average_growth_rate']}, 水实验平均增长速率: {water_global['overall_average_growth_rate']}")
print(f"粉末实验状态切换次数: {powder_global['total_state_switches']}, 水实验状态切换次数: {water_global['total_state_switches']}")


#----------------------------------------------- 样本间差异分析---------------------------------------
print("----------------------------------------------- 样本间差异分析---------------------------------------")
# 将样本数据转为 DataFrame
powder_df = pd.DataFrame(powder_samples)
water_df = pd.DataFrame(water_samples)

# 只选择数值列进行计算
powder_mean = powder_df.select_dtypes(include=["float64"]).mean()
water_mean = water_df.select_dtypes(include=["float64"]).mean()

# 样本均值对比
comparison = pd.concat([powder_mean, water_mean], axis=1, keys=["Powder", "Water"])
print("样本均值对比：")
print(comparison)

#----------------------------------------------- 聚类与分类分析---------------------------------------
print("----------------------------------------------- 聚类与分类分析---------------------------------------")
# 合并数据
powder_df["label"] = 0  # 粉末实验标签为 0
water_df["label"] = 1  # 水实验标签为 1
data = pd.concat([powder_df, water_df], ignore_index=True)

# 特征选择
features = ["mean_weight", "std_weight", "average_growth_rate"]
X = data[features]

# 标准化样本数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 聚类分析
kmeans = KMeans(n_clusters=2, random_state=0)
data["cluster"] = kmeans.fit_predict(X_scaled)

# 计算轮廓系数
silhouette_avg = silhouette_score(X_scaled, data["cluster"])
print(f"Silhouette Score: {silhouette_avg}")

#----------------------------------------------- 可视化分析---------------------------------------
print("----------------------------------------------- 可视化分析---------------------------------------")
# 最大重量对比
plt.boxplot([powder_df["max_weight"], water_df["max_weight"]],
            labels=["Powder", "Water"])
plt.title("Max Weight Comparison")
plt.savefig(os.path.join(output_dir, "max_weight_comparison.png"))
plt.close()

# 平均增长速率对比
plt.boxplot([powder_df["average_growth_rate"], water_df["average_growth_rate"]],
            labels=["Powder", "Water"])
plt.title("Average Growth Rate Comparison")
plt.savefig(os.path.join(output_dir, "average_growth_rate_comparison.png"))
plt.close()

# 聚类可视化
plt.scatter(data["mean_weight"], data["std_weight"], c=data["cluster"], cmap="viridis", label="Clusters")
plt.title("Clustering of Experiments")
plt.xlabel("Mean Weight")
plt.ylabel("Std Weight")
plt.colorbar(label="Cluster")
plt.savefig(os.path.join(output_dir, "clustering_of_experiments.png"))
plt.close()
# 状态比例对比
print("----------------------------------------------- 状态比例对比---------------------------------------")
powder_state_ratios = [sample["state_ratios"] for sample in powder_samples]
water_state_ratios = [sample["state_ratios"] for sample in water_samples]

# 转为 DataFrame
powder_state_ratios_df = pd.DataFrame(powder_state_ratios)
water_state_ratios_df = pd.DataFrame(water_state_ratios)

# 可视化
plt.boxplot([powder_state_ratios_df["ST"], water_state_ratios_df["ST"]],
            labels=["Powder", "Water"])
plt.title("Stable State Ratio (ST) Comparison")
plt.savefig(os.path.join(output_dir, "stable_state_ratio_comparison.png"))
plt.close()

plt.boxplot([powder_state_ratios_df["US"], water_state_ratios_df["US"]],
            labels=["Powder", "Water"])
plt.title("Unstable State Ratio (US) Comparison")
plt.savefig(os.path.join(output_dir, "unstable_state_ratio_comparison.png"))
plt.close()
# 状态切换次数对比
print("----------------------------------------------- 状态切换次数对比---------------------------------------")
powder_switch_counts = [int(sample["state_switches"]) for sample in powder_samples]
water_switch_counts = [int(sample["state_switches"]) for sample in water_samples]

# Visualization
x_positions = range(len(powder_switch_counts))  # x positions for Powder
x_labels = [str(i + 1) for i in x_positions]  # Increment x-axis labels by 1

plt.figure(figsize=(12, 8))  # Increase figure size for better readability

plt.bar(x_positions, powder_switch_counts, alpha=0.8, color="white", edgecolor="black", label="Powder", hatch="/") 
plt.bar(x_positions, water_switch_counts, alpha=0.8, color="gray", edgecolor="black", label="Water", bottom=powder_switch_counts) 

# Enlarge text elements
#plt.title("State Switch Count Comparison", fontsize=16)
plt.xlabel("Sample Index", fontsize=25)
plt.ylabel("State Switch Count", fontsize=25)

# Set x-axis ticks to align with each bar and enlarge tick labels
plt.xticks(ticks=x_positions, labels=x_labels, fontsize=20)
plt.yticks(fontsize=20)  # Enlarge y-axis tick labels

plt.legend(fontsize=20)  # Enlarge legend text
plt.grid(axis='y', linestyle='--', alpha=0.6)  # Optional: Add grid lines for clarity

plt.savefig(os.path.join(output_dir, "state_switch_count_comparison.png"))
plt.close()

# 动态增长速率分析
print("----------------------------------------------- 动态增长速率分析---------------------------------------")
plt.figure(figsize=(12, 8))
for idx, sample in enumerate(powder_samples + water_samples, start=1):
    # 创建增长速率时间序列
    growth_rate_time_series = [sample["average_growth_rate"]] * int(sample["total_records"])
    # 绘图
    plt.plot(range(len(growth_rate_time_series)), growth_rate_time_series,
             label=f"{'Powder' if idx <= len(powder_samples) else 'Water'} {idx}",
             alpha=0.5)

plt.title("Growth Rate Time Series Analysis")
plt.xlabel("Time (Records)")
plt.ylabel("Growth Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "growth_rate_time_series_comparison.png"))
plt.close()

#----------------------------------------------- 实验指标随次数变化分析---------------------------------------
print("----------------------------------------------- 实验指标随次数变化分析---------------------------------------")

# 提取每次实验的指标
powder_metrics = pd.DataFrame({
    "max_weight": [sample["max_weight"] for sample in powder_samples],
    "mean_weight": [sample["mean_weight"] for sample in powder_samples],
    "average_growth_rate": [sample["average_growth_rate"] for sample in powder_samples],
    "state_switches": [sample["state_switches"] for sample in powder_samples]
})

water_metrics = pd.DataFrame({
    "max_weight": [sample["max_weight"] for sample in water_samples],
    "mean_weight": [sample["mean_weight"] for sample in water_samples],
    "average_growth_rate": [sample["average_growth_rate"] for sample in water_samples],
    "state_switches": [sample["state_switches"] for sample in water_samples]
})

# 绘制每次实验的指标变化
# metrics_to_plot = ["max_weight", "mean_weight", "average_growth_rate", "state_switches"]
metrics_to_plot = ["average_growth_rate"]
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), powder_metrics[metric], label="Powder", marker="o")
    plt.plot(range(1, 21), water_metrics[metric], label="Water", marker="x", linestyle="dashed")
    #中間発表用
    # plt.plot(range(1, 21), powder_metrics[metric], label="Powder", linestyle="solid", color = "black")
    # plt.plot(range(1, 21), water_metrics[metric], label="Water", linestyle="dashed", color = "black")
    #plt.title(f"{metric.capitalize()} Over Experiments")
    plt.xlabel("Experiment Index", fontsize = 25)
    plt.ylabel(metric.replace("_", " ").capitalize(), fontsize = 25)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{metric}_trend_comparison.png"))
    plt.close()

print(f"实验指标趋势图已保存至 {output_dir}")
