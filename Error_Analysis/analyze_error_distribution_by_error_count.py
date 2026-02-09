import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ======= 路径配置 =======
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_typr.jsonl"
output_csv = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/model_error_distribution.csv"
output_png = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/error_distribution_gapped.png"

# ======= 1. 加载数据并解析错误类型 =======
records = []
with open(input_path, 'r') as f:
    for line in tqdm(f, desc="Parsing"):
        entry = json.loads(line)
        model = entry.get("model_name", "unknown_model")
        sentence_labels = entry.get("labels", [])
        for label_list in sentence_labels:
            for error_type in label_list:
                records.append({
                    "Model": model,
                    "Error Type": error_type
                })

df = pd.DataFrame(records)

# ======= 2. 统计各类错误比例 =======
error_counts = df.groupby(["Model", "Error Type"]).size().reset_index(name="Count")
model_totals = df.groupby("Model").size().reset_index(name="Total")
result_df = pd.merge(error_counts, model_totals, on="Model")
result_df["Proportion"] = result_df["Count"] / result_df["Total"]
result_df.to_csv(output_csv, index=False)
print("Error distribution saved to:", output_csv)

# ======= 3. 标签映射与颜色 =======
label_mapping = {
    "value_error": "Value Error",
    "label_error": "Label Error",
    "trend_error": "Trend Error",
    "ooc_error": "Out Of Context Error",
    "magnitude_error": "Magnitude Error",
    "nonsense_error": "Nonsense Error"
}
error_order = list(label_mapping.keys())
color_map = {
    "value_error": "#F58787",
    "label_error": "#FEAD76",
    "trend_error": "#FFD869",
    "ooc_error": "#7CAEF0",
    "magnitude_error": "#D1AEEC",
    "nonsense_error": "#b3e19b"
}

# ======= 4. 模型顺序 + 分组配置 =======
model_order = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B", "LLaVA-v1.5-13B",          # General
    "UniChart-201M", "Matcha-282M", "Pix2Struct-282M",         # Chart MLLMs
    "ChartInstruct-T5-3B", "MMCA-7B", "ChartVLM-13B"                  # Specialist
]
group_labels = ["General\nMLLMs", "Specialist\nChart Models", "Chart\nMLLMs"]
group_starts = [0, 3, 6]

# ======= 5. 创建透视表并排序 =======
pivot = result_df.pivot(index="Model", columns="Error Type", values="Proportion").fillna(0)
pivot = pivot.reindex(model_order).fillna(0)

models = model_order
y_pos = np.arange(len(models))

# ======= 6. 绘图（带白色分隔线 + 分组标签）=======
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.7
left = np.zeros(len(models))

for error in error_order:
    if error in pivot.columns:
        values = pivot[error].values
        ax.barh(
            y_pos,
            values,
            height=bar_height,
            left=left,
            color=color_map[error],
            label=label_mapping[error]
        )
        for i, v in enumerate(values):
            if v > 0:
                ax.plot([left[i], left[i]], [y_pos[i] - bar_height / 2, y_pos[i] + bar_height / 2],
                        color='white', linewidth=1)
        left += values

# ======= 7. 添加分组虚线和左侧标签 =======
for y in [2.5, 5.5]:
    ax.axhline(y=y, color="gray", linestyle="--", linewidth=1)

for label, start in zip(group_labels, group_starts):
    y_center = start + 1
    ax.text(-0.30, y_center, label, fontsize=12, fontweight='bold',
            ha='left', va='center', transform=ax.transData)

# ======= 8. 图标设置 =======
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=12)
ax.set_xlabel("Proportion of Errors", fontsize=12)
ax.invert_yaxis()
ax.xaxis.grid(True, linestyle='--', alpha=0.6)

# ======= 9. 图例和布局调整 =======
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper left", bbox_to_anchor=(1.02, 1),
    borderaxespad=0.,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.subplots_adjust(left=0.35, bottom=0.10)  # 给左边和底部更多空间
plt.savefig(output_png, bbox_inches="tight", dpi=300)
print("Plot saved to:", output_png)
plt.show()
