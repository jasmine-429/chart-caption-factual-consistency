import os
import json
from tqdm import tqdm
from tools.ChartVLM import infer_ChartVLM

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_path = "/data/jguo376/pretrained_models/ChartVLM/base"
prompt = "Provide a short analytical description of the chart based on the data it shows."

# ===== 路径配置 =====
dataset_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
input_json = os.path.join(dataset_root, "chartve_missing_subset.json")
output_json = os.path.join(dataset_root, "chartve_missing_caption.json")
real_image_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

# ===== 加载输入 JSON =====
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ===== 加载已处理图像路径（标准化路径作为唯一标识）=====
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
            for entry in existing_results:
                processed_imgs.add(entry["img"])  # 使用绝对路径作为唯一标识
        except Exception:
            existing_results = []
else:
    existing_results = []

results = existing_results.copy()

# ===== 推理参数 =====
max_test = None
save_every = 20
count = 0

print("Start ChartVLM caption generation...")

for item in tqdm(data_list, desc="Generating captions"):
    raw_img_path = item.get("img", "")
    rel_path = raw_img_path.replace(real_image_root, "").lstrip("/")
    image_path = os.path.normpath(os.path.join(real_image_root, rel_path))

    # 统一为绝对路径格式
    item["img"] = image_path

    if image_path in processed_imgs:
        continue

    if max_test is not None and count >= max_test:
        break

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = infer_ChartVLM(image_path, prompt, model_path)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = "ChartVLM"
    item["generated_caption"] = caption
    results.append(item)
    processed_imgs.add(image_path)
    count += 1

    print(f"[✓] {item.get('imgname')}")
    print(f"    → {caption}\n")

    if count % save_every == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ===== 最终保存 =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nChartVLM 推理完成！共生成 {len(results)} 条图表描述，输出保存在: {output_json}")
