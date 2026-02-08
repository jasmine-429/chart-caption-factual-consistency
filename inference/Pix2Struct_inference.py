import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_id = "oroikon/ft_pix2struct_chart_captioning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📦 Loading model and processor...")
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to(device).eval()

# ===== 推理函数 =====
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.replace("\x0A", "").strip()

# ===== 路径配置 =====
dataset_root = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data"
output_root="/data/jguo376/project/model/pix2struct"
input_json = os.path.join(dataset_root, "eva_test.json")
output_json = os.path.join(output_root, "pix2struct_caption_chartx_eva.json")
chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

# ===== 加载输入数据 =====
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ===== 加载已完成（断点续跑，按 img 唯一）=====
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
            for entry in existing_results:
                processed_imgs.add(entry["img"])
        except Exception:
            existing_results = []
else:
    existing_results = []

results = existing_results.copy()

# ===== 参数配置 =====
max_test = None    # 设置为整数进行测试，None 表示处理全部
save_every = 20    # 每处理多少条保存一次

print("Start caption generation...")
count = 0
for item in tqdm(data_list, desc="Generating captions"):
    rel_path = item["img"]
    image_path = os.path.join(chart_root, rel_path.lstrip("./"))

    if image_path in processed_imgs:
        continue
    if max_test is not None and count >= max_test:
        break

    # 推理处理
    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = generate_caption(image_path)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = model_id
    item["img"] = image_path  # 使用绝对路径覆盖
    item["generated_caption"] = caption
    results.append(item)
    processed_imgs.add(image_path)
    count += 1

    print(f"{item.get('imgname', os.path.basename(image_path))}")
    print(f"{caption}\n")

    # 中间保存
    if count % save_every == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ===== 最终保存 =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nDone! Total: {len(results)} captions saved to: {output_json}")
