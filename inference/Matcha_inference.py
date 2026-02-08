import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_id = "google/matcha-chart2text-statista"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to(device).eval()
query = "Please describe the chart."

# ===== 路径配置 =====
Img_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

dataset_root = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data"
output_root="/data/jguo376/project/model/matcha"
input_json = os.path.join(dataset_root, "eva_test.json")
output_json = os.path.join(output_root, "matcha_caption_chartx_eva.json")

# ===== 控制处理条数 =====
max_test = None  # 可设置为整数，如 100，仅处理前100条；None表示处理全部
save_every = 20  # 每处理N条保存一次

# ===== 读取输入数据 =====
with open(input_json, "r", encoding="utf-8") as f:
    all_data = json.load(f)

if max_test is not None:
    all_data = all_data[:max_test]

# ===== 加载已处理记录（断点保护）=====
results = []
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
        processed_imgs = {item["img"] for item in results}
    print(f"🔁 已加载 {len(results)} 条历史结果，将跳过已处理项")

# ===== 推理函数 =====
def generate_caption(image_path, query):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption.replace("\x0A", "").strip()

# ===== 批量处理 =====
new_results = []
for idx, item in enumerate(tqdm(all_data, desc="Generating captions")):
    img_key = item["img"]
    if img_key in processed_imgs:
        continue

    relative_path = item["img"]
    image_path = os.path.join(Img_root, relative_path.replace("./", "")) if not os.path.isabs(relative_path) else relative_path

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = generate_caption(image_path, query)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = model_id
    item["img"] = image_path  # 替换为绝对路径
    item["generated_caption"] = caption

    results.append(item)
    new_results.append(item)
    processed_imgs.add(img_key)

    # ===== 每20条保存一次 =====
    if len(new_results) >= save_every:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"已保存 {len(results)} 条结果）")
        new_results.clear()

# ===== 最终保存 =====
if new_results:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"最终保存 {len(results)} 条结果")

print(f"任务完成！总共生成 {len(results)} 条 caption，输出路径：{output_json}")
