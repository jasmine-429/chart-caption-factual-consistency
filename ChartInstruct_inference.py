import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForSeq2SeqLM

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_id = "ahmed-masry/ChartInstruct-FlanT5-XL"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===== 路径配置 =====
input_json = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
output_path = "/data/jguo376/project/model/chartInstruction"
output_json = os.path.join(output_path, "chartx_caption.json")
real_image_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

# ===== 加载模型 =====
print("加载模型中...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
image_processor = AutoImageProcessor.from_pretrained(model_id)

# ===== 加载输入 JSON =====
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ===== 断点续跑：加载已处理 =====
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

# ===== 推理参数 =====
max_test = None  # 设置为整数以限制测试数量
save_every = 20
count = 0

# ===== 推理函数 =====
def infer_chartinstruct(image_path, input_question="Please describe the chart."):
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>\n Question: {input_question} Answer: "

    text_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )

    inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": pixel_values
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=512,
            early_stopping=True
        )

    output_text = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    return output_text

# ===== 批量推理 =====
print("开始批量生成 ChartInstruct-FlanT5-XL 图表描述...")
for item in tqdm(data_list, desc="Generating captions"):
    raw_img_path = item.get("img", "")
    rel_path = raw_img_path.replace(real_image_root, "").lstrip("/")
    image_path = os.path.normpath(os.path.join(real_image_root, rel_path))

    if image_path in processed_imgs:
        continue
    if max_test is not None and count >= max_test:
        break
    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = infer_chartinstruct(image_path)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    # 保留精简字段
    new_item = {
        "chart_type": item.get("chart_type", ""),
        "imgname": item.get("imgname", os.path.basename(image_path).split(".")[0]),
        "img": image_path,
        "topic": item.get("topic", ""),
        "title": item.get("title", ""),
        "csv": item.get("csv", ""),
        "generated_caption": caption
    }

    results.append(new_item)
    processed_imgs.add(image_path)
    count += 1

    print(f"{new_item['imgname']}")
    print(f"{caption}\n")

    if count % save_every == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ===== 最终保存 =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n推理完成！共生成 {len(results)} 条图表描述，输出保存在: {output_json}")
