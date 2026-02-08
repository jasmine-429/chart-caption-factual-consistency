import os
import sys

# ===== 设定环境 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")

import json
from PIL import Image
from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

base_model_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
lora_path = "/data/jguo376/pretrained_models/MMCA/mmca_lora_weights.bin"
input_root = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data"
output_root = "/data/jguo376/project/model/MMCA"
input_json = os.path.join(input_root, "eva_test.json")
output_json = os.path.join(output_root, "batch_test.json")


chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

# ===== 加载模型与 LoRA 权重 =====
print("加载模型与 LoRA 权重...")
model = MplugOwlForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
peft_config = LoraConfig(
    target_modules=r'.*language_model.*\.(q_proj|v_proj)',
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
lora_weights = torch.load(lora_path, map_location="cpu")
model.load_state_dict(lora_weights, strict=False)

# ===== 加载处理器 =====
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
image_processor = MplugOwlImageProcessor.from_pretrained(base_model_path)
processor = MplugOwlProcessor(image_processor, tokenizer)

# ===== 推理参数与 prompt =====
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}
query_prompt = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Please describe the chart.
AI:"""

# ===== 加载输入数据 =====
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ===== 断点续跑支持（使用 image_path 唯一标识）=====
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
            for entry in existing_results:
                processed_imgs.add(entry["img"])  # 使用完整路径避免重复处理
        except Exception:
            existing_results = []
else:
    existing_results = []

results = existing_results.copy()

# ===== 推理主循环 =====
max_test = 5    # 调试时可设为整数
save_every = 1    # 每隔 N 条保存一次
count = 0

print("开始批量生成图表描述...")
for item in tqdm(data_list, desc="Generating captions"):
    rel_path = item.get("img")
    image_path = os.path.join(chart_root, rel_path.lstrip("./"))

    if image_path in processed_imgs:
        continue

    if max_test is not None and count >= max_test:
        break

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=[query_prompt], images=[image], return_tensors="pt")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, **generate_kwargs)
                caption = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = "mPLUG-Owl + MMCA"
    item["img"] = image_path  # 更新为绝对路径
    item["generated_caption"] = caption
    results.append(item)
    processed_imgs.add(image_path)
    count += 1

    print(f"[✓] {item.get('imgname')}")
    print(f"    → {caption}\n")

    # ===== 中间保存 =====
    if count % save_every == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ===== 最终保存 =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n推理完成！共生成 {len(results)} 条图表描述，结果保存至：{output_json}")
