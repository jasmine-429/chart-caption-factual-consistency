import os
import json
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from tqdm import tqdm

# ===== 模型与环境配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_name = "ahmed-masry/unichart-chart2text-statista-960"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

input_prompt = "<summarize_chart> <s_answer>"

# ===== 路径配置 =====
dataset_root = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data"
output_root="/data/jguo376/project/model/Unichart"
image_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
input_json = os.path.join(dataset_root, "eva_test.json")
output_json = os.path.join(output_root, "unichart_caption_chartx_eva.json")

# ===== 推理函数 =====
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.config.decoder.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    return sequence.split("<s_answer>")[-1].strip()

# ===== 加载输入数据 =====
with open(input_json, "r", encoding="utf-8") as infile:
    data_list = json.load(infile)

# ===== 加载已完成结果（使用 img 路径判断）=====
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
            for entry in existing_results:
                processed_imgs.add(entry["img"])
        except json.JSONDecodeError:
            existing_results = []
else:
    existing_results = []

results = existing_results.copy()

# ===== 推理主循环 =====
max_test = None    # 设为整数进行测试
save_every = 20     # 每 N 条保存一次结果
count = 0

print("Start caption generation...")

for data in tqdm(data_list, desc="Generating captions"):
    rel_path = data["img"]
    image_path = os.path.join(image_root, rel_path.lstrip("./"))

    if image_path in processed_imgs:
        continue
    if max_test is not None and count >= max_test:
        break

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = generate_caption(image_path)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    output = {
        "chart_type": data.get("chart_type"),
        "imgname": data.get("imgname"),
        "img": image_path,
        "topic": data.get("topic"),
        "title": data.get("title"),
        "csv": data.get("csv"),
        "model_name": model_name,
        "generated_caption": caption
    }
    results.append(output)
    processed_imgs.add(image_path)
    count += 1

    print(f"{data.get('imgname')}")
    print(f" {caption}\n")

    if count % save_every == 0:
        with open(output_json, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()

# ===== 最终写入完整结果 =====
with open(output_json, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, ensure_ascii=False, indent=2)

print(f"\nDone! Total: {len(results)} captions saved to: {output_json}")

