import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_id = "/data/jguo376/pretrained_models/Qwen-VL-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    bf16=True,
    ignore_mismatched_sizes=True
).eval()

input_json = "/data/jguo376/project/dataset/ChartX_dataset/chartx.json"
output_json = "/data/jguo376/project/dataset/test_dataset/train_test/dataset/qwen_caption_output.json"

results = []
max_items = 1  # 控制样本数量


with open(input_json, "r", encoding="utf-8") as fin:
    data_list = json.load(fin)

for idx, item in enumerate(tqdm(data_list, desc="Generating captions")):
    if max_items is not None and idx >= max_items:
        break
    try:
        base_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
        image_path = os.path.join(base_dir, item["img"].lstrip("./"))
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": "Provide a short analytical description of the chart, including specific values, comparisons, and trends."}
        ])

        response, _ = model.chat(tokenizer, query=query, history=None)

        item["model_name"] = model_id
        item["generated_caption"] = response
        results.append(item)

        print(f"{item.get('imgname', f'idx_{idx}')} => {response[:60]}...")

    except Exception as e:
        print(f"Failed to process line {idx}: {e}")

# 写出结果
with open(output_json, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

print(f"批处理完成，结果保存在：{output_json}")
