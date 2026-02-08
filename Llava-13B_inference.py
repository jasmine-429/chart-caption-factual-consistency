import os
import sys
import io
import json
from tqdm import tqdm
from llava.eval.run_llava import eval_model
from collections import OrderedDict

# ===== 设置 GPU 和模型信息 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
LLAVA_MODEL = "liuhaotian/llava-v1.5-13b"
DEVICE = "cuda"

# ===== 构造参数对象 =====
def make_args(image_path):
    return type('Args', (), {
        "model_path": LLAVA_MODEL,
        "model_base": None,
        "model_name": "llava-v1.5-13b",
        "query": "Provide a short analytical description of the chart, including specific values, comparisons, and trends.",
        #"query": "Provide a short analytical description of the chart based on the data it shows.",
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 256
    })()

# ===== 路径配置 =====
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/chartx_selected_fields.json"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/llava_caption_output.json"
image_base = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

# ===== 加载数据（只取前3条）=====
with open(input_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)
#input_data = input_data[:400]

# ===== 推理并收集结果 =====
results = []
for item in tqdm(input_data, desc="📊 推理中"):
    image_path = os.path.join(image_base, item["img"].lstrip("./"))
    if not os.path.exists(image_path):
        print(f"[!] 图像不存在：{image_path}")
        continue

    try:
        args = make_args(image_path)

        # 重定向 stdout 以捕获模型生成的输出
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        eval_model(args)
        sys.stdout = old_stdout

        # 获取输出内容
        output = mystdout.getvalue()
        lines = output.strip().split('\n')
        caption = lines[-1] if lines else ""

        if caption.strip().lower() == "none" or not caption.strip():
            print(f"生成失败或为空：{item['imgname']}")
            caption = ""
        else:
            print(f"{item['imgname']} => {caption[:60]}...")

        # === 关键：构造有序字典，调整字段顺序 ===
        ordered_item = OrderedDict()
        for key in item:
            if key not in ["model_name", "generated_caption"]:
                ordered_item[key] = item[key]
        ordered_item["model_name"] = LLAVA_MODEL
        ordered_item["generated_caption"] = caption.strip()

        results.append(ordered_item)

    except Exception as e:
        print(f"推理失败：{item.get('imgname', 'unknown')} - {e}")
        item["model_name"] = LLAVA_MODEL
        item["generated_caption"] = ""
        results.append(item)


# ===== 合并写入 JSON 文件 =====
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
else:
    existing_data = []

existing_data.extend(results)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=2)

print(f"批处理完成，已将 {len(results)} 条样本写入 {output_path}")
