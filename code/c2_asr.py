import json
import torch
import jiwer
import re
from transformers import pipeline
from tqdm import tqdm

# 1. 读取生成好的 dataset.json
dataset_path = "../common_data/dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# --- 缩小数据量：建议先取 100 条进行测试 ---
test_dataset = dataset[:100] 
audio_paths = [item["audio"] for item in test_dataset]
references = [item["text"] for item in test_dataset]

# 2. 加载本地 Whisper 模型
print("正在加载本地 Whisper-Large-V3 模型...")
asr = pipeline(
    "automatic-speech-recognition",
    model="../whisper-large-v3",
    torch_dtype=torch.float16, # 使用半精度减少显存占用
    device=0
)

# 3. 批量推理 (显存极度优化版)
print(f"开始识别 {len(audio_paths)} 条音频...")

hyps = []
# 将 batch_size 设为 1，确保 24GB 显存绝对够用
for out in tqdm(asr(audio_paths, batch_size=1, generate_kwargs={"language":"english", "task":"transcribe"}), total=len(audio_paths)):
    hyps.append(out["text"].strip())

# 4. 文本归一化并计算 WER / CER
def normalize(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

refs_norm = [normalize(t) for t in references]
hyps_norm = [normalize(h) for h in hyps]

wer = jiwer.wer(refs_norm, hyps_norm)
cer = jiwer.cer(refs_norm, hyps_norm)

print("\n==== 评测结果 (采样数: 100) ====")
print(f"WER (词错误率): {wer*100:.2f}%")
print(f"CER (字符错误率): {cer*100:.2f}%")

# 5. 保存识别结果
predictions = []
for i, item in enumerate(test_dataset):
    predictions.append({
        "id": item["id"],
        "audio": item["audio"],
        "reference_en": item["text"],
        "asr_text": hyps[i]
    })

output_path = "../outputs/asr_predictions.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)
    
print(f"\n✅ 识别结果已保存到 {output_path}")