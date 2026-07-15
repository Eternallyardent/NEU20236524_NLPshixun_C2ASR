import os
import json

# 1. 设置音频所在的真实解压目录和输出的 json 路径
data_dir = "../common_data/tess_audio"
output_json = "../common_data/dataset.json"
dataset = []

# 2. 遍历所有音频文件
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            # TESS 数据集命名规则：OAF_词汇_情感.wav，例如 OAF_back_angry.wav
            # 对应的标准发音文本是 "Say the word <词汇>."
            parts = file.replace(".wav", "").split('_')
            if len(parts) >= 2:
                word = parts[1]
                text = f"Say the word {word}."
            else:
                text = "unknown"

            dataset.append({
                "id": file.replace(".wav", ""),
                "audio": audio_path,
                "text": text
            })

# 3. 将数据写入 dataset.json
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"✅ 数据准备完成！共找到 {len(dataset)} 条音频，已保存至 {output_json}")