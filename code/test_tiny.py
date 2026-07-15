import json
from transformers import pipeline

print("正在加载本地 Whisper-Tiny 模型...")
# 加载外层目录的 tiny 模型
tiny_asr = pipeline(
    "automatic-speech-recognition",
    model="../../whisper-tiny",  
    device=0
)

# 直接从之前生成好的 JSON 中读取一条绝对正确的音频路径
with open("../common_data/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    
test_audio = dataset[0]["audio"] # 取第一条音频

print(f"成功找到音频，开始提取字级时间戳：{test_audio}")
# 开始推理，开启 return_timestamps="word"
result = tiny_asr(test_audio, return_timestamps="word", generate_kwargs={"language":"english", "task":"transcribe"})

print("\n==== 提取结果 ====")
if "chunks" in result:
    for chunk in result["chunks"]:
        print(f"[{chunk['timestamp'][0]:.2f}s -> {chunk['timestamp'][1]:.2f}s]: {chunk['text']}")
else:
    print("未能提取到时间戳，请检查模型版本或参数。")