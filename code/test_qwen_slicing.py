import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 模拟我们前面两个 Track 的输出数据 =================
# 在实际运行中，这些数据将来自你调用 tiny_asr 和 pipeline 的结果
# 为了方便测试切分逻辑，我们这里伪造一段“两人抢话”的长音频数据
mock_whisper_chunks = [
    {"timestamp": (0.0, 0.5), "text": " 这个"},
    {"timestamp": (0.5, 0.9), "text": " 项目"},
    {"timestamp": (0.9, 1.5), "text": " 预算"},
    {"timestamp": (1.5, 1.8), "text": " 是"},
    {"timestamp": (1.8, 2.5), "text": " 多少？"},
    {"timestamp": (2.2, 2.8), "text": " 我觉得"},  # 注意这里发生了抢话
    {"timestamp": (2.8, 3.5), "text": " 至少"},
    {"timestamp": (3.5, 4.0), "text": " 五十万。"}
]

# 模拟 PyAnnote 的声纹区间 (Speaker_00 在问，Speaker_01 抢答)
mock_diarization_tracks = [
    {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
    {"start": 2.2, "end": 4.0, "speaker": "SPEAKER_01"} 
]

# ================= 2. 时间轴对齐算法 (空间缝合) =================
def align_tracks(whisper_chunks, diarization_tracks):
    aligned_sentences = []
    current_speaker = None
    current_sentence = ""
    start_time = 0.0
    
    for chunk in whisper_chunks:
        word_start, word_end = chunk["timestamp"]
        word_midpoint = (word_start + word_end) / 2.0
        word_text = chunk["text"].strip()
        
        # 寻找这个词属于哪个说话人
        speaker_for_word = "UNKNOWN"
        for track in diarization_tracks:
            if track["start"] <= word_midpoint <= track["end"]:
                speaker_for_word = track["speaker"]
                break
                
        # 如果说话人没变，就把词拼接到当前句子里
        if speaker_for_word == current_speaker:
            current_sentence += " " + word_text
        else:
            # 说话人变了，保存上一句话（如果存在）
            if current_speaker is not None:
                aligned_sentences.append(f"[{start_time:.2f}s - {word_start:.2f}s] {current_speaker}: {current_sentence}")
            # 开启新的一句话
            current_speaker = speaker_for_word
            current_sentence = word_text
            start_time = word_start
            
    # 把最后一句也加进去
    if current_speaker is not None:
         aligned_sentences.append(f"[{start_time:.2f}s - {whisper_chunks[-1]['timestamp'][1]:.2f}s] {current_speaker}: {current_sentence}")
         
    return "\n".join(aligned_sentences)

# 执行对齐
aligned_transcript = align_tracks(mock_whisper_chunks, mock_diarization_tracks)
print("==== 步骤1：前端双轨对齐结果 ====")
print(aligned_transcript)


# ================= 3. LLM 智能中枢切片 =================
print("\n==== 步骤2：正在加载 Qwen2.5 (0.5B) 大脑... ====")
# 从你截图中的目录加载本地 Qwen 模型
qwen_path = "/root/siton-tmp/MintyKid/20260617/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(qwen_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

# 编写给 Qwen 的 Prompt (核心提示词)
system_prompt = """你是一个专业的音频切片助手。
你的任务是根据带有时间戳和说话人标记的语音转写记录，决定如何将长音频切分成独立的短音频块(Chunk)。
切分原则：
1. 必须在句子语义完整的地方切断。
2. 尽量在说话人切换的地方切断。
3. 避免在句子中间或者抢话中途切断。
请输出合理的切片时间表，格式为：
[切片1] 开始时间 - 结束时间 : 完整内容
[切片2] 开始时间 - 结束时间 : 完整内容
"""

user_prompt = f"以下是对齐后的音频时间轴数据：\n{aligned_transcript}\n\n请给出最优的智能切片方案。"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("正在思考切片方案...")
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n==== 步骤3：Qwen 智能切片决策结果 ====")
print(response)
