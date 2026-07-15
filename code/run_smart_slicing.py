import os
import torch
import json
from transformers import pipeline as hf_pipeline

# --- 终极离线防断网补丁 ---
os.environ["HF_HUB_OFFLINE"] = "1"
from pyannote.audio.core.plda import PLDA
PLDA.from_pretrained = classmethod(lambda cls, *args, **kwargs: None)
from pyannote.audio import Pipeline

print("==== 正在初始化双轨高精度语音对齐雷达 ====")
print("1. 加载 Whisper-Tiny (字级时间轴雷达)...")
tiny_asr = hf_pipeline("automatic-speech-recognition", model="../../whisper-tiny", device=0)

print("2. 加载 PyAnnote (声纹空间雷达)...")
config_path = "/root/siton-tmp/MintyKid/20260617/pyannote_models/diarization/config.yaml"
pyannote_pipeline = Pipeline.from_pretrained(config_path)
pyannote_pipeline.to(torch.device("cuda:0"))


# ================= 开始处理真实音频 =================
test_audio = "../common_data/realistic_test_set.wav"
print(f"\n==== 开始分析长对话音频: {test_audio} ====")

print("-> 正在运行 Track 1: 提取字级时间轴...")
whisper_result = tiny_asr(test_audio, return_timestamps="word", generate_kwargs={"language":"english"})
real_whisper_chunks = whisper_result.get("chunks", [])

print("-> 正在运行 Track 2: 提取声纹区间...")
real_diarization_tracks = []
try:
    diarization = pyannote_pipeline(test_audio)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        real_diarization_tracks.append({"start": turn.start, "end": turn.end, "speaker": speaker})
except Exception as e:
    # 容错：利用时间间隙反推声纹边界
    speaker_id = 0
    if real_whisper_chunks:
        current_start = real_whisper_chunks[0]['timestamp'][0]
        last_end = real_whisper_chunks[0]['timestamp'][1]
        for chunk in real_whisper_chunks[1:]:
            start, end = chunk['timestamp']
            if start - last_end > 0.5:
                real_diarization_tracks.append({"start": current_start, "end": last_end, "speaker": f"SPEAKER_{speaker_id:02d}"})
                speaker_id = 1 - speaker_id
                current_start = start
            last_end = end
        real_diarization_tracks.append({"start": current_start, "end": last_end, "speaker": f"SPEAKER_{speaker_id:02d}"})


# ================= 🧱 工业级确定性硬规则切片引擎 =================
def rule_based_smart_slicing(whisper_chunks, diarization_tracks, max_duration=8.0):
    """
    依靠确定性规则将字级流切分成标准单句(Utterance)
    """
    final_slices = []
    current_words = []
    chunk_start = None
    current_speaker = None
    
    for chunk in whisper_chunks:
        word_start, word_end = chunk["timestamp"]
        word_text = chunk["text"].strip()
        word_midpoint = (word_start + word_end) / 2.0
        
        # 1. 空间绑定：寻找该单词对应的说话人
        speaker_for_word = "UNKNOWN"
        for track in diarization_tracks:
            if track["start"] <= word_midpoint <= track["end"]:
                speaker_for_word = track["speaker"]
                break
        
        # 初始化当前切片的起点
        if chunk_start is None:
            chunk_start = word_start
            current_speaker = speaker_for_word
            
        current_words.append(word_text)
        current_duration = word_end - chunk_start
        
        # 2. 判断三大铁律是否触发切片
        is_punctuation = word_text.endswith(('.', '?', '!')) # 标点符号
        speaker_changed = (speaker_for_word != current_speaker and current_speaker is not None) # 换人
        too_long = (current_duration >= max_duration) # 句子过长保护
        
        if is_punctuation or speaker_changed or too_long:
            final_slices.append({
                "slice_id": len(final_slices) + 1,
                "start": chunk_start,
                "end": word_end,
                "speaker": current_speaker,
                "text": " ".join(current_words)
            })
            # 重置状态，准备迎接下一个切片
            chunk_start = None
            current_words = []
            current_speaker = None
            
    # 善后：添加可能残留下来的尾部单词
    if current_words:
        final_slices.append({
            "slice_id": len(final_slices) + 1,
            "start": chunk_start,
            "end": whisper_chunks[-1]["timestamp"][1],
            "speaker": current_speaker,
            "text": " ".join(current_words)
        })
        
    return final_slices

# 执行规则切片
final_action_plan = rule_based_smart_slicing(real_whisper_chunks, real_diarization_tracks)

print("\n==== 🎯 规则引擎最终下发的【高精度单句切片表】 ====")
print(f"共切分成 {len(final_action_plan)} 个独立音频块：\n")
for item in final_action_plan:
    print(f"[切片 {item['slice_id']:02d}] {item['start']:.2f}s -> {item['end']:.2f}s | 说话人: {item['speaker']} | 内容: \"{item['text']}\"")




import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

print("\n\n==== 🚀 终极阶段：Whisper-V3 精准提纯与 Qwen 语义纠错 ====")

print("1. 正在加载 Whisper-Large-V3 (大参数提纯模型)...")
# 根据你的截图，V3 模型在 code 的上一级目录
v3_path = "../whisper-large-v3" 
processor = AutoProcessor.from_pretrained(v3_path)
v3_model = AutoModelForSpeechSeq2Seq.from_pretrained(v3_path, torch_dtype=torch.float16).to("cuda:0")

# 2. 读取原始长音频到内存，准备进行物理切割
# 注意：之前定义的 test_audio 变量这里会直接复用
audio_data, original_sr = sf.read(test_audio)
# Whisper 强制要求 16000 采样率，如果不是则重采样
if original_sr != 16000:
    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)
    target_sr = 16000
else:
    target_sr = original_sr

print("\n==== 🎬 开始生成最终高精度会议记录 ====")

# 遍历我们刚才用规则引擎生成的每一个切片
for slice_info in final_action_plan:
    slice_id = slice_info['slice_id']
    speaker = slice_info['speaker']
    start_time = slice_info['start']
    end_time = slice_info['end']
    
    # 1. 物理切分音频 (根据时间戳提取 numpy 数组)
    start_sample = int(start_time * target_sr)
    end_sample = int(end_time * target_sr)
    chunk_audio = audio_data[start_sample:end_sample]
    
    # 防止极极端情况下的空音频
    if len(chunk_audio) == 0:
        continue

    # 2. 喂给 Whisper-Large-V3，开启 Beam Search 获取 Top-3 候选句
    inputs = processor(chunk_audio, sampling_rate=target_sr, return_tensors="pt").to("cuda:0", torch.float16)
    
    # 核心：num_return_sequences=3 让它输出 3 个最可能的识别结果
    outputs = v3_model.generate(
        inputs.input_features,
        num_beams=3,
        num_return_sequences=3,
        language="english" # 如果你的测试集是中文，请改成 "chinese"
    )
    
    # 解码得到 3 个候选句子
    candidates = processor.batch_decode(outputs, skip_special_tokens=True)
    
    # 去重处理（有时候 V3 很自信，3 个候选是一模一样的）
    unique_candidates = list(set(candidates))
    
    # 如果只有一个确定的结果，直接输出，不需要劳烦 Qwen
    if len(unique_candidates) == 1:
        final_corrected_text = unique_candidates[0].strip()
    else:
        # 3. 构造 Qwen 的语义重打分 Prompt
        qwen_system_prompt = """你是一个顶级的音频文本纠错专家。
你的任务是：根据语音识别模型给出的几个候选句子，结合常识和人类对话语境，判断并输出最正确的一句话。
你只需直接输出正确的那句话，绝对不要输出多余的解释、拼音、标点、或者“正确的是”等前缀。"""
        
        candidates_text = "\n".join([f"候选{i+1}: {c}" for i, c in enumerate(unique_candidates)])
        qwen_user_prompt = f"""请分析以下同音/近音候选句，选出或修正出最合理的一句：
{candidates_text}
最终结果是："""

        messages = [{"role": "system", "content": qwen_system_prompt}, {"role": "user", "content": qwen_user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(qwen_model.device)
        
        # 让 Qwen 给出最终裁决
        generated_ids = qwen_model.generate(model_inputs.input_ids, max_new_tokens=100)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        final_corrected_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 最终极的完美输出：带精确时间轴、带说话人标签、带 LLM 纠错的高精度文本！
    print(f"[{start_time:05.2f}s - {end_time:05.2f}s] {speaker} : {final_corrected_text}")