import os
import re
import numpy as np
import soundfile as sf
import librosa
import jiwer
import time
import torch
import gc
import json  # [NEW] 引入 json 模块
from tqdm import tqdm
import warnings

# ================= 0. 环境与配置 =================
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_ENVIRONMENT"] = "offline"

AUDIO_PATH = "../common_data/EN2001a.Mix-Headset.wav"
GT_PATH = "../common_data/EN2001a_ground_truth.txt"
WHISPER_PATH = "../whisper-large-v3"
SENSEVOICE_PATH = os.path.abspath("../SenseVoiceSmall")

# [NEW] 定义并创建输出目录
OUTPUT_DIR = "/root/siton-tmp/MintyKid/20260617/C2_ASR/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 输出目录已准备: {OUTPUT_DIR}")

# ================= 1. 数据增强与辅助函数 =================
def add_white_noise(audio, snr_db=10):
    signal_power = np.mean(audio ** 2)
    if signal_power == 0: return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)

def normalize_text(t): 
    return re.sub(r'[^\w\s]', '', t.lower()).strip()

# [NEW] 文件名净化函数（防止带有特殊字符导致保存失败）
def sanitize_filename(name):
    clean_name = re.sub(r'[ \(\)\.=]', '_', name)
    return re.sub(r'_+', '_', clean_name).strip('_')

# ================= 2. 准备 5 组多维实验数据 =================
print("==== 1. 正在加载并生成 5 组鲁棒性实验数据 (请稍候...) ====")
with open(GT_PATH, "r", encoding="utf-8") as f:
    ground_truth = normalize_text(f.read().strip())

audio_base, sr = librosa.load(AUDIO_PATH, sr=16000)
if len(audio_base.shape) > 1: audio_base = audio_base.mean(axis=1)

print("-> 生成低强度噪声 (SNR=20dB)...")
audio_noise_low = add_white_noise(audio_base, snr_db=20)

print("-> 生成高强度噪声 (SNR=5dB)...")
audio_noise_high = add_white_noise(audio_base, snr_db=5)

print("-> 生成中等加速 (1.25x)...")
audio_fast_125 = librosa.effects.time_stretch(y=audio_base, rate=1.25)

print("-> 生成极限加速 (1.50x)...")
audio_fast_150 = librosa.effects.time_stretch(y=audio_base, rate=1.5)

test_suite = {
    "1. 纯净原声 (Baseline)": audio_base,
    "2. 低强度噪声 (SNR=20dB)": audio_noise_low,
    "3. 高强度噪声 (SNR=5dB)": audio_noise_high,
    "4. 中等加速 (1.25倍速)": audio_fast_125,
    "5. 极限加速 (1.50倍速)": audio_fast_150
}

results_matrix = {"Whisper-V3": {}, "SenseVoice": {}}
vram_matrix = {}
chunk_len = 30 * 16000  

# ================= 3. Whisper-V3 专场测试 =================
print("\n" + "="*50)
print("🚀 阶段一：Whisper-Large-V3 测试")
print("="*50)

# 重置显存探针
torch.cuda.reset_peak_memory_stats()

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
w_processor = AutoProcessor.from_pretrained(WHISPER_PATH)
w_model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_PATH, torch_dtype=torch.float16).to("cuda:0")

for condition, audio in test_suite.items():
    start_t = time.time()
    pieces = []
    
    for i in tqdm(range(0, len(audio), chunk_len), desc=f"Whisper {condition}"):
        chunk = audio[i : i + chunk_len]
        if len(chunk) == 0: continue
        inputs = w_processor(chunk, sampling_rate=16000, return_tensors="pt").to("cuda:0", torch.float16)
        out = w_model.generate(inputs.input_features, language="english")
        text = w_processor.batch_decode(out, skip_special_tokens=True)[0]
        pieces.append(text.strip())
        
    fused_text = normalize_text(" ".join(pieces))
    wer = jiwer.wer(ground_truth, fused_text)
    cost_time = time.time() - start_t
    results_matrix["Whisper-V3"][condition] = {"WER": wer * 100, "Time": cost_time}
    
    # [NEW] 组装结果并输出为 JSON 文件
    output_data = {
        "model": "Whisper-Large-V3",
        "condition": condition,
        "wer_percent": round(wer * 100, 2),
        "time_seconds": round(cost_time, 2),
        "raw_text_output": " ".join(pieces)
    }
    file_name = f"Whisper_{sanitize_filename(condition)}.json"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

# 抓取 Whisper 峰值显存
vram_matrix["Whisper-V3"] = torch.cuda.max_memory_allocated() / (1024**3)

print("\n-> 🧹 Whisper 测试完毕，释放显存...")
del w_model
del w_processor
gc.collect()
torch.cuda.empty_cache()


# ================= 4. SenseVoice 专场测试 =================
print("\n" + "="*50)
print("⚡ 阶段二：SenseVoiceSmall 测试")
print("="*50)

torch.cuda.reset_peak_memory_stats()

from funasr import AutoModel
s_model = AutoModel(model=SENSEVOICE_PATH, trust_remote_code=True, disable_update=True, device="cuda:0")

for condition, audio in test_suite.items():
    start_t = time.time()
    pieces = []
    
    for i in tqdm(range(0, len(audio), chunk_len), desc=f"SenseVoice {condition}"):
        chunk = audio[i : i + chunk_len]
        if len(chunk) == 0: continue
        res = s_model.generate(input=chunk, disable_pbar=True)
        if res and len(res) > 0:
            raw = res[0].get('text', '')
            pieces.append(re.sub(r'<\|.*?\|>', '', raw).strip())
            
    fused_text = normalize_text(" ".join(pieces))
    wer = jiwer.wer(ground_truth, fused_text)
    cost_time = time.time() - start_t
    results_matrix["SenseVoice"][condition] = {"WER": wer * 100, "Time": cost_time}
    
    # [NEW] 组装结果并输出为 JSON 文件
    output_data = {
        "model": "SenseVoiceSmall",
        "condition": condition,
        "wer_percent": round(wer * 100, 2),
        "time_seconds": round(cost_time, 2),
        "raw_text_output": " ".join(pieces)
    }
    file_name = f"SenseVoice_{sanitize_filename(condition)}.json"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

# 抓取 SenseVoice 峰值显存
vram_matrix["SenseVoice"] = torch.cuda.max_memory_allocated() / (1024**3)


# ================= 5. 输出终极数据对比大表 =================
print("\n\n" + "█"*70)
print("📊 ASR 鲁棒性终极对比报告 (45分钟会议长音频)")
print("█"*70)

# 先打印显存开销对比
print("\n【💾 算力与显存开销对比】")
print(f"  > Whisper-Large-V3 (1.5B)  | 峰值显存占用: {vram_matrix['Whisper-V3']:.2f} GB")
print(f"  > SenseVoiceSmall (50M)    | 峰值显存占用: {vram_matrix['SenseVoice']:.2f} GB")
print("-" * 70)

for condition in test_suite.keys():
    print(f"\n【测试环境：{condition}】")
    w_res = results_matrix["Whisper-V3"][condition]
    s_res = results_matrix["SenseVoice"][condition]
    
    print(f"  > Whisper-Large-V3  | WER: {w_res['WER']:>6.2f}% | 耗时: {w_res['Time']:>5.1f} 秒")
    print(f"  > SenseVoiceSmall   | WER: {s_res['WER']:>6.2f}% | 耗时: {s_res['Time']:>5.1f} 秒")
print("-" * 70)
print(f"\n✅ 所有详细转录文本已成功导出至 JSON 文件目录: {OUTPUT_DIR}")