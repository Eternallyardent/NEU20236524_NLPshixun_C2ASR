import os
import json
import torch
import jiwer
import re
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import time
import warnings
import gc

# 🛑 核心护航：强行闭嘴！屏蔽掉所有吓人的 transformers 红色警告
import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"

from pyannote.audio.core.plda import PLDA
PLDA.from_pretrained = classmethod(lambda cls, *args, **kwargs: None)
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# ================= 1. 加载真实会议数据 =================
print("==== 1. 正在加载 AMI 45分钟真实会议数据 ====")
test_audio_path = "../common_data/EN2001a.Mix-Headset.wav"
if not os.path.exists(test_audio_path): raise FileNotFoundError(f"找不到: {test_audio_path}")
with open("../common_data/EN2001a_ground_truth.txt", "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

# ================= 2. 优先进行智能切片 (避开显存死锁) =================
print("\n==== 2. 正在提取语义时间轴与声纹特征 ====")
tiny_asr = hf_pipeline("automatic-speech-recognition", model="../../whisper-tiny", device=0, chunk_length_s=30)
pyannote_pipeline = Pipeline.from_pretrained("/root/siton-tmp/MintyKid/20260617/pyannote_models/diarization/config.yaml").to(torch.device("cuda:0"))

print("⏳ [系统提示] Tiny_ASR 正在后台静默处理 45 分钟长音频。")
print("⏳ [系统提示] 这一步没有任何进度条，开机冷启动大概需要 2 到 5 分钟。")
print("⏳ [系统提示] 请去喝口水，千万千万不要按 Ctrl+C ！！！...")
start_time = time.time()

# 黑盒操作开始
w_res = tiny_asr(test_audio_path, return_timestamps="word", generate_kwargs={"language":"english"})
w_chunks = w_res.get("chunks", [])

print(f"✅ Tiny 提取完毕！耗时: {int(time.time() - start_time)} 秒。")
print("⏳ [系统提示] 正在提取 PyAnnote 声纹，大约需要 1-2 分钟，请稍候...")
start_time = time.time()

diar_tracks = []
try:
    for turn, _, speaker in pyannote_pipeline(test_audio_path).itertracks(yield_label=True):
        diar_tracks.append({"start": turn.start, "end": turn.end, "speaker": speaker})
except Exception: pass 
print(f"✅ 声纹提取完毕！耗时: {int(time.time() - start_time)} 秒。")

# --- 物理切片打包 ---
base_slices = []
current_words, chunk_start, current_speaker = [], None, None
for chunk in w_chunks:
    if not chunk.get("timestamp") or len(chunk["timestamp"]) < 2: continue
    w_start, w_end, w_text = chunk["timestamp"][0], chunk["timestamp"][1], chunk["text"].strip()
    spk = "UNKNOWN"
    for track in diar_tracks:
        if track["start"] <= (w_start + w_end)/2.0 <= track["end"]: spk = track["speaker"]; break
    if chunk_start is None: chunk_start, current_speaker = w_start, spk
    current_words.append(w_text)
    if w_text.endswith(('.', '?', '!')) or (spk != current_speaker and current_speaker is not None) or (w_end - chunk_start >= 12.0):
        base_slices.append({"start": chunk_start, "end": w_end})
        chunk_start, current_words, current_speaker = None, [], None
if current_words: base_slices.append({"start": chunk_start, "end": w_chunks[-1]["timestamp"][1]})

super_chunks, current_super, current_duration = [], [], 0.0
PAD_SEC = 0.1
for slc in base_slices:
    duration = slc['end'] - slc['start']
    if current_duration + duration + PAD_SEC > 28.0 and current_super:
        super_chunks.append(current_super)
        current_super, current_duration = [slc], duration
    else:
        current_super.append(slc)
        current_duration += duration + PAD_SEC
if current_super: super_chunks.append(current_super)
print(f"-> 📦 动态重组完成：成功压缩为 {len(super_chunks)} 个高密度超级块！")

# 🧹 卸载轻量模型，清空显存给 V3 让路
del tiny_asr
del pyannote_pipeline
gc.collect()
torch.cuda.empty_cache()

# ================= 3. 加载终极大模型 Whisper V3 =================
print("\n==== 3. 正在安全加载声学大模型 Whisper-Large-V3 ====")
v3_path = "../whisper-large-v3"
v3_processor = AutoProcessor.from_pretrained(v3_path)
v3_model = AutoModelForSpeechSeq2Seq.from_pretrained(v3_path, torch_dtype=torch.float16).to("cuda:0")

# ================= 4. 对照组：原生 V3 =================
print("\n==== 4. 开始测试 [对照组：原生 Whisper-V3 (机械 30s 切片)] ====")
audio_data_base, sr_base = librosa.load(test_audio_path, sr=16000)
chunk_samples = 30 * 16000  
baseline_pieces = []
for i in tqdm(range(0, len(audio_data_base), chunk_samples), desc="原生 V3 进度"):
    chunk = audio_data_base[i : i + chunk_samples]
    if len(chunk) == 0: continue
    inputs = v3_processor(chunk, sampling_rate=16000, return_tensors="pt").to("cuda:0", torch.float16)
    outputs = v3_model.generate(inputs.input_features, language="english")
    baseline_pieces.append(v3_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip())
baseline_text = " ".join(baseline_pieces)

# ================= 5. 实验组：超级块重组纯净版 =================
print("\n==== 5. 开始测试 [实验组：智能切片 + 块重组 + V3纯净提纯] ====")
audio_data, _ = sf.read(test_audio_path)
silence_pad = np.zeros(int(PAD_SEC * 16000), dtype=np.float32) 
fused_pieces = []
for super_chunk in tqdm(super_chunks, desc="实验组进度"):
    combined_audio = []
    for i, slc in enumerate(super_chunk):
        chunk_audio = audio_data[int(slc['start'] * 16000) : int(slc['end'] * 16000)]
        combined_audio.extend(chunk_audio)
        if i < len(super_chunk) - 1: combined_audio.extend(silence_pad) 
    if len(combined_audio) == 0: continue
    
    inputs = v3_processor(np.array(combined_audio, dtype=np.float32), sampling_rate=16000, return_tensors="pt").to("cuda:0", torch.float16)
    outputs = v3_model.generate(inputs.input_features, language="english")
    fused_pieces.append(v3_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip())
fused_text = " ".join(fused_pieces)

# ================= 6. 计算评测指标 =================
def normalize(t): return re.sub(r'[^\w\s]', '', t.lower()).strip()
print("\n==== 6. 正在计算最终评测指标 ====")
norm_truth = normalize(ground_truth)
norm_base = normalize(baseline_text)
norm_fused = normalize(fused_text)
base_wer, base_cer = jiwer.wer(norm_truth, norm_base), jiwer.cer(norm_truth, norm_base)
fused_wer, fused_cer = jiwer.wer(norm_truth, norm_fused), jiwer.cer(norm_truth, norm_fused)

print("\n" + "█"*65)
print("🏆 最终实训验收成果：纯智能切片 VS 机械切片 评测结果")
print("█"*65)
print(f"【对照组：单纯 Whisper-V3 (原生机械 30s 切割)】")
print(f"WER: {base_wer*100:.2f}% | CER: {base_cer*100:.2f}%")
print("-" * 65)
print(f"【实验组：智能 VAD 切片 + 28s超级块重组 】")
print(f"WER: {fused_wer*100:.2f}% | CER: {fused_cer*100:.2f}%")
print("█"*65)