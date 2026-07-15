import os
import re
import numpy as np
import soundfile as sf
import jiwer
import warnings
import time
from tqdm import tqdm

# ================= 1. 彻底封锁联网与离线配置 =================
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_ENVIRONMENT"] = "offline"

print("==== 1. 正在加载 AMI 45分钟真实会议标准答案 ====")
gt_path = "../common_data/EN2001a_ground_truth.txt"
with open(gt_path, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

# ================= 2. 加载 SenseVoiceSmall =================
print("==== 2. 正在加载 SenseVoiceSmall (离线环境) ====")
from funasr import AutoModel 

sensevoice_path = os.path.abspath("../SenseVoiceSmall") 
sense_model = AutoModel(
    model=sensevoice_path,
    trust_remote_code=True,
    disable_update=True,
    device="cuda:0"
)

# ================= 3. 机械切块防爆显存推理 =================
print("\n==== 3. 开始执行 SenseVoiceSmall 防爆显存推理 ====")
test_audio_path = "../common_data/EN2001a.Mix-Headset.wav"

start_time = time.time()
audio_data, sr = sf.read(test_audio_path)

# 强制转换为单声道 float32
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)
audio_data = np.array(audio_data, dtype=np.float32)

# 核心防御：每 30 秒机械切一刀
chunk_samples = 30 * sr 
fused_pieces = []

for i in tqdm(range(0, len(audio_data), chunk_samples), desc="SenseVoice 极速推理进度"):
    chunk = audio_data[i : i + chunk_samples]
    if len(chunk) == 0: continue
    
    res = sense_model.generate(input=chunk, disable_pbar=True)
    if not res or len(res) == 0: continue
        
    raw_text = res[0].get('text', '')
    clean_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
    if clean_text:
        fused_pieces.append(clean_text)

end_time = time.time()
sensevoice_duration = end_time - start_time
fused_text = " ".join(fused_pieces)

# ================= 4. 指标绝对公平计算 =================
def normalize(t): 
    return re.sub(r'[^\w\s]', '', t.lower()).strip()

print("\n==== 4. 正在计算评测指标 ====")
norm_truth = normalize(ground_truth)
norm_fused = normalize(fused_text)

fused_wer = jiwer.wer(norm_truth, norm_fused)
fused_cer = jiwer.cer(norm_truth, norm_fused)

print("\n" + "="*65)
print("🏆 独立对比组：SenseVoiceSmall (50M) 独立评测结果")
print("="*65)
print(f"WER (词错误率): {fused_wer*100:.2f}%")
print(f"CER (字符错误率): {fused_cer*100:.2f}%")
print(f"⚡ 推理总耗时: {sensevoice_duration:.2f} 秒")
print("="*65)