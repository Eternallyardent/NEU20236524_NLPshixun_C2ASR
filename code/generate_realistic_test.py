import os
import numpy as np
import soundfile as sf
import random
import glob

# 1. 动态获取所有音频文件，避免重复
base_dir = "../common_data/tess_audio/TESS Toronto emotional speech set data"
# 获取 OAF(老奶奶) 和 YAF(年轻女孩) 的所有音频
oaf_files = glob.glob(f"{base_dir}/OAF_*/*.wav")
yaf_files = glob.glob(f"{base_dir}/YAF_*/*.wav")

# 随机打乱，确保每次生成的话都不一样
random.shuffle(oaf_files)
random.shuffle(yaf_files)

print("==== 正在合成【带真实底噪】的会议测试集 ====")

combined_audio = []
target_sr = None

def generate_white_noise(duration_sec, sr, amplitude=0.002):
    """生成模拟真实环境的白噪音，彻底消灭绝对静音陷阱！"""
    samples = int(duration_sec * sr)
    # 使用正态分布生成随机噪音，方差不为0，PyAnnote 再也不会崩溃了！
    noise = np.random.normal(0, amplitude, samples)
    return noise.astype(np.float32)

# 模拟 4 个对话回合 (老奶奶和女孩交替说话)
for round_num in range(4):
    print(f"-> 第 {round_num + 1} 回合开始...")
    
    # --- 老奶奶发言 (随机挑2-3句不同的话连在一起) ---
    sentences_to_say = random.randint(2, 3)
    for _ in range(sentences_to_say):
        if not oaf_files: break
        f = oaf_files.pop()
        data, sr = sf.read(f)
        if target_sr is None: target_sr = sr
        combined_audio.extend(data)
        
        # 句与句之间加入 0.2 秒的极短底噪呼吸停顿
        combined_audio.extend(generate_white_noise(0.2, target_sr))
    
    # 换人思考时间：加入 1.0 秒的真实环境底噪
    combined_audio.extend(generate_white_noise(1.0, target_sr))
    
    # --- 年轻女孩发言 (随机挑2-3句不同的话连在一起) ---
    sentences_to_say = random.randint(2, 3)
    for _ in range(sentences_to_say):
        if not yaf_files: break
        f = yaf_files.pop()
        data, sr = sf.read(f)
        combined_audio.extend(data)
        
        # 句与句之间加入 0.2 秒的极短底噪呼吸停顿
        combined_audio.extend(generate_white_noise(0.2, target_sr))
        
    # 换人思考时间：加入 0.8 秒的真实环境底噪
    combined_audio.extend(generate_white_noise(0.8, target_sr))

output_file = "../common_data/realistic_test_set.wav"
sf.write(output_file, combined_audio, target_sr)

duration = len(combined_audio) / target_sr
print(f"\n✅ 真实测试集生成完毕！总时长: {duration:.2f} 秒")
print("因为加入了底噪并随机化了句子，PyAnnote 和 Whisper 都将发挥出真实的实力！")