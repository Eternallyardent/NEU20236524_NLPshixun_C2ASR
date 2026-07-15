import os
import numpy as np
import soundfile as sf

base_dir = "../common_data/tess_audio/TESS Toronto emotional speech set data"

# 把两个人的台词分开
oaf_files = [
    f"{base_dir}/OAF_neutral/OAF_back_neutral.wav",       
    f"{base_dir}/OAF_angry/OAF_bath_angry.wav",           
    f"{base_dir}/OAF_happy/OAF_bean_happy.wav",           
]

yaf_files = [
    f"{base_dir}/YAF_neutral/YAF_back_neutral.wav",       
    f"{base_dir}/YAF_sad/YAF_bath_sad.wav",               
    f"{base_dir}/YAF_happy/YAF_bean_happy.wav",           
]

print("==== 正在合成【高密度连续语音】测试音频 ====")

combined_audio = []
target_sr = None

def append_speaker_block(file_list, loops=2):
    """让同一个说话人连续无缝说话，把语音块撑厚到 3 秒以上"""
    global target_sr
    for _ in range(loops):
        for f in file_list:
            if os.path.exists(f):
                data, sr = sf.read(f)
                if target_sr is None: target_sr = sr
                combined_audio.extend(data)

# 模拟真实的交谈回合：
for round_num in range(3):
    print(f"-> 第 {round_num + 1} 回合：老奶奶发言 (连续不间断)...")
    append_speaker_block(oaf_files, loops=2) # 连续说6个短句，无缝拼接
    
    # 加入 0.8 秒停顿，模拟换人
    combined_audio.extend(np.zeros(int(target_sr * 0.8), dtype=np.float32))
    
    print(f"-> 第 {round_num + 1} 回合：年轻女孩反驳 (连续不间断)...")
    append_speaker_block(yaf_files, loops=2)
    
    # 加入 0.8 秒停顿，模拟换人
    combined_audio.extend(np.zeros(int(target_sr * 0.8), dtype=np.float32))

output_file = "../common_data/mock_hard_conversation.wav"
sf.write(output_file, combined_audio, target_sr)

duration = len(combined_audio) / target_sr
print(f"\n✅ 完美测试音频生成完毕！总时长: {duration:.2f} 秒")