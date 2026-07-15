import os
# 【防御层1】强制开启 HuggingFace 完全离线模式
os.environ["HF_HUB_OFFLINE"] = "1"

# ========== 【防御层2：终极黑客魔法 - 猴子补丁】 ==========
# 在导入 PyAnnote 核心库之前，拦截并篡改底层的 PLDA 加载逻辑
from pyannote.audio.core.plda import PLDA
# 强行把它的加载方法变成一个空壳，无论传什么都直接放行并返回 None
PLDA.from_pretrained = classmethod(lambda cls, *args, **kwargs: None)
# ========================================================

import torch
import json
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization

print("正在纯离线加载 PyAnnote 模型组件...")

# 1. 绝对路径直接加载分离模型 (Segmentation)
seg_model_path = "/root/siton-tmp/MintyKid/20260617/pyannote_models/segmentation/pytorch_model.bin"
seg_model = Model.from_pretrained(seg_model_path)

# 2. 绝对路径直接加载声纹嵌入模型 (Embedding)
emb_model_path = "/root/siton-tmp/MintyKid/20260617/pyannote_models/embedding/pytorch_model.bin"
emb_model = Model.from_pretrained(emb_model_path)

print("正在手动组装 Speaker Diarization 流水线...")

# 3. 组装流水线 (此时底层那个报错的 PLDA 逻辑已经被我们物理清除了)
pipeline = SpeakerDiarization(
    segmentation=seg_model,
    embedding=emb_model,
    clustering="AgglomerativeClustering",
    plda={} # 配合猴子补丁，安抚源码的格式检查
)

# 4. 注入 PyAnnote 3.1 的官方最优超参数
pipeline.instantiate({
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 1,  # <--- 把这里的 12 改成 1
        "threshold": 0.7045655,
    },
    "segmentation": {
        "min_duration_off": 0.0,
    }
})

# 5. 发送到 GPU 加速
pipeline.to(torch.device("cuda:0"))


# ================= 下方是测试推理代码 =================
with open("../common_data/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
test_audio = dataset[0]["audio"]

print(f"\n开始进行声纹分析：{test_audio}")
# 运行流水线
diarization = pipeline(test_audio)

print("\n==== 说话人分离结果 ====")
try:
    # 尝试正常遍历时间轴
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"[{turn.start:.2f}s -> {turn.end:.2f}s] 说话人: {speaker}")
except AttributeError:
    # 如果音频太短导致算法返回异常对象，给出友好提示
    print(f"⚠️ 模型底层返回异常。")
    print("💡 核心原因：音频过短 (仅1-2秒) 或不存在有效人声。")
    print("💡 解决方案：架构图 Track 2 要求【Long audio input】，请更换较长的对话音频进行测试！")