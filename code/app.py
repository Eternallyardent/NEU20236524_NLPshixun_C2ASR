import os
import re
import numpy as np
import warnings
import librosa
import gradio as gr
from funasr import AutoModel

# ================= 1. 离线配置与模型加载 =================
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODELSCOPE_ENVIRONMENT"] = "offline"

print("==== 1. 正在加载 SenseVoiceSmall 极速模型 ====")
sensevoice_path = os.path.abspath("../SenseVoiceSmall")
model = AutoModel(
    model=sensevoice_path,
    trust_remote_code=True,
    disable_update=True,
    device="cuda:0"
)

# ================= 2. 引入“缓冲池”的流式逻辑 =================
BUFFER_SECONDS = 3.0 

def process_audio(audio_chunk, state):
    history_text, audio_buffer = state

    if audio_chunk is None:
        return history_text, state

    sr, y = audio_chunk

    # 1. 音频格式转换
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32)
        
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    # 2. 缓冲音频碎片
    if audio_buffer.size == 0:
        audio_buffer = y
    else:
        audio_buffer = np.concatenate([audio_buffer, y])

    # 3. 拦截卡口：是否攒够时长
    current_duration = len(audio_buffer) / sr
    if current_duration < BUFFER_SECONDS:
        return history_text, [history_text, audio_buffer]

    # 4. 强制重采样 
    if sr != 16000:
        audio_16k = librosa.resample(audio_buffer, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio_buffer

    # 5. 音量门限过滤
    volume = np.max(np.abs(audio_16k))
    if volume < 0.02:  
        audio_buffer = np.array([], dtype=np.float32)
        return history_text, [history_text, audio_buffer]

    # 6. 召唤模型推理
    res = model.generate(
        input=audio_16k, 
        language="zh",  
        use_itn=True,   
        disable_pbar=True
    )
    
    # 7. 终极文本与标点清洗
    if res and len(res) > 0:
        raw_text = res[0].get('text', '')
        clean_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
        
        if clean_text and clean_text not in ["。", "，", ".", " "]:
            history_text += clean_text
            # 暴力清洗机械标点
            history_text = history_text.replace("。，", "。").replace("，。", "，").replace("。。", "。").replace("，，", "，")

    # 清空缓冲池，迎接下一波
    audio_buffer = np.array([], dtype=np.float32)

    return history_text, [history_text, audio_buffer]

# ================= 3. Web UI 搭建 =================
print("==== 2. 正在启动 Web 服务 ====")
with gr.Blocks(title="ASR流式翻译平台", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ ASR 流式翻译平台演示（中文）
        **引擎**：SenseVoiceSmall (50M) | **策略**：3秒动态缓冲池 (Buffer Pool)
        
        👉 **说明**：为了解决极短音频切片导致的上下文丢失问题，本架构引入了缓冲机制。您说话时，系统会默默收集声音，每攒够 **3秒钟** 输出一次字幕。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], streaming=True, label="本地麦克风信号")
        with gr.Column(scale=2):
            text_output = gr.Textbox(label="实时字幕", lines=8, placeholder="每攒够3秒语音，字幕将在此刷新...")

    initial_state = ["", np.array([], dtype=np.float32)]
    state = gr.State(value=initial_state)

    audio_input.stream(
        fn=process_audio, 
        inputs=[audio_input, state], 
        outputs=[text_output, state]
    )

    clear_btn = gr.Button("🗑️ 清空字幕与缓冲池")
    clear_btn.click(fn=lambda: ("", ["", np.array([], dtype=np.float32)]), inputs=None, outputs=[text_output, state])

# ================= 4. 对外广播服务 =================
demo.launch(server_name="0.0.0.0", server_port=7860)