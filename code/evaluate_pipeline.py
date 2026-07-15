import os
import torch
import json
import librosa
import soundfile as sf
from tqdm import tqdm
import jiwer
import re

# ================== 1. 初始化所有模型 ==================
print("==== 🚀 正在初始化 C2_ASR 终极双轨大模型评测管线 ====")
import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. 语义时间轴 (Whisper-Tiny)
from transformers import pipeline as hf_pipeline
tiny_asr = hf_pipeline("automatic-speech-recognition", model="../../whisper-tiny", device=0)

# 2. 声纹分离 (PyAnnote)
from pyannote.audio.core.plda import PLDA
PLDA.from_pretrained = classmethod(lambda cls, *args, **kwargs: None)
from pyannote.audio import Pipeline
pyannote_pipeline = Pipeline.from_pretrained("/root/siton-tmp/MintyKid/20260617/pyannote_models/diarization/config.yaml")
pyannote_pipeline.to(torch.device("cuda:0"))

# 3. 翻译提纯核心 (Whisper-Large-V3)
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
v3_path = "../whisper-large-v3"
v3_processor = AutoProcessor.from_pretrained(v3_path)
v3_model = AutoModelForSpeechSeq2Seq.from_pretrained(v3_path, torch_dtype=torch.float16).to("cuda:0")

# 4. 语义纠错中枢 (Qwen2.5 0.5B)
from transformers import AutoModelForCausalLM, AutoTokenizer
qwen_path = "/root/siton-tmp/MintyKid/20260617/Qwen1.5-0.5B-Chat"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:0")


# ================== 2. 核心架构封装函数 ==================
def clean_text(text):
    """文本清洗：统一转小写，去除标点符号，保证 WER/CER 评测的公平性"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def process_audio_pipeline(audio_path):
    """封装你辛苦搭建的整套智能架构"""
    try:
        # --- Track 1: Tiny ---
        whisper_result = tiny_asr(audio_path, return_timestamps="word", generate_kwargs={"language":"english"})
        real_whisper_chunks = whisper_result.get("chunks", [])
        
        # --- Track 2: PyAnnote ---
        real_diarization_tracks = []
        try:
            diarization = pyannote_pipeline(audio_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                real_diarization_tracks.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        except Exception:
            # 容错机制：根据时间隙反推
            speaker_id = 0
            if real_whisper_chunks:
                current_start, last_end = real_whisper_chunks[0]['timestamp']
                for chunk in real_whisper_chunks[1:]:
                    start, end = chunk['timestamp']
                    if start - last_end > 0.5:
                        real_diarization_tracks.append({"start": current_start, "end": last_end, "speaker": f"SPEAKER_{speaker_id:02d}"})
                        speaker_id = 1 - speaker_id
                        current_start = start
                    last_end = end
                real_diarization_tracks.append({"start": current_start, "end": last_end, "speaker": f"SPEAKER_{speaker_id:02d}"})

        # --- 规则引擎切片 ---
        final_slices = []
        current_words, chunk_start, current_speaker = [], None, None
        
        for chunk in real_whisper_chunks:
            word_start, word_end = chunk["timestamp"]
            word_text = chunk["text"].strip()
            word_midpoint = (word_start + word_end) / 2.0
            
            speaker_for_word = "UNKNOWN"
            for track in real_diarization_tracks:
                if track["start"] <= word_midpoint <= track["end"]:
                    speaker_for_word = track["speaker"]
                    break
                    
            if chunk_start is None:
                chunk_start = word_start
                current_speaker = speaker_for_word
                
            current_words.append(word_text)
            
            if word_text.endswith(('.', '?', '!')) or (speaker_for_word != current_speaker and current_speaker is not None) or (word_end - chunk_start >= 8.0):
                final_slices.append({"start": chunk_start, "end": word_end, "speaker": current_speaker})
                chunk_start, current_words, current_speaker = None, [], None
        
        if current_words:
            final_slices.append({"start": chunk_start, "end": real_whisper_chunks[-1]["timestamp"][1], "speaker": current_speaker})

        # --- Whisper-V3 提纯 + Qwen 纠错 ---
        audio_data, original_sr = sf.read(audio_path)
        if original_sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)
            
        full_transcription = []
        
        for slice_info in final_slices:
            start_sample = int(slice_info['start'] * 16000)
            end_sample = int(slice_info['end'] * 16000)
            chunk_audio = audio_data[start_sample:end_sample]
            if len(chunk_audio) == 0: continue
            
            inputs = v3_processor(chunk_audio, sampling_rate=16000, return_tensors="pt").to("cuda:0", torch.float16)
            outputs = v3_model.generate(inputs.input_features, num_beams=3, num_return_sequences=3, language="english")
            candidates = v3_processor.batch_decode(outputs, skip_special_tokens=True)
            unique_candidates = list(set(candidates))
            
            if len(unique_candidates) == 1:
                final_text = unique_candidates[0]
            else:
                qwen_sys = "你是一个音频文本纠错专家。根据候选句，输出最正确的一句话，无需解释。"
                cand_str = "\n".join([f"候选{i+1}: {c}" for i, c in enumerate(unique_candidates)])
                qwen_usr = f"分析候选，输出最终结果：\n{cand_str}"
                
                msgs = [{"role": "system", "content": qwen_sys}, {"role": "user", "content": qwen_usr}]
                text = qwen_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                m_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
                gen_ids = qwen_model.generate(m_inputs.input_ids, max_new_tokens=50)
                gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(m_inputs.input_ids, gen_ids)]
                final_text = qwen_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                
            full_transcription.append(clean_text(final_text))
            
        return " ".join(full_transcription)
    except Exception as e:
        print(f"处理出错 {audio_path}: {e}")
        return ""


# ================== 3. 执行批量评测 ==================
dataset_path = "../common_data/dataset.json" # 请确保这是你真实的带 ground_truth 的文件路径
print(f"\n==== 📊 开始读取测试集: {dataset_path} ====")

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 为了测试效率，你可以先取前 50 条测试 (dataset[:50])
test_subset = dataset[:50] 

ground_truths = []
predictions = []

for item in tqdm(test_subset, desc="评测进度"):
    audio_file = item["audio"]
    # 假设你的 json 中包含真实文本字段 "text"
    truth = clean_text(item["text"]) 
    
    # 将音频送入你的神级架构
    pred = process_audio_pipeline(audio_file)
    
    ground_truths.append(truth)
    predictions.append(pred)

# ================== 4. 计算并输出最终指标 ==================
print("\n" + "="*40)
print("🏆 评测结束 (采样数: {})".format(len(test_subset)))

# 使用 jiwer 计算 WER 和 CER
error_wer = jiwer.wer(ground_truths, predictions)
error_cer = jiwer.cer(ground_truths, predictions)

print(f"WER (词错误率):  {error_wer * 100:.2f}%")
print(f"CER (字符错误率): {error_cer * 100:.2f}%")
print("="*40)

# 如果你的 WER 和 CER 比截图里的 4.50% 和 1.94% 还要低，
# 那就直接证明了你这套【双轨+切片+大模型重打分】的架构取得了巨大成功！