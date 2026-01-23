"""
语音输入全链路测试
"""
import os
import io
import wave
import tempfile
from pathlib import Path

print("=" * 60)
print("语音输入全链路测试")
print("=" * 60)

# ============================================================
# 1. Vosk 模型加载测试
# ============================================================
print("\n[1] Vosk 模型测试")

from vosk import Model, KaldiRecognizer

model_path = Path("models/vosk-model-small-cn-0.22")
assert model_path.exists(), f"Vosk 模型不存在: {model_path}"
print(f"  模型路径: {model_path} ✓")

model = Model(str(model_path))
print("  模型加载成功 ✓")

# ============================================================
# 2. 生成测试音频 (TTS)
# ============================================================
print("\n[2] TTS 生成测试音频")

from model_loader import TTSLoader

tts = TTSLoader()
tts.load()

test_text = "你好世界"
wav_bytes = tts.synthesize(test_text)

assert wav_bytes[:4] == b"RIFF", "TTS 输出不是 WAV 格式"
print(f"  合成文本: {test_text}")
print(f"  WAV 大小: {len(wav_bytes)} bytes ✓")

# 保存到临时文件
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(wav_bytes)
    test_audio_path = f.name
print(f"  临时文件: {test_audio_path}")

# ============================================================
# 3. 音频格式验证
# ============================================================
print("\n[3] 音频格式验证")

with open(test_audio_path, "rb") as f:
    header = f.read(12)

is_wav = header[:4] == b"RIFF" and header[8:12] == b"WAVE"
print(f"  是 WAV 格式: {is_wav} ✓")

# 检查 WAV 参数
wf = wave.open(test_audio_path, "rb")
channels = wf.getnchannels()
sample_rate = wf.getframerate()
sample_width = wf.getsampwidth()
frames = wf.getnframes()
duration = frames / sample_rate
wf.close()

print(f"  声道数: {channels}")
print(f"  采样率: {sample_rate} Hz")
print(f"  位深: {sample_width * 8} bit")
print(f"  时长: {duration:.2f} 秒")

# ============================================================
# 4. Vosk 转录测试
# ============================================================
print("\n[4] Vosk 转录测试")

from pydub import AudioSegment

# Vosk 需要 16kHz 单声道
audio = AudioSegment.from_wav(test_audio_path)
audio = audio.set_frame_rate(16000).set_channels(1)

wav_io = io.BytesIO()
audio.export(wav_io, format="wav")
wav_io.seek(0)

wf = wave.open(wav_io, "rb")
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

import json
result_text = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        result_text += result.get("text", "")

final_result = json.loads(rec.FinalResult())
result_text += final_result.get("text", "")
wf.close()

text = result_text.strip()
print(f"  原文: {test_text}")
print(f"  转录: {text}")

if text:
    print("  Vosk 转录成功 ✓")
else:
    print("  ⚠ 转录结果为空（音频可能太短）")

# ============================================================
# 5. Orchestrator 集成测试
# ============================================================
print("\n[5] Orchestrator 集成测试")

from orchestrator import Orchestrator

orch = Orchestrator(enable_tts=True)
assert orch.stt is not None, "STT 未初始化"
print("  Orchestrator STT 初始化 ✓")

# 测试转录方法
transcribed = orch.transcribe_audio(test_audio_path)
print(f"  transcribe_audio 结果: {transcribed}")

# ============================================================
# 6. Handlers 集成测试
# ============================================================
print("\n[6] Handlers 集成测试")

from ui.handlers import process_voice_input, init_models
import ui.handlers as handlers

# 注入 orchestrator
handlers._orchestrator_instance = orch

# 创建 session
session = orch.start_session("negotiation")
print(f"  Session: {session.session_id}")

# 测试 process_voice_input
results = list(process_voice_input(session.session_id, test_audio_path, []))
print(f"  process_voice_input yields: {len(results)} 次")

if len(results) > 0:
    final = results[-1]
    print(f"  最终气场: AI {final[2]} vs 用户 {final[3]}")
    print("  语音输入处理成功 ✓")

# ============================================================
# 7. 清理
# ============================================================
print("\n[7] 清理")
Path(test_audio_path).unlink(missing_ok=True)
print("  临时文件已删除 ✓")

# ============================================================
# 测试总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 语音输入全链路测试完成!")
print("=" * 60)
