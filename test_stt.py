"""测试语音识别功能"""
import os
import io
from pathlib import Path

def find_audio_files():
    audio_dir = Path("outputs/audio")
    return list(audio_dir.rglob("*.wav"))

def test_audio_format(audio_path: str):
    with open(audio_path, "rb") as f:
        header = f.read(12)
    
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        print(f"✓ {audio_path}: WAV")
        return "wav"
    elif header[:2] in (b"\xff\xfb", b"\xff\xf3"):
        print(f"→ {audio_path}: MP3")
        return "mp3"
    return None

def convert_mp3_to_wav(mp3_path: str) -> bytes:
    from pydub import AudioSegment
    print(f"[转换] {mp3_path}")
    audio = AudioSegment.from_mp3(mp3_path)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()

def test_stt(audio_path: str, is_mp3: bool = False):
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    print(f"\n[STT] {audio_path}")
    
    if is_mp3:
        wav_bytes = convert_mp3_to_wav(audio_path)
        audio_source = io.BytesIO(wav_bytes)
    else:
        audio_source = audio_path
    
    with sr.AudioFile(audio_source) as source:
        audio = recognizer.record(source)
    
    text = recognizer.recognize_google(audio, language="zh-CN")
    print(f"✓ 结果: {text}")
    return text

if __name__ == "__main__":
    print("=" * 50)
    print("TalkArena STT 测试")
    print("=" * 50)
    
    audio_files = find_audio_files()
    print(f"\n找到 {len(audio_files)} 个音频文件\n")
    
    for f in audio_files[:3]:
        fmt = test_audio_format(str(f))
        if fmt:
            test_stt(str(f), is_mp3=(fmt == "mp3"))