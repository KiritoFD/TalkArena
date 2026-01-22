import os
from pathlib import Path

# 模型存储路径
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ModelScope 模型配置
MODELS_CONFIG = {
    "llm": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",  # Qwen2.5-3B存在且性能好
        "cache_dir": "./models/qwen",
        "device": "auto",
        "trust_remote_code": True,
    },
    "tts": {
        # Modelscope 上的简化情感 TTS
        "model_id": "damo/speech_sambert-hifigan_tts_zh-cn_16k",  # 修复：使用正确的模型ID
        "cache_dir": "./models/chattts",
        "revision": "v1.0.0",
        "speaker": "8051",
        "default_style": "neutral",
        "emotion_styles": {
            "neutral": "neutral",
            "happy": "warm",
            "sad": "calm"
        }
    }
}

# 确保缓存目录存在
for model_type in MODELS_CONFIG:
    cache_dir = MODELS_CONFIG[model_type]["cache_dir"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
