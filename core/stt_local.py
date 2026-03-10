import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly
from vosk import KaldiRecognizer, Model

from core.multimodal_analyzer import VoiceEmotionAnalyzer


class LocalSTTService:
    def __init__(self):
        self.sample_rate = 16000
        self.model_dir = self._resolve_model_dir()
        self.model = None
        self.voice_analyzer = VoiceEmotionAnalyzer()

    def _resolve_model_dir(self) -> Path:
        env_dir = os.getenv("LOCAL_STT_MODEL_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        return Path("models") / "stt" / "vosk-cn"

    def _resolve_model_url(self) -> str:
        return (
            os.getenv("VOSK_MODEL_URL", "").strip()
            or "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
        )

    def _ensure_model(self) -> Path:
        if self.model_dir.exists():
            return self.model_dir

        self.model_dir.mkdir(parents=True, exist_ok=True)
        url = self._resolve_model_url()
        zip_path = self.model_dir.parent / "vosk-cn.zip"
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.model_dir.parent)

        try:
            zip_path.unlink()
        except Exception:
            pass

        # If the zip created a nested folder, use it as model_dir.
        subdirs = [p for p in self.model_dir.parent.iterdir() if p.is_dir()]
        for sd in subdirs:
            if sd.name.startswith("vosk-model") and sd != self.model_dir:
                if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
                    self.model_dir = sd
                break
        return self.model_dir

    def load(self):
        self._ensure_model()
        self.model = Model(str(self.model_dir))

    def _read_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.sample_rate:
            return audio
        return resample_poly(audio, self.sample_rate, sr)

    def _to_pcm16(self, audio: np.ndarray) -> bytes:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def transcribe(self, audio_bytes: bytes) -> Dict:
        if self.model is None:
            self.load()

        audio, sr = self._read_audio(audio_bytes)
        audio = self._resample(audio, sr)

        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        rec.AcceptWaveform(self._to_pcm16(audio))
        result = json.loads(rec.FinalResult() or "{}")

        voice_features = self.voice_analyzer.analyze(audio)
        voice_features_dict = voice_features.to_dict()

        return {
            "text": result.get("text", "").strip(),
            "result": result.get("result", []),
            "voice_features": voice_features_dict,
        }
