import io
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import requests


class LocalTTSService:
    def __init__(self):
        self.device = (os.getenv("LOCAL_TTS_DEVICE", "") or "cpu").lower()
        self.sample_rate = int(os.getenv("LOCAL_TTS_SAMPLE_RATE", "24000"))
        self.engine = None
        self.model_dir = self._resolve_model_dir()
        self.model_url = os.getenv("STYLETTS2_MODEL_URL", "").strip()

        self.ref_dir = Path(os.getenv("LOCAL_TTS_REF_DIR", "assets/tts_ref"))
        self.ref_map = {
            "neutral": self.ref_dir / "neutral.wav",
            "happy": self.ref_dir / "happy.wav",
            "sad": self.ref_dir / "sad.wav",
            "angry": self.ref_dir / "angry.wav",
        }

    def _resolve_model_dir(self) -> Path:
        env_dir = os.getenv("LOCAL_TTS_MODEL_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        return Path("models") / "tts" / "styletts2"

    def _ensure_model(self) -> Optional[Path]:
        if not self.model_url:
            return None
        self.model_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(self.model_url).name or "styletts2_model.pth"
        model_path = self.model_dir / filename
        if model_path.exists():
            return model_path
        with requests.get(self.model_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return model_path

    def _resolve_engine(self):
        import importlib

        styletts2 = importlib.import_module("styletts2")
        for attr in ("StyleTTS2", "TTS", "StyleTTS"):
            if hasattr(styletts2, attr):
                return getattr(styletts2, attr)

        if hasattr(styletts2, "tts") and hasattr(styletts2.tts, "StyleTTS2"):
            return styletts2.tts.StyleTTS2

        raise RuntimeError("styletts2 package does not expose a supported engine class")

    def load(self):
        Engine = self._resolve_engine()
        model_path = self._ensure_model()
        kwargs = {}
        init_vars = Engine.__init__.__code__.co_varnames
        if "device" in init_vars:
            kwargs["device"] = self.device
        if model_path:
            if "model_dir" in init_vars:
                kwargs["model_dir"] = str(self.model_dir)
            elif "checkpoint_path" in init_vars:
                kwargs["checkpoint_path"] = str(model_path)
        self.engine = Engine(**kwargs) if kwargs else Engine()

    def _emotion_params(self, emotion: str) -> Dict:
        preset = {
            "neutral": {"alpha": 0.3, "beta": 0.7, "embedding_scale": 1.0},
            "happy": {"alpha": 0.2, "beta": 0.9, "embedding_scale": 1.2},
            "sad": {"alpha": 0.6, "beta": 0.4, "embedding_scale": 0.85},
            "angry": {"alpha": 0.1, "beta": 0.6, "embedding_scale": 1.35},
        }
        return preset.get(emotion, preset["neutral"])

    def _reference_audio(self, emotion: str) -> Optional[str]:
        path = self.ref_map.get(emotion)
        if path and path.exists():
            return str(path)
        return None

    def _post_emotion(self, audio: np.ndarray, emotion: str) -> np.ndarray:
        if emotion not in ("happy", "sad", "angry"):
            return audio
        try:
            import librosa

            if emotion == "happy":
                return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=1.5)
            if emotion == "sad":
                return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=-1.5)
            if emotion == "angry":
                boosted = audio * 1.2
                return np.clip(boosted, -1.0, 1.0)
        except Exception:
            return audio
        return audio

    def synthesize(self, text: str, emotion: str = "neutral") -> Optional[bytes]:
        if self.engine is None:
            self.load()

        emotion = (emotion or "neutral").lower()
        params = self._emotion_params(emotion)
        ref_audio = self._reference_audio(emotion)

        try:
            audio = self.engine.inference(
                text,
                alpha=params.get("alpha", 0.3),
                beta=params.get("beta", 0.7),
                embedding_scale=params.get("embedding_scale", 1.0),
                reference_audio=ref_audio,
            )
        except TypeError:
            audio = self.engine.inference(text)
        except Exception:
            return None

        if audio is None:
            return None

        audio = np.asarray(audio, dtype=np.float32)
        audio = self._post_emotion(audio, emotion)

        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
