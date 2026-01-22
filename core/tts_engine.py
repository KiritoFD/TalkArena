import os
import gc
import uuid
import torch
from pathlib import Path
from typing import Optional

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False


class TTSEngine:
    DEFAULT_MODEL = "iic/CosyVoice-300M-SFT"
    
    SPEAKER_MAPPING = {
        "male_serious": "中文男",
        "female_young": "中文女", 
        "male_casual": "中文男",
        "female_professional": "中文女",
        "default": "中文女",
    }
    
    def __init__(self, model_path: Optional[str] = None, output_dir: str = "outputs"):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        if not COSYVOICE_AVAILABLE:
            raise ImportError(
                "CosyVoice not installed. Please install from source:\n"
                "git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "cd CosyVoice && pip install -e ."
            )
        
        self.model = CosyVoice(self.model_path)
        self._loaded = True
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def synthesize(
        self,
        text: str,
        speaker_id: str = "default",
        output_filename: Optional[str] = None
    ) -> str:
        if not self._loaded:
            self.load()
        
        if output_filename is None:
            output_filename = f"{uuid.uuid4()}.wav"
        
        output_path = self.output_dir / output_filename
        
        speaker = self.SPEAKER_MAPPING.get(speaker_id, self.SPEAKER_MAPPING["default"])
        
        output_audio = self.model.inference_sft(text, speaker)
        
        import torchaudio
        torchaudio.save(
            str(output_path),
            output_audio["tts_speech"],
            22050
        )
        
        return str(output_path)
    
    def get_available_speakers(self) -> list:
        return list(self.SPEAKER_MAPPING.keys())


class DummyTTSEngine:
    """Fallback TTS engine when CosyVoice is not available."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True
    
    def load(self):
        pass
    
    def unload(self):
        pass
    
    def synthesize(
        self,
        text: str,
        speaker_id: str = "default",
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        return None
    
    def get_available_speakers(self) -> list:
        return ["default"]


def create_tts_engine(model_path: Optional[str] = None, output_dir: str = "outputs"):
    """Factory function to create appropriate TTS engine."""
    if COSYVOICE_AVAILABLE:
        return TTSEngine(model_path, output_dir)
    else:
        print("Warning: CosyVoice not available, using dummy TTS engine (no audio output)")
        return DummyTTSEngine(output_dir)
