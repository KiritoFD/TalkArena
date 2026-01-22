from pathlib import Path
from config.models import MODELS_CONFIG

class LLMLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load(self):
        print("[LLMLoader] 正在下载 Qwen2.5-3B 模型...")
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        
        config = MODELS_CONFIG["llm"]
        
        print(f"[LLMLoader] 下载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"],
            cache_dir=config["cache_dir"],
            trust_remote_code=True
        )
        
        print(f"[LLMLoader] 下载模型文件 (这可能需要几分钟)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            cache_dir=config["cache_dir"],
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("[LLMLoader] ✓ Qwen2.5-3B 模型加载成功")
    
    def generate(self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant\n")[-1].strip() if "assistant" in response else response

class TTSLoader:
    def __init__(self):
        self.voice = "zh-CN-YunxiNeural"
        self.sample_rate = 24000
    
    def load(self):
        print("[TTSLoader] 正在初始化 Edge-TTS...")
        import edge_tts
        self._edge_tts = edge_tts
        print("[TTSLoader] ✓ Edge-TTS 就绪")
    
    def synthesize(self, text: str, emotion: str = "neutral", voice: str = None) -> bytes:
        import asyncio
        import io
        
        resolved_voice = voice or self._emotion_to_voice(emotion)
        print(f"[TTSLoader] 合成语音: {text[:50]}...")
        
        async def _synthesize():
            communicate = self._edge_tts.Communicate(text, resolved_voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        wav_bytes = asyncio.run(_synthesize())
        print(f"[TTSLoader] ✓ 合成成功，音频大小: {len(wav_bytes)} bytes")
        return wav_bytes

    def _emotion_to_voice(self, emotion: str) -> str:
        emotion_voice_map = {
            "happy": "zh-CN-XiaoxiaoNeural",
            "sad": "zh-CN-YunyangNeural",
            "neutral": "zh-CN-YunxiNeural",
            "angry": "zh-CN-YunjianNeural"
        }
        return emotion_voice_map.get(emotion, "zh-CN-YunxiNeural")
