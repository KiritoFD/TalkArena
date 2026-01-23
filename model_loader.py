from pathlib import Path
from config.models import MODELS_CONFIG

class LLMLoader:
    # 魔搭 API-Inference 配置
    API_BASE_URL = "https://api-inference.modelscope.cn/v1/"
    API_KEY = "ms-0cf515d7-87f9-4a63-b9bb-2c35c574674a"
    MODELS_TO_TRY = [
        "ZhipuAI/GLM-4.7-Flash",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    
    def __init__(self):
        self.client = None
        self.model_name = None
        self.use_api = False
        # 本地模型 fallback
        self.local_model = None
        self.local_tokenizer = None
        
    def load(self):
        """加载 LLM，优先使用魔搭 API"""
        # 尝试魔搭 API
        print("[LLMLoader] 尝试连接魔搭 API-Inference...")
        
        from openai import OpenAI
        
        self.client = OpenAI(
            base_url=self.API_BASE_URL,
            api_key=self.API_KEY
        )
        
        # 依次尝试模型
        for model in self.MODELS_TO_TRY:
            print(f"[LLMLoader] 测试模型: {model}")
            if self._test_model(model):
                self.model_name = model
                self.use_api = True
                print(f"[LLMLoader] ✓ 使用魔搭 API: {model}")
                return
        
        # API 全部失败，回退到本地模型
        print("[LLMLoader] API 模型均不可用，回退到本地模型...")
        self._load_local_model()
    
    def _test_model(self, model: str) -> bool:
        """测试模型是否可用"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10,
                timeout=10
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            print(f"[LLMLoader] {model} 不可用: {e}")
            return False
    
    def _load_local_model(self):
        """加载本地模型"""
        print("[LLMLoader] 加载本地 Qwen2.5-3B 模型...")
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        
        config = MODELS_CONFIG["llm"]
        
        self.local_tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"],
            cache_dir=config["cache_dir"],
            trust_remote_code=True
        )
        
        self.local_model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            cache_dir=config["cache_dir"],
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.use_api = False
        print("[LLMLoader] ✓ 本地模型加载成功")
    
    def generate(self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7) -> str:
        """生成回复"""
        if self.use_api:
            return self._generate_api(text, max_new_tokens, temperature)
        else:
            return self._generate_local(text, max_new_tokens, temperature)
    
    def _generate_api(self, text: str, max_new_tokens: int, temperature: float) -> str:
        """使用魔搭 API 生成"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": text}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    
    def _generate_local(self, text: str, max_new_tokens: int, temperature: float) -> str:
        """使用本地模型生成"""
        messages = [{"role": "user", "content": text}]
        inputs = self.local_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        attention_mask = (inputs != self.local_tokenizer.pad_token_id).long()
        
        outputs = self.local_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.local_tokenizer.eos_token_id
        )
        
        response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        from pydub import AudioSegment
        
        resolved_voice = voice or self._emotion_to_voice(emotion)
        print(f"[TTSLoader] 合成语音: {text[:50]}...")
        
        async def _synthesize():
            communicate = self._edge_tts.Communicate(text, resolved_voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        mp3_bytes = asyncio.run(_synthesize())
        
        # MP3 转 WAV
        mp3_io = io.BytesIO(mp3_bytes)
        audio = AudioSegment.from_mp3(mp3_io)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()
        
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
