from pathlib import Path
from config.models import MODELS_CONFIG


class LLMLoader:
    # NVIDIA API 配置 - 优先尝试
    NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = (
        "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"
    )
    NVIDIA_MODELS_TO_TRY = [
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "meta/llama-3.3-70b-instruct",
        "qwen/qwen2.5-7b-instruct",
    ]

    # 魔搭 API-Inference 配置 - 备选
    MODELSCOPE_API_BASE_URL = "https://api-inference.modelscope.cn/v1/"
    MODELSCOPE_API_KEY = "ms-0cf515d7-87f9-4a63-b9bb-2c35c574674a"
    MODELSCOPE_MODELS_TO_TRY = [
        "ZhipuAI/GLM-4.7-Flash",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    def __init__(self):
        self.client = None
        self.model_name = None
        self.use_api = False
        self.api_provider = None  # 'nvidia' 或 'modelscope'
        # 本地模型 fallback
        self.local_model = None
        self.local_tokenizer = None

    def load(self):
        """加载 LLM，优先使用 NVIDIA API，其次魔搭 API，最后本地模型"""
        from openai import OpenAI

        # 1. 首先尝试 NVIDIA API
        print("[LLMLoader] 尝试连接 NVIDIA API...")

        self.client = OpenAI(
            base_url=self.NVIDIA_API_BASE_URL, api_key=self.NVIDIA_API_KEY
        )

        # 依次尝试 NVIDIA 模型
        for model in self.NVIDIA_MODELS_TO_TRY:
            print(f"[LLMLoader] 测试 NVIDIA 模型: {model}")
            if self._test_model(model):
                self.model_name = model
                self.use_api = True
                self.api_provider = "nvidia"
                print(f"[LLMLoader] [OK] 使用 NVIDIA API: {model}")
                return

        # 2. NVIDIA 失败，尝试魔搭 API
        print("[LLMLoader] NVIDIA API 不可用，尝试魔搭 API...")

        self.client = OpenAI(
            base_url=self.MODELSCOPE_API_BASE_URL, api_key=self.MODELSCOPE_API_KEY
        )

        # 依次尝试魔搭模型
        for model in self.MODELSCOPE_MODELS_TO_TRY:
            print(f"[LLMLoader] 测试魔搭模型: {model}")
            if self._test_model(model):
                self.model_name = model
                self.use_api = True
                self.api_provider = "modelscope"
                print(f"[LLMLoader] [OK] 使用魔搭 API: {model}")
                return

        # 3. 全部 API 失败，回退到本地模型
        print("[LLMLoader] 所有 API 均不可用，回退到本地模型...")
        self._load_local_model()

    def _test_model(self, model: str) -> bool:
        """测试模型是否可用"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10,
                timeout=10,
            )
            content = response.choices[0].message.content
            return content is not None and content.strip() != ""
        except Exception as e:
            print(f"[LLMLoader] {model} 不可用: {e}")
            return False

    def _load_local_model(self):
        """加载本地模型"""
        print("[LLMLoader] 加载本地 Qwen2.5-3B 模型...")
        from modelscope import AutoModelForCausalLM, AutoTokenizer

        config = MODELS_CONFIG["llm"]

        self.local_tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], cache_dir=config["cache_dir"], trust_remote_code=True
        )

        self.local_model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            cache_dir=config["cache_dir"],
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.use_api = False
        print("[LLMLoader] [OK] 本地模型加载成功")

    def get_model_name(self) -> str:
        """获取当前使用的模型名称"""
        if self.use_api:
            provider = self.api_provider.upper() if self.api_provider else "API"
            model_short = (
                self.model_name.split("/")[-1] if self.model_name else "Unknown"
            )
            return f"{model_short} ({provider})"
        else:
            return "Qwen2.5-3B (local)"

    def generate(
        self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        """生成回复"""
        if self.use_api:
            try:
                return self._generate_api(text, max_new_tokens, temperature)
            except Exception as e:
                print(f"[LLMLoader] API调用失败: {e}")
                print("[LLMLoader] 自动切换到本地模型...")
                if not self.local_model:
                    self._load_local_model()
                self.use_api = False
                return self._generate_local(text, max_new_tokens, temperature)
        else:
            return self._generate_local(text, max_new_tokens, temperature)

    def _generate_api(self, text: str, max_new_tokens: int, temperature: float) -> str:
        """使用魔搭 API 生成，带重试和模型轮换"""
        import time

        # 如果当前模型多次失败，可能需要重新选择模型
        # 这里简化处理：如果在3次重试中都失败，则在下一次调用时切换模型

        for attempt in range(3):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                )
                elapsed = time.time() - start

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                print(
                    f"[LLMLoader] API响应: {elapsed:.1f}s, finish_reason={finish_reason}"
                )

                if content is None or content.strip() == "":
                    print(
                        f"[LLMLoader] 警告: API返回空内容 (attempt {attempt + 1}/3). Message: {response.choices[0].message}"
                    )
                    if attempt < 2:
                        # 尝试增加一点延时
                        time.sleep(2)
                        continue
                    # 如果三次都空，尝试换个模型（如果还有的话）
                    print(
                        f"[LLMLoader] 模型 {self.model_name} 持续返回空内容，尝试下架它..."
                    )
                    # 从当前 provider 的模型列表中移除并切换
                    current_models = (
                        self.NVIDIA_MODELS_TO_TRY
                        if self.api_provider == "nvidia"
                        else self.MODELSCOPE_MODELS_TO_TRY
                        if self.api_provider == "modelscope"
                        else []
                    )
                    if self.model_name in current_models:
                        current_models.remove(self.model_name)
                    if current_models:
                        self.model_name = current_models[0]
                        print(f"[LLMLoader] 切换到新模型: {self.model_name}")
                        # 递归尝试一次新模型
                        return self._generate_api(text, max_new_tokens, temperature)
                    return ""

                result = content.strip()
                print(f"[LLMLoader] API返回: {len(result)}字符")
                return result

            except Exception as e:
                print(f"[LLMLoader] API调用异常 (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
                    continue
                raise

        return ""

    def _generate_local(
        self, text: str, max_new_tokens: int, temperature: float
    ) -> str:
        """使用本地模型生成"""
        messages = [{"role": "user", "content": text}]
        inputs = self.local_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        attention_mask = (inputs != self.local_tokenizer.pad_token_id).long()

        outputs = self.local_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.local_tokenizer.eos_token_id,
        )

        response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (
            response.split("assistant\n")[-1].strip()
            if "assistant" in response
            else response
        )


class TTSLoader:
    def __init__(self):
        self.voice = "zh-CN-YunxiNeural"
        self.sample_rate = 24000

    def load(self):
        print("[TTSLoader] 正在初始化 Edge-TTS...")
        import edge_tts

        self._edge_tts = edge_tts
        print("[TTSLoader] [OK] Edge-TTS 就绪")

    def synthesize(
        self, text: str, emotion: str = "neutral", voice: str = None
    ) -> bytes:
        import io
        import subprocess
        import tempfile
        import os
        from pydub import AudioSegment

        resolved_voice = voice or self._emotion_to_voice(emotion)
        print(f"[TTSLoader] 合成语音: {text[:50]}...")
        print(f"[TTSLoader] 文本长度: {len(text)} 字符")

        # 使用 subprocess 调用 edge-tts CLI
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            print(f"[TTSLoader] 临时文件: {tmp_path}")

            cmd = [
                "edge-tts",
                "--voice",
                resolved_voice,
                "--text",
                text,
                "--write-media",
                tmp_path,
            ]
            print(
                f"[TTSLoader] 执行命令: edge-tts --voice {resolved_voice} --text <{len(text)}字符>"
            )

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            print(f"[TTSLoader] 返回码: {result.returncode}")
            if result.stdout:
                print(f"[TTSLoader] stdout: {result.stdout[:200]}")
            if result.stderr:
                print(f"[TTSLoader] stderr: {result.stderr[:200]}")

            if result.returncode != 0:
                print(f"[TTSLoader] CLI错误: {result.stderr}")
                return None

            if not os.path.exists(tmp_path):
                print("[TTSLoader] 临时文件不存在")
                return None

            file_size = os.path.getsize(tmp_path)
            print(f"[TTSLoader] 文件大小: {file_size} bytes")

            with open(tmp_path, "rb") as f:
                mp3_bytes = f.read()

            try:
                os.unlink(tmp_path)
            except:
                pass

            print(f"[TTSLoader] 读取完成: {len(mp3_bytes)} bytes")

            if len(mp3_bytes) < 1024:
                print(f"[TTSLoader] 警告: MP3数据无效")
                return None

            # 转换为 WAV
            mp3_io = io.BytesIO(mp3_bytes)
            try:
                audio = AudioSegment.from_mp3(mp3_io)
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_bytes = wav_io.getvalue()
                print(f"[TTSLoader] [OK] WAV: {len(wav_bytes)} bytes")
                return wav_bytes
            except Exception as e:
                print(f"[TTSLoader] MP3转WAV失败: {e}")
                return mp3_bytes

        except subprocess.TimeoutExpired:
            print("[TTSLoader] 超时")
            return None
        except Exception as e:
            print(f"[TTSLoader] 异常: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _emotion_to_voice(self, emotion: str) -> str:
        emotion_voice_map = {
            "happy": "zh-CN-XiaoxiaoNeural",
            "sad": "zh-CN-YunyangNeural",
            "neutral": "zh-CN-YunxiNeural",
            "angry": "zh-CN-YunjianNeural",
        }
        return emotion_voice_map.get(emotion, "zh-CN-YunxiNeural")
