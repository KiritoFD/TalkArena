import os
import time
from typing import Dict, List, Optional


class LocalFallbackLLM:
    """Final local fallback using a 0.5B model from ModelScope."""

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self):
        self.model_id = os.getenv("LLM_LOCAL_MODEL_ID", self.DEFAULT_MODEL_ID)
        self.cache_dir = os.getenv("LLM_LOCAL_CACHE_DIR", "models/modelscope")
        self.device = os.getenv("LLM_LOCAL_DEVICE", "auto")
        self.torch_dtype = os.getenv("LLM_LOCAL_DTYPE", "auto")
        self.max_new_tokens = int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "256"))
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        from modelscope import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        local_dir = snapshot_download(self.model_id, cache_dir=self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self._loaded = True
        print(
            f"[LocalFallbackLLM] Loaded ModelScope model={self.model_id}, cache_dir={self.cache_dir}"
        )

    def generate(self, text: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        self.load()
        limit = min(max_new_tokens, self.max_new_tokens)
        outputs = self.generator(
            text,
            max_new_tokens=limit,
            do_sample=temperature > 0,
            temperature=max(0.01, temperature),
            top_p=0.9,
            return_full_text=False,
        )
        generated = outputs[0].get("generated_text", "").strip()
        if not generated:
            raise RuntimeError("Local fallback model returned empty content")
        return generated


class LLMLoader:
    """API LLM loader with OpenAI/NVIDIA support and local final fallback."""

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    OPENAI_BASE_URL = "https://api.openai.com/v1"

    def __init__(self):
        self.client = None
        self.provider = (os.getenv("LLM_PROVIDER", "auto") or "auto").lower()
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.request_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.enable_local_fallback = (
            os.getenv("LLM_ENABLE_LOCAL_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}
        )

        self.endpoints = self._build_endpoints()
        self.active_endpoint = 0
        self.local_fallback: Optional[LocalFallbackLLM] = None

        if self.endpoints:
            current = self.endpoints[self.active_endpoint]
            self.provider = current["provider"]
            self.model_name = current["model"]
            self.base_url = current["base_url"]
        else:
            self.provider = "local"
            self.model_name = os.getenv("LLM_LOCAL_MODEL_ID", LocalFallbackLLM.DEFAULT_MODEL_ID)
            self.base_url = "local"

    def _split_env(self, name: str) -> List[str]:
        raw = os.getenv(name, "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _default_model_for_provider(self, provider: str) -> str:
        if provider == "nvidia":
            return os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct")
        return "gpt-4o-mini"

    def _default_base_url_for_provider(self, provider: str) -> str:
        if provider == "nvidia":
            return self.NVIDIA_BASE_URL
        return os.getenv("OPENAI_BASE_URL", self.OPENAI_BASE_URL)

    def _build_endpoints(self) -> List[Dict[str, str]]:
        providers = self._split_env("LLM_PROVIDERS")
        api_keys = self._split_env("LLM_API_KEYS")
        models = self._split_env("LLM_MODELS")
        base_urls = self._split_env("LLM_BASE_URLS")

        if not api_keys:
            backup_keys = self._split_env("LLM_BACKUP_API_KEYS")
            primary_key = self.api_key
            api_keys = [primary_key] if primary_key else []
            api_keys.extend(backup_keys)

        endpoints = []
        fallback_models = self._split_env("LLM_BACKUP_MODELS")
        fallback_base_urls = self._split_env("LLM_BACKUP_BASE_URLS")
        fallback_providers = self._split_env("LLM_BACKUP_PROVIDERS")

        for idx, key in enumerate(api_keys):
            provider_hint = (
                providers[idx]
                if idx < len(providers)
                else (
                    fallback_providers[idx - 1]
                    if idx > 0 and idx - 1 < len(fallback_providers)
                    else self.provider
                )
            )
            provider = self._resolve_provider(provider_hint, key)

            if idx < len(models):
                model = models[idx]
            elif idx == 0 and os.getenv("LLM_MODEL"):
                model = os.getenv("LLM_MODEL")
            elif idx > 0 and idx - 1 < len(fallback_models):
                model = fallback_models[idx - 1]
            else:
                model = self._default_model_for_provider(provider)

            if idx < len(base_urls):
                base_url = base_urls[idx]
            elif idx == 0 and os.getenv("LLM_BASE_URL"):
                base_url = os.getenv("LLM_BASE_URL")
            elif idx > 0 and idx - 1 < len(fallback_base_urls):
                base_url = fallback_base_urls[idx - 1]
            else:
                base_url = self._default_base_url_for_provider(provider)

            endpoints.append(
                {
                    "provider": provider,
                    "api_key": key,
                    "model": model,
                    "base_url": base_url,
                }
            )

        return endpoints

    def _resolve_provider(self, provider: str, api_key: str) -> str:
        if provider in ("openai", "nvidia"):
            return provider
        if api_key and api_key.startswith("nvapi-"):
            return "nvidia"
        return "openai"

    def load(self):
        if not self.endpoints:
            if not self.enable_local_fallback:
                raise RuntimeError(
                    "No API endpoints configured and local fallback disabled."
                )
            self._get_local_fallback().load()
            return

        from openai import OpenAI

        endpoint = self.endpoints[self.active_endpoint]
        kwargs = {"api_key": endpoint["api_key"]}
        if endpoint["base_url"]:
            kwargs["base_url"] = endpoint["base_url"]

        self.client = OpenAI(**kwargs)
        self.provider = endpoint["provider"]
        self.model_name = endpoint["model"]
        self.base_url = endpoint["base_url"]
        print(
            f"[LLMLoader] API mode enabled, endpoint={self.active_endpoint + 1}/{len(self.endpoints)}, provider={self.provider}, model={self.model_name}, base_url={self.base_url or 'default'}"
        )

    def _get_local_fallback(self) -> LocalFallbackLLM:
        if self.local_fallback is None:
            self.local_fallback = LocalFallbackLLM()
        return self.local_fallback

    def get_model_name(self) -> str:
        return f"{self.model_name} ({self.provider})"

    def list_models(self) -> List[str]:
        if not self.endpoints:
            if self.enable_local_fallback:
                local_model = os.getenv("LLM_LOCAL_MODEL_ID", LocalFallbackLLM.DEFAULT_MODEL_ID)
                return [local_model]
            raise RuntimeError("No API endpoints configured.")

        errors = []
        for endpoint_idx in range(len(self.endpoints)):
            try:
                self._switch_endpoint(endpoint_idx)
                models = self.client.models.list()
                return sorted(m.id for m in models.data)
            except Exception as e:
                errors.append(f"endpoint#{endpoint_idx + 1}: {type(e).__name__}: {e}")

        if self.enable_local_fallback:
            local_model = os.getenv("LLM_LOCAL_MODEL_ID", LocalFallbackLLM.DEFAULT_MODEL_ID)
            print("[LLMLoader] list_models API failed, fallback to local model metadata")
            return [local_model]

        raise RuntimeError("All API endpoints failed for list_models: " + " | ".join(errors))

    def generate(
        self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        errors = []
        endpoint_order = [self.active_endpoint] + [
            idx for idx in range(len(self.endpoints)) if idx != self.active_endpoint
        ]

        for endpoint_idx in endpoint_order:
            try:
                self._switch_endpoint(endpoint_idx)
                return self._generate_api(text, max_new_tokens, temperature)
            except Exception as e:
                errors.append(f"endpoint#{endpoint_idx + 1}: {type(e).__name__}: {e}")
                print(f"[LLMLoader] Switching to next endpoint due to error: {e}")

        if self.enable_local_fallback:
            try:
                local = self._get_local_fallback()
                self.provider = "local"
                self.model_name = local.model_id
                self.base_url = "local"
                print("[LLMLoader] Falling back to local ModelScope 0.5B model...")
                return local.generate(text, max_new_tokens=max_new_tokens, temperature=temperature)
            except Exception as e:
                errors.append(f"local_fallback: {type(e).__name__}: {e}")

        raise RuntimeError("All API endpoints failed: " + " | ".join(errors))

    def _switch_endpoint(self, endpoint_idx: int) -> None:
        if self.active_endpoint == endpoint_idx and self.client is not None:
            return
        self.active_endpoint = endpoint_idx
        self.client = None
        self.load()

    def _generate_api(self, text: str, max_new_tokens: int, temperature: float) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    timeout=self.request_timeout,
                )
                elapsed = time.time() - start
                finish_reason = response.choices[0].finish_reason
                print(
                    f"[LLMLoader] API response: {elapsed:.1f}s, finish_reason={finish_reason}"
                )

                content = response.choices[0].message.content
                if content is None or not content.strip():
                    raise RuntimeError("LLM API returned empty content")

                result = content.strip()
                print(f"[LLMLoader] API returned {len(result)} chars")
                return result
            except Exception as e:
                print(
                    f"[LLMLoader] API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                raise


class TTSLoader:
    """Optional TTS loader; kept for compatibility with local scripts."""

    def __init__(self):
        self.voice = "zh-CN-YunxiNeural"
        self.sample_rate = 24000

    def load(self):
        print("[TTSLoader] Initializing Edge-TTS...")
        import edge_tts

        self._edge_tts = edge_tts
        print("[TTSLoader] [OK] Ready")

    def synthesize(self, text: str, emotion: str = "neutral", voice: str = None) -> bytes:
        import io
        import os
        import subprocess
        import tempfile

        resolved_voice = voice or self._emotion_to_voice(emotion)

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "edge-tts",
                "--voice",
                resolved_voice,
                "--text",
                text,
                "--write-media",
                tmp_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0 or not os.path.exists(tmp_path):
                return None

            with open(tmp_path, "rb") as f:
                mp3_bytes = f.read()

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            if len(mp3_bytes) < 1024:
                return None

            try:
                from pydub import AudioSegment

                mp3_io = io.BytesIO(mp3_bytes)
                audio = AudioSegment.from_mp3(mp3_io)
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                return wav_io.getvalue()
            except Exception:
                return mp3_bytes

        except Exception:
            return None

    def _emotion_to_voice(self, emotion: str) -> str:
        emotion_voice_map = {
            "happy": "zh-CN-XiaoxiaoNeural",
            "sad": "zh-CN-YunyangNeural",
            "neutral": "zh-CN-YunxiNeural",
            "angry": "zh-CN-YunjianNeural",
        }
        return emotion_voice_map.get(emotion, "zh-CN-YunxiNeural")
