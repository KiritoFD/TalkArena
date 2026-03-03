import os
import time
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("ModelLoader")


class _LocalModelFallback:
    model_id = "Qwen/Qwen3.5-4B"

    def __init__(self):
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._dtype = None
        self._model_dir = self._resolve_model_dir()
        self._device_pref = (os.getenv("LOCAL_LLM_DEVICE", "cuda") or "cuda").strip().lower()

    def _resolve_model_dir(self) -> Path:
        env_dir = os.getenv("LOCAL_LLM_MODEL_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        return Path("models") / "llm" / "Qwen" / "Qwen3.5-4B"

    def _resolve_model_ref(self) -> str:
        if self._model_dir.exists():
            return str(self._model_dir)

        # Prefer ModelScope for this repo's local model workflow.
        try:
            from modelscope import snapshot_download

            cache_dir = snapshot_download(self.model_id)
            if cache_dir:
                logger.info("[LocalModelFallback] downloaded from ModelScope: %s", cache_dir)
                return cache_dir
        except Exception as e:
            logger.warning("[LocalModelFallback] ModelScope download skipped: %s", e)

        # Fallback to direct model id (e.g. HuggingFace mirror/local cache).
        return self.model_id

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device_map = "cpu"
        dtype = torch.float32
        if self._device_pref in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("LOCAL_LLM_DEVICE=cuda but CUDA is not available")
            device_map = "cuda"
            dtype = torch.float16
            self._device = "cuda"
        elif self._device_pref == "cpu":
            self._device = "cpu"
        else:
            if torch.cuda.is_available():
                device_map = "cuda"
                dtype = torch.float16
                self._device = "cuda"
            else:
                self._device = "cpu"

        model_ref = self._resolve_model_ref()
        # Force eager download/load once so runtime fallback does not fail mid-request.
        self._tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if not self._model_dir.exists():
            self._model_dir.mkdir(parents=True, exist_ok=True)
            self._tokenizer.save_pretrained(self._model_dir)
            self._model.save_pretrained(self._model_dir)
        self._dtype = dtype
        logger.info(
            "[LocalModelFallback] loaded model=%s device=%s dtype=%s",
            self.model_id,
            self._device,
            str(self._dtype).replace("torch.", ""),
        )
        self._loaded = True

    def generate(
        self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        if not self._loaded:
            self.load()

        import torch

        prompt = text if isinstance(text, str) else str(text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(0.01, float(temperature)),
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        resp = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not resp:
            raise RuntimeError("Local fallback returned empty content")
        return resp


class LLMLoader:
    """API LLM loader with endpoint failover and optional local fallback."""

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_NVIDIA_API_KEY = "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"
    DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

    def __init__(self):
        self.client = None
        self.request_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.enable_local_fallback = (
            os.getenv("LLM_ENABLE_LOCAL_FALLBACK", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        self._local_fallback = None

        self.endpoints = self._build_endpoints()
        self.active_endpoint = 0

        if self.endpoints:
            ep = self.endpoints[0]
            self.provider = ep["provider"]
            self.model_name = ep["model"]
            self.base_url = ep["base_url"]
        else:
            self.provider = "openai"
            self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self.base_url = os.getenv("LLM_BASE_URL")

    def _resolve_provider(self, provider: str, api_key: str) -> str:
        if provider in ("openai", "nvidia"):
            return provider
        if api_key and api_key.startswith("nvapi-"):
            return "nvidia"
        return "openai"

    def _split_csv_env(self, key: str) -> List[str]:
        raw = (os.getenv(key, "") or "").strip()
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    def _build_endpoints(self) -> List[Dict[str, str]]:
        keys = self._split_csv_env("LLM_API_KEYS")
        models = self._split_csv_env("LLM_MODELS")
        providers = self._split_csv_env("LLM_PROVIDERS")
        base_urls = self._split_csv_env("LLM_BASE_URLS")

        if not keys:
            single_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if single_key:
                keys = [single_key]
            else:
                # Code-level default so the app can run without env vars.
                keys = [self.DEFAULT_NVIDIA_API_KEY]
                if not providers:
                    providers = ["nvidia"]
                if not models:
                    models = [self.DEFAULT_NVIDIA_MODEL]
                if not base_urls:
                    base_urls = [self.NVIDIA_BASE_URL]

        endpoints: List[Dict[str, str]] = []
        for i, key in enumerate(keys):
            provider = (
                providers[i]
                if i < len(providers)
                else (os.getenv("LLM_PROVIDER", "auto") or "auto").lower()
            )
            provider = self._resolve_provider(provider, key)

            default_model = (
                os.getenv("NVIDIA_LLM_MODEL", self.DEFAULT_NVIDIA_MODEL)
                if provider == "nvidia"
                else "gpt-4o-mini"
            )
            model = (
                models[i]
                if i < len(models)
                else os.getenv("LLM_MODEL", default_model)
            )
            base_url = (
                base_urls[i]
                if i < len(base_urls)
                else os.getenv("LLM_BASE_URL")
            )
            if not base_url and provider == "nvidia":
                base_url = self.NVIDIA_BASE_URL

            endpoints.append(
                {
                    "provider": provider,
                    "api_key": key,
                    "model": model,
                    "base_url": base_url,
                }
            )
        return endpoints

    def _get_local_fallback(self):
        if self._local_fallback is None:
            self._local_fallback = _LocalModelFallback()
        return self._local_fallback

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

    def get_model_name(self) -> str:
        return f"{self.model_name} ({self.provider})"

    def _endpoint_order(self) -> List[int]:
        if not self.endpoints:
            return []
        return [self.active_endpoint] + [
            idx for idx in range(len(self.endpoints)) if idx != self.active_endpoint
        ]

    def list_models(self) -> List[str]:
        errors = []
        for endpoint_idx in self._endpoint_order():
            try:
                self._switch_endpoint(endpoint_idx)
                models = self.client.models.list()
                return sorted(m.id for m in models.data)
            except Exception as e:
                errors.append(f"endpoint#{endpoint_idx + 1}: {type(e).__name__}: {e}")
        raise RuntimeError("All API endpoints failed: " + " | ".join(errors))

    def generate(
        self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        errors = []
        for endpoint_idx in self._endpoint_order():
            try:
                self._switch_endpoint(endpoint_idx)
                return self._generate_api(text, max_new_tokens, temperature)
            except Exception as e:
                errors.append(f"endpoint#{endpoint_idx + 1}: {type(e).__name__}: {e}")

        if self.enable_local_fallback:
            try:
                local = self._get_local_fallback()
                self.provider = "local"
                self.model_name = local.model_id
                self.base_url = "local"
                return local.generate(
                    text, max_new_tokens=max_new_tokens, temperature=temperature
                )
            except Exception as e:
                errors.append(f"local_fallback: {type(e).__name__}: {e}")

        raise RuntimeError("All API endpoints failed: " + " | ".join(errors))

    def _switch_endpoint(self, endpoint_idx: int) -> None:
        if self.active_endpoint == endpoint_idx and self.client is not None:
            return
        self.active_endpoint = endpoint_idx
        self.client = None
        self.load()

    def _generate_api(
        self, text: str, max_new_tokens: int, temperature: float
    ) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                _ = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    timeout=self.request_timeout,
                )
                content = response.choices[0].message.content
                if content is None or not content.strip():
                    raise RuntimeError("LLM API returned empty content")
                return content.strip()
            except Exception:
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
