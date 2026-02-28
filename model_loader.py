import os
import time
from typing import List


class LLMLoader:
    """API LLM loader with OpenAI/NVIDIA-compatible support."""

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self):
        self.client = None
        self.provider = (os.getenv("LLM_PROVIDER", "auto") or "auto").lower()
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        resolved_provider = self._resolve_provider(self.provider, self.api_key)
        self.provider = resolved_provider

        default_model = (
            os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct")
            if resolved_provider == "nvidia"
            else "gpt-4o-mini"
        )
        self.model_name = os.getenv("LLM_MODEL", default_model)
        self.base_url = os.getenv("LLM_BASE_URL") or (
            self.NVIDIA_BASE_URL if resolved_provider == "nvidia" else None
        )
        self.request_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.enable_local_fallback = (
            os.getenv("LLM_ENABLE_LOCAL_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}
        )

    def _resolve_provider(self, provider: str, api_key: str) -> str:
        if provider in ("openai", "nvidia"):
            return provider
        if api_key and api_key.startswith("nvapi-"):
            return "nvidia"
        return "openai"

    def load(self):
        from openai import OpenAI

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
        print(
            f"[LLMLoader] API mode enabled, provider={self.provider}, model={self.model_name}, base_url={self.base_url or 'default'}"
        )

    def get_model_name(self) -> str:
        return f"{self.model_name} ({self.provider})"

    def list_models(self) -> List[str]:
        if self.client is None:
            self.load()
        models = self.client.models.list()
        return sorted(m.id for m in models.data)

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
