import os
import time


class LLMLoader:
    """Pure API LLM loader for serverless environments."""

    def __init__(self):
        self.client = None
        self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("LLM_BASE_URL") or None
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.request_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

    def load(self):
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set LLM_API_KEY or OPENAI_API_KEY in environment variables."
            )

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)
        print(f"[LLMLoader] API mode enabled, model={self.model_name}")

    def get_model_name(self) -> str:
        return f"{self.model_name} (API)"

    def generate(
        self, text: str, max_new_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        if self.client is None:
            self.load()
        return self._generate_api(text, max_new_tokens, temperature)

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
