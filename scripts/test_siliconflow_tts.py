import argparse
import os
import time
from pathlib import Path

import requests


SILICONFLOW_BASE_URL_DEFAULT = "https://api.siliconflow.cn/v1"
SILICONFLOW_TTS_MODEL_DEFAULT = "fnlp/MOSS-TTSD-v0.5"
SILICONFLOW_TTS_VOICE_DEFAULT = "fnlp/MOSS-TTSD-v0.5:diana"
SILICONFLOW_API_KEY_DEFAULT = "sk-zowfpdzeiqchwkdomuljrzfdumsejnogqsjvpnpguwxyazsq"
DEFAULT_TEXT = "龙，是帝王之征啊"


def get_api_key() -> str:
    return (
        os.getenv("SILICONFLOW_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or SILICONFLOW_API_KEY_DEFAULT
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Test SiliconFlow TTS and save audio to local file.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize.")
    parser.add_argument("--model", default=os.getenv("SILICONFLOW_TTS_MODEL", SILICONFLOW_TTS_MODEL_DEFAULT))
    parser.add_argument("--voice", default=os.getenv("SILICONFLOW_TTS_VOICE", SILICONFLOW_TTS_VOICE_DEFAULT))
    parser.add_argument(
        "--response-format",
        default=os.getenv("SILICONFLOW_TTS_RESPONSE_FORMAT", "wav"),
        choices=["wav", "mp3", "pcm"],
        help="Output audio format.",
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("SILICONFLOW_TTS_TIMEOUT", "60")))
    parser.add_argument("--out", default="", help="Output file path. Auto-generated if omitted.")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("Missing API key. Set SILICONFLOW_API_KEY (or LLM_API_KEY / OPENAI_API_KEY).")
        return 1

    base_url = (os.getenv("SILICONFLOW_BASE_URL") or SILICONFLOW_BASE_URL_DEFAULT).strip().rstrip("/")
    url = f"{base_url}/audio/speech"
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Calling SiliconFlow TTS: model={args.model}, voice={args.voice}, format={args.response_format}")
    resp = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
    if resp.status_code >= 400:
        print(f"TTS request failed: HTTP {resp.status_code}")
        print(resp.text[:500])
        return 2

    suffix = args.response_format.lower()
    output = Path(args.out) if args.out else Path("outputs") / f"siliconflow_tts_{int(time.time() * 1000)}.{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(resp.content)

    print(f"Success. Audio saved to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
