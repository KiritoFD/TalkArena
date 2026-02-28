#!/usr/bin/env python3
"""Check NVIDIA API model availability and simple chat capability."""

import os
import sys
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
CANDIDATE_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
]


def main() -> int:
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Missing NVIDIA_API_KEY/LLM_API_KEY/OPENAI_API_KEY")
        return 2

    client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL", BASE_URL))

    try:
        models = client.models.list()
        ids = sorted(m.id for m in models.data)
        print(f"[OK] models.list success, count={len(ids)}")
        for mid in ids[:80]:
            print(mid)
    except Exception as e:
        print(f"[ERROR] models.list failed: {type(e).__name__}: {e}")
        return 1

    print("\n[INFO] probing candidate models...")
    for model in CANDIDATE_MODELS:
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK only."}],
                max_tokens=8,
                temperature=0,
                timeout=20,
            )
            text = (rsp.choices[0].message.content or "").strip().replace("\n", " ")
            print(f"[OK] {model}: {text[:80]}")
        except Exception as e:
            print(f"[FAIL] {model}: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
