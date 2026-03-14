"""SiliconFlow integration configuration.

Keep all SiliconFlow model and voice selections centralized here for easier maintenance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class SiliconFlowConfig:
    api_key_default: str = "sk-zowfpdzeiqchwkdomuljrzfdumsejnogqsjvpnpguwxyazsq"
    base_url_default: str = "https://api.siliconflow.cn/v1"

    llm_model_default: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    stt_model_default: str = "FunAudioLLM/SenseVoiceSmall"

    # Prefer Chinese-focused TTS by default.
    tts_model_default: str = "fnlp/MOSS-TTSD-v0.5"
    tts_voice_default: str = "female"

    # Emotion-based fallback map.
    tts_emotion_voice_map: Dict[str, str] = field(
        default_factory=lambda: {
            "happy": "female",
            "sad": "female_elder",
            "neutral": "female",
            "angry": "male",
        }
    )

    # Role-aware map: key format "{gender}:{age_group}".
    # age_group in {child, youth, adult, elder}.
    tts_role_voice_map: Dict[str, str] = field(
        default_factory=lambda: {
            "female:child": "female_youth",
            "female:youth": "female_youth",
            "female:adult": "female",
            "female:elder": "female_elder",
            "male:child": "male_youth",
            "male:youth": "male_youth",
            "male:adult": "male",
            "male:elder": "male_elder",
        }
    )


SILICONFLOW_CONFIG = SiliconFlowConfig()


def siliconflow_api_key() -> str:
    return (
        os.getenv("SILICONFLOW_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or SILICONFLOW_CONFIG.api_key_default
    ).strip()


def siliconflow_base_url() -> str:
    return (os.getenv("SILICONFLOW_BASE_URL") or SILICONFLOW_CONFIG.base_url_default).strip().rstrip("/")


def normalize_gender(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"male", "m", "man", "boy", "男", "男性"}:
        return "male"
    if v in {"female", "f", "woman", "girl", "女", "女性"}:
        return "female"
    return ""


def infer_age_group(age) -> str:
    if age is None:
        return ""
    try:
        n = int(age)
    except (TypeError, ValueError):
        return ""
    if n <= 14:
        return "child"
    if n <= 28:
        return "youth"
    if n <= 59:
        return "adult"
    return "elder"


def infer_demographics_from_text(text: str) -> Dict[str, str]:
    raw = (text or "").strip()
    result = {"gender": "", "age_group": ""}
    if not raw:
        return result

    if any(k in raw for k in ["先生", "大爷", "大伯", "叔", "爷爷", "爸", "男"]):
        result["gender"] = "male"
    elif any(k in raw for k in ["女士", "阿姨", "奶奶", "妈", "姐", "女"]):
        result["gender"] = "female"

    if any(k in raw for k in ["小朋友", "儿童", "孩子"]):
        result["age_group"] = "child"
    elif any(k in raw for k in ["学生", "青年", "小伙", "姑娘"]):
        result["age_group"] = "youth"
    elif any(k in raw for k in ["大爷", "大妈", "伯", "叔", "阿姨", "爷爷", "奶奶", "老人"]):
        result["age_group"] = "elder"
    else:
        result["age_group"] = "adult"

    return result
