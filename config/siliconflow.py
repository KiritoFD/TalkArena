"""SiliconFlow integration configuration.

Keep all SiliconFlow model and voice selections centralized here for easier maintenance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class SiliconFlowConfig:
    api_key_default: str = "sk-zowfpdzeiqchwkdomuljrzfdumsejnogqsjvpnpguwxyazsq"
    base_url_default: str = "https://api.siliconflow.cn/v1"

    llm_model_default: str = "deepseek-ai/DeepSeek-V3.2"
    stt_model_default: str = "TeleAI/TeleSpeechASR"

    # Prefer Chinese-focused TTS by default.
    tts_model_default: str = "fnlp/MOSS-TTSD-v0.5"
    tts_voice_default: str = "fnlp/MOSS-TTSD-v0.5:diana"
    tts_official_voices: Dict[str, list] = field(default_factory=dict)

    # Emotion-based fallback map.
    tts_emotion_voice_map: Dict[str, str] = field(
        default_factory=lambda: {
            "happy": "fnlp/MOSS-TTSD-v0.5:anna",
            "sad": "fnlp/MOSS-TTSD-v0.5:claire",
            "neutral": "fnlp/MOSS-TTSD-v0.5:diana",
            "angry": "fnlp/MOSS-TTSD-v0.5:alex",
        }
    )

    # Role-aware map: key format "{gender}:{age_group}".
    # age_group in {child, youth, adult, elder}.
    tts_role_voice_map: Dict[str, str] = field(
        default_factory=lambda: {
            "female:child": "fnlp/MOSS-TTSD-v0.5:anna",
            "female:youth": "fnlp/MOSS-TTSD-v0.5:bella",
            "female:adult": "fnlp/MOSS-TTSD-v0.5:diana",
            "female:elder": "fnlp/MOSS-TTSD-v0.5:claire",
            "male:child": "fnlp/MOSS-TTSD-v0.5:benjamin",
            "male:youth": "fnlp/MOSS-TTSD-v0.5:david",
            "male:adult": "fnlp/MOSS-TTSD-v0.5:alex",
            "male:elder": "fnlp/MOSS-TTSD-v0.5:charles",
        }
    )
    preset_role_voice_map: Dict[str, str] = field(
        default_factory=lambda: {
            "主持人": "fnlp/MOSS-TTSD-v0.5:diana",
            "引导者": "fnlp/MOSS-TTSD-v0.5:diana",
            "长辈": "fnlp/MOSS-TTSD-v0.5:charles",
            "晚辈": "fnlp/MOSS-TTSD-v0.5:bella",
            "同事": "fnlp/MOSS-TTSD-v0.5:alex",
            "面试官": "fnlp/MOSS-TTSD-v0.5:charles",
            "hr": "fnlp/MOSS-TTSD-v0.5:claire",
            "竞争者": "fnlp/MOSS-TTSD-v0.5:david",
            "正方辩手": "fnlp/MOSS-TTSD-v0.5:alex",
            "反方辩手": "fnlp/MOSS-TTSD-v0.5:charles",
            "甲方负责人": "fnlp/MOSS-TTSD-v0.5:charles",
            "乙方商务": "fnlp/MOSS-TTSD-v0.5:diana",
            "风险顾问": "fnlp/MOSS-TTSD-v0.5:benjamin",
        }
    )


def _load_tts_json_overrides() -> Dict:
    path = Path(__file__).with_name("siliconflow_tts_voices.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


_BASE_CONFIG = SiliconFlowConfig()
_TTS_JSON = _load_tts_json_overrides()

if _TTS_JSON:
    defaults = _TTS_JSON.get("tts_defaults") or {}
    SILICONFLOW_CONFIG = replace(
        _BASE_CONFIG,
        tts_model_default=str(defaults.get("model") or _BASE_CONFIG.tts_model_default),
        tts_voice_default=str(defaults.get("voice") or _BASE_CONFIG.tts_voice_default),
        tts_official_voices=dict(_TTS_JSON.get("official_voices") or {}),
        tts_emotion_voice_map=dict(_TTS_JSON.get("emotion_voice_map") or _BASE_CONFIG.tts_emotion_voice_map),
        tts_role_voice_map=dict(_TTS_JSON.get("demographic_voice_map") or _BASE_CONFIG.tts_role_voice_map),
        preset_role_voice_map=dict(_TTS_JSON.get("preset_role_voice_map") or _BASE_CONFIG.preset_role_voice_map),
    )
else:
    SILICONFLOW_CONFIG = _BASE_CONFIG


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
