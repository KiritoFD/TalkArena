"""
TalkArena FastAPI 服务端
整合 Multi-Agent、RAG、决策引擎、防幻觉机制
"""

import sys
import os
import logging
import importlib.util
import tempfile
import time
import hashlib
import json
import threading
import copy
import builtins
import pickle
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response, FileResponse
import base64
import requests
from pydantic import BaseModel
from typing import List, Optional, Dict

from config.siliconflow import (
    SILICONFLOW_CONFIG,
    infer_age_group,
    infer_demographics_from_text,
    normalize_gender,
    siliconflow_api_key,
    siliconflow_base_url,
)

app = FastAPI(title="TalkArena")
MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None
logger = logging.getLogger("TalkArenaAPI")

# 所有 print 日志统一加时间戳（毫秒）
_ORIGINAL_PRINT = builtins.print
def _ts_print(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    _ORIGINAL_PRINT(f"[{ts}]", *args, **kwargs)
builtins.print = _ts_print

# 引入认证路由
try:
    from core.auth.auth_routes import router as auth_router
    app.include_router(auth_router)
    print("[Auth] routes loaded")
except Exception as e:
    print(f"[Auth] routes load skipped: {e}")

engine = None
mm_analyzer = None
stt_service = None
tts_service = None
AUDIO_OUTPUT_DIR = Path(tempfile.gettempdir()) / "talkarena_audio"
TTS_FIXED_CACHE_MAX_ITEMS = int(os.getenv("TTS_FIXED_CACHE_MAX_ITEMS", "256"))
TTS_FIXED_AUDIO_CACHE: "OrderedDict[str, str]" = OrderedDict()
SCENE_PRESET_PROMPTS_PATH = Path(__file__).parent / "config" / "scene_preset_prompts.json"
SHARE_REPORT_DIR = Path(tempfile.gettempdir()) / "talkarena_share_reports"
SHARE_REPORT_MAX_ITEMS = int(os.getenv("SHARE_REPORT_MAX_ITEMS", "200"))
SHARE_REPORT_INDEX: "OrderedDict[str, str]" = OrderedDict()
IMGBB_API_KEY_DEFAULT = os.getenv("IMGBB_API_KEY_DEFAULT", "a99c8f498d65ff27bcfe2404998a5fc3")
SESSION_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "talkarena_session_snapshots"


def _session_snapshot_path(session_id: str) -> Path:
    safe = "".join(ch for ch in str(session_id or "") if ch.isalnum() or ch in ("-", "_"))
    return SESSION_SNAPSHOT_DIR / f"{safe}.pkl"


def _save_session_snapshot(eng, session_id: str) -> None:
    try:
        if not session_id or not hasattr(eng, "sessions"):
            return
        session = eng.sessions.get(session_id)
        if session is None:
            return
        SESSION_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = _session_snapshot_path(session_id)
        with open(path, "wb") as f:
            pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SessionSnapshot] saved sid={session_id} path={path.name}")
    except Exception as e:
        print(f"[SessionSnapshot] save failed sid={session_id}: {e}")


def _try_restore_session_snapshot(eng, session_id: str) -> bool:
    try:
        if not session_id or not hasattr(eng, "sessions"):
            return False
        if session_id in eng.sessions:
            return True
        path = _session_snapshot_path(session_id)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            session = pickle.load(f)
        if not isinstance(session, dict):
            return False
        eng.sessions[session_id] = session
        print(f"[SessionSnapshot] restored sid={session_id} path={path.name}")
        return True
    except Exception as e:
        print(f"[SessionSnapshot] restore failed sid={session_id}: {e}")
        return False


def _delete_session_snapshot(session_id: str) -> None:
    try:
        path = _session_snapshot_path(session_id)
        if path.exists():
            path.unlink()
            print(f"[SessionSnapshot] deleted sid={session_id} path={path.name}")
    except Exception as e:
        print(f"[SessionSnapshot] delete failed sid={session_id}: {e}")


def _public_base_url() -> str:
    # Must be a public reachable URL for QR sharing (not localhost).
    return str(
        os.getenv("TALKARENA_PUBLIC_BASE_URL")
        or os.getenv("PUBLIC_BASE_URL")
        or ""
    ).strip().rstrip("/")


def _decode_png_data_url(image_data: str) -> bytes:
    raw = str(image_data or "").strip()
    prefix = "data:image/png;base64,"
    if not raw.startswith(prefix):
        raise HTTPException(status_code=400, detail="image_data must be a PNG data URL")
    b64 = raw[len(prefix):].strip()
    try:
        image_bytes = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {e}") from e
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image payload")
    if len(image_bytes) > 3 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="image payload too large (>3MB)")
    return image_bytes


def _upload_report_image_public(image_bytes: bytes) -> str:
    provider_raw = (os.getenv("TALKARENA_IMAGE_HOST_PROVIDER") or "").strip().lower()
    provider = provider_raw or ("imgbb" if (os.getenv("IMGBB_API_KEY") or IMGBB_API_KEY_DEFAULT) else "catbox")
    timeout = float(os.getenv("TALKARENA_IMAGE_HOST_TIMEOUT", "30"))

    if provider == "imgbb":
        api_key = (os.getenv("IMGBB_API_KEY") or IMGBB_API_KEY_DEFAULT or "").strip()
        if not api_key:
            raise RuntimeError("IMGBB_API_KEY is required when TALKARENA_IMAGE_HOST_PROVIDER=imgbb")
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
            "image": base64.b64encode(image_bytes).decode("ascii"),
            "name": f"talkarena_{int(time.time())}",
        }
        resp = requests.post(url, data=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"imgbb upload failed: HTTP {resp.status_code} {resp.text[:220]}")
        data = resp.json() or {}
        up = ((data.get("data") or {}).get("url") or "").strip()
        if not up.startswith("http"):
            raise RuntimeError("imgbb upload response missing public URL")
        return up

    # default: anonymous upload to catbox
    url = "https://catbox.moe/user/api.php"
    files = {"fileToUpload": ("talkarena_report.png", image_bytes, "image/png")}
    data = {"reqtype": "fileupload"}
    resp = requests.post(url, files=files, data=data, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"catbox upload failed: HTTP {resp.status_code} {resp.text[:220]}")
    up = (resp.text or "").strip()
    if not up.startswith("http"):
        raise RuntimeError(f"catbox upload returned unexpected body: {up[:220]}")
    return up

def _truthy_env(name: str, default: str = "1") -> bool:
    return (os.getenv(name, default) or default).strip().lower() not in {"0", "false", "no", "off"}


def _siliconflow_api_key() -> str:
    return siliconflow_api_key()


def _siliconflow_base_url() -> str:
    return siliconflow_base_url()


def _ensure_llm_env_defaults() -> None:
    # Configure LLM defaults for SiliconFlow only when the user did not set explicit values.
    if not os.getenv("LLM_API_KEYS") and not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["LLM_API_KEY"] = _siliconflow_api_key()
    if not os.getenv("LLM_MODEL") and not os.getenv("LLM_MODELS"):
        os.environ["LLM_MODEL"] = os.getenv("SILICONFLOW_LLM_MODEL", SILICONFLOW_CONFIG.llm_model_default)
    if not os.getenv("LLM_BASE_URL") and not os.getenv("LLM_BASE_URLS"):
        os.environ["LLM_BASE_URL"] = _siliconflow_base_url()


class SiliconFlowSTTService:
    remote = True

    def __init__(self):
        self.api_key = _siliconflow_api_key()
        self.base_url = _siliconflow_base_url()
        self.model = os.getenv("SILICONFLOW_STT_MODEL", SILICONFLOW_CONFIG.stt_model_default).strip()
        self.timeout = float(os.getenv("SILICONFLOW_STT_TIMEOUT", "45"))
        self.last_latency_ms = 0

    def transcribe(self, audio_bytes: bytes, filename: str = "speech.wav") -> Dict:
        if not audio_bytes:
            return {"text": "", "voice_features": {}}
        t0 = time.perf_counter()
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": self.model}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=self.timeout)
        self.last_latency_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code >= 400:
            raise RuntimeError(f"SiliconFlow STT failed: HTTP {resp.status_code} {resp.text[:240]}")
        payload = resp.json()
        return {"text": (payload.get("text") or "").strip(), "voice_features": {}}


class SiliconFlowTTSService:
    remote = True

    _EMOTION_TO_VOICE = SILICONFLOW_CONFIG.tts_emotion_voice_map

    def __init__(self):
        self.api_key = _siliconflow_api_key()
        self.base_url = _siliconflow_base_url()
        self.model = os.getenv("SILICONFLOW_TTS_MODEL", SILICONFLOW_CONFIG.tts_model_default).strip()
        self.default_voice = os.getenv("SILICONFLOW_TTS_VOICE", SILICONFLOW_CONFIG.tts_voice_default).strip()
        self.timeout = float(os.getenv("SILICONFLOW_TTS_TIMEOUT", "15"))
        self.response_format = (os.getenv("SILICONFLOW_TTS_RESPONSE_FORMAT", "wav") or "wav").strip().lower()
        self.last_latency_ms = 0

    def _stable_voice_pick(self, seed_text: str) -> str:
        voices = SILICONFLOW_CONFIG.tts_official_voices.get(self.model) or []
        if not voices:
            return self.default_voice
        seed = (seed_text or "default").strip().lower()
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(voices)
        return str(voices[idx]).strip() or self.default_voice

    def _resolve_voice(self, emotion: str = "neutral", voice: str = None, speaker_profile: Optional[Dict] = None) -> str:
        if voice:
            return voice
        profile = speaker_profile or {}
        profile_voice = str(
            profile.get("tts_voice")
            or profile.get("voice")
            or profile.get("voice_id")
            or ""
        ).strip()
        if profile_voice:
            return profile_voice

        tts_role = str(
            profile.get("tts_role")
            or profile.get("ttsRole")
            or profile.get("tts角色")
            or ""
        ).strip()
        if not tts_role and profile:
            try:
                tts_role = _infer_tts_role_for_character(profile)
            except Exception:
                tts_role = ""
        if tts_role:
            alias_voice = _voice_from_tts_role_alias(tts_role, self.model)
            if alias_voice:
                return alias_voice

        role_text = " ".join(
            [
                tts_role,
                str(profile.get("identity") or profile.get("identity_tag") or ""),
                str(profile.get("role") or ""),
                str(profile.get("name") or ""),
                str(profile.get("personality") or ""),
                str(profile.get("background") or ""),
            ]
        ).strip().lower()
        if role_text:
            for role_key, role_voice in SILICONFLOW_CONFIG.preset_role_voice_map.items():
                if str(role_key).strip().lower() in role_text and role_voice:
                    return role_voice
        gender = normalize_gender(str(profile.get("gender") or profile.get("sex") or ""))
        age_group = str(profile.get("age_group") or "").strip().lower() or infer_age_group(profile.get("age"))

        if not (gender and age_group):
            hint_text = " ".join(
                [
                    str(profile.get("name") or ""),
                    str(profile.get("role") or ""),
                    str(profile.get("personality") or ""),
                    str(profile.get("background") or ""),
                ]
            ).strip()
            inferred = infer_demographics_from_text(hint_text)
            gender = gender or inferred.get("gender", "")
            age_group = age_group or inferred.get("age_group", "adult")

        if gender and age_group:
            role_voice = SILICONFLOW_CONFIG.tts_role_voice_map.get(f"{gender}:{age_group}")
            if role_voice:
                return role_voice
        # Keep role speech identity stable and distinct even when mapping misses.
        if profile:
            stable_seed = " ".join(
                [
                    tts_role,
                    str(profile.get("name") or ""),
                    str(profile.get("role") or ""),
                ]
            ).strip()
            return self._stable_voice_pick(stable_seed or "npc")
        return self._EMOTION_TO_VOICE.get((emotion or "neutral").lower(), self.default_voice)

    def synthesize(self, text: str, emotion: str = "neutral", voice: str = None, speaker_profile: Optional[Dict] = None) -> Optional[bytes]:
        content = (text or "").strip()
        if not content:
            return None
        chosen_voice = self._resolve_voice(emotion=emotion, voice=voice, speaker_profile=speaker_profile)
        t0 = time.perf_counter()
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": content,
            "response_format": self.response_format,
        }
        if chosen_voice:
            payload["voice"] = chosen_voice
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        self.last_latency_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code >= 400:
            raise RuntimeError(f"SiliconFlow TTS failed: HTTP {resp.status_code} {resp.text[:240]}")
        return resp.content if resp.content else None


class _MockResult:
    def __init__(self, stage: str, data: Dict):
        self.stage = stage
        self.data = data


class _MockThinkResult:
    def __init__(self, content: str, speaker: str):
        self.content = content
        self.metadata = {"speaker": speaker}


class _MockDialogueAgent:
    def think(self, context: Dict):
        chars = context.get("characters") or []
        speaker = chars[0].get("name", "主持人") if chars else "主持人"
        return _MockThinkResult("欢迎进入多NPC实战演练，我们开始吧。", speaker)


class _MockMultiAgent:
    def __init__(self):
        self.agents_list = [_MockDialogueAgent()]


class FallbackTalkArenaEngine:
    """轻量回退引擎：模型不可用时保障前端全流程可用。"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.multi_agent = _MockMultiAgent()

    def start_session(self, scenario_id: str, characters: List[Dict], scene_name: str, scene_description: str, user_info: Dict):
        session_id = f"demo_{len(self.sessions) + 1}"
        normalized_chars = []
        for c in (characters or []):
            if not isinstance(c, dict):
                continue
            item = dict(c)
            if not item.get("name") and item.get("n"):
                item["name"] = item.get("n")
            if not item.get("role") and item.get("r"):
                item["role"] = item.get("r")
            normalized_chars.append(item)
        self.sessions[session_id] = {
            "scenario_id": scenario_id or "shandong_dinner",
            "scenario": {"characters": normalized_chars or [{"name": "主持人", "role": "引导者"}]},
            "turn": 0,
            "scene_name": scene_name,
            "dominance": {"user": 50, "ai": 50},
            "history": [],
        }
        return session_id

    def process_turn(self, session_id: str, message: str, multimodal: Dict):
        session = self.sessions[session_id]
        session["turn"] += 1
        turn = session["turn"]
        chars = session["scenario"].get("characters", [])
        speaker = "主持人"
        if chars:
            s = chars[turn % len(chars)]
            speaker = s.get("name") or s.get("n") or "主持人"

        emotion = (multimodal or {}).get("emotion", {})
        nervous = int(emotion.get("nervous", 20))
        focus = int(emotion.get("focus", 60))
        confidence = int(emotion.get("confidence", 55))

        response_quality = max(45, min(95, 65 + (focus - nervous) // 3))
        pressure_handling = max(40, min(95, 68 + (confidence - nervous) // 4))
        emotional_intelligence = max(45, min(95, 66 + (focus // 8)))
        cultural_fit = max(45, min(95, 64 + (confidence // 10)))

        quality_label = "优秀" if response_quality >= 75 and pressure_handling >= 72 else "良好" if response_quality >= 65 else "待提升"
        judgment = f"NPC反馈质量评估：{quality_label}。建议继续用结构化表达并给出量化证据。"

        payload = {
            "ai_text": "我们继续往下聊，你先把最关键的一点讲具体一点。",
            "speaker": speaker,
            "judgment": judgment,
            "npc_feedback_quality": {
                "label": quality_label,
                "response_quality": response_quality,
                "pressure_handling": pressure_handling,
            },
            "new_dominance": {"user": 50 + min(20, turn * 2), "ai": 50 - min(20, turn * 2)},
            "scores": {
                "emotional_intelligence": emotional_intelligence,
                "response_quality": response_quality,
                "pressure_handling": pressure_handling,
                "cultural_fit": cultural_fit,
            },
            "game_over": turn >= 4,
        }
        if message:
            session.setdefault("history", []).append({"speaker": "用户", "text": message})
        session.setdefault("history", []).append({"speaker": speaker, "text": payload["ai_text"]})
        session["dominance"] = payload.get("new_dominance", session.get("dominance", {"user": 50, "ai": 50}))
        yield _MockResult("complete", payload)

    def _generate_rescue_by_prompt(self, prompt: str) -> str:
        api_key = _siliconflow_api_key()
        base_url = _siliconflow_base_url()
        model = os.getenv("SILICONFLOW_LLM_MODEL", SILICONFLOW_CONFIG.llm_model_default).strip()
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是高情商救场助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 220,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"rescue llm failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json() or {}
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()

    def get_rescue_suggestion(self, session_id: str):
        session = self.sessions.get(session_id) or {}
        scenario_id = session.get("scenario_id", "shandong_dinner")
        scene_name = session.get("scene_name", "场景")
        dominance = session.get("dominance", {"user": 50, "ai": 50})
        user_dominance = int((dominance or {}).get("user", 50))
        ai_dominance = int((dominance or {}).get("ai", 50))
        npc_list = (session.get("scenario") or {}).get("characters", []) or []
        ai_name = (npc_list[0] or {}).get("name", "AI") if npc_list else "AI"
        history = session.get("history", []) or []
        context = ""
        for turn in history[-6:]:
            speaker = str(turn.get("speaker", "NPC"))
            text = str(turn.get("text", ""))
            if text:
                context += f"{speaker}：{text}\n"

        from core.prompts.registry import get_rescue_master_prompt
        prompt = get_rescue_master_prompt(
            scenario_id=scenario_id,
            scene_name=scene_name,
            ai_name=ai_name,
            user_dominance=user_dominance,
            ai_dominance=ai_dominance,
            context=context,
        )
        try:
            suggestion = self._generate_rescue_by_prompt(prompt)
            if suggestion:
                return suggestion
        except Exception as e:
            print(f"[FallbackEngine] rescue generate failed: {e}")
        return "救场生成失败，请重试。"

    def end_session(self, session_id: str):
        session = self.sessions.get(session_id, {})
        chars = (session.get("scenario") or {}).get("characters", []) or []
        npc_os_list = []
        for c in chars:
            name = c.get("name") or c.get("n") or "NPC"
            avatar = c.get("avatar") or c.get("a") or "👤"
            npc_os_list.append(
                {
                    "name": name,
                    "avatar": avatar,
                    "os": "整体表现还不错，继续用结构化表达会更稳。",
                }
            )
        return {
            "scene_name": session.get("scene_name", "模拟对话"),
            "medal": "🥇",
            "scores": {
                "oily": 78,
                "friendliness": 82,
                "logic": 79,
                "humor": 74,
                "respect": 81,
                "total": 81,
            },
            "summary": "你在多NPC环境中维持了稳定表达，能在追问下保持结构。",
            "suggestion": "下一轮提升点：减少重复句，增加结果数字与复盘反思。",
            "npc_os_list": npc_os_list,
        }


def get_engine():
    global engine
    if engine is None:
        _ensure_llm_env_defaults()
        from model_loader import LLMLoader
        from core.engine import TalkArenaEngine

        llm = LLMLoader()
        llm.load()
        print(f"[LLM] provider={llm.provider} model={llm.model_name} base_url={llm.base_url}")
        # TTS is provided by SiliconFlow API routes in this service.
        # Disable legacy local TTSLoader bootstrap (edge_tts dependency).
        engine = TalkArenaEngine(llm, enable_tts=False, use_unified_agent=True)
    return engine


class InterruptReq(BaseModel):
    session_id: str


def get_mm_analyzer():
    global mm_analyzer
    if mm_analyzer is None:
        try:
            from core.multimodal_analyzer import MultimodalAnalyzer

            mm_analyzer = MultimodalAnalyzer()
        except Exception as e:
            raise RuntimeError(
                "Multimodal analyzer is unavailable. Ensure related dependencies are installed."
            ) from e
    return mm_analyzer


def get_stt_service():
    global stt_service
    if stt_service is None:
        if not _truthy_env("SILICONFLOW_STT_ENABLED", "1"):
            raise RuntimeError("SILICONFLOW_STT_ENABLED is disabled; fallback is removed.")
        stt_service = SiliconFlowSTTService()
        print(f"[STT] using siliconflow model={stt_service.model}")
    return stt_service


def get_tts_service():
    global tts_service
    if tts_service is None:
        if not _truthy_env("SILICONFLOW_TTS_ENABLED", "1"):
            raise RuntimeError("SILICONFLOW_TTS_ENABLED is disabled; fallback is removed.")
        tts_service = SiliconFlowTTSService()
        print(f"[TTS] using siliconflow model={tts_service.model} voice={tts_service.default_voice}")
    return tts_service


@app.on_event("startup")
async def _prewarm_runtime():
    t0 = time.perf_counter()
    try:
        get_engine()
    except Exception as e:
        print(f"[Warmup] get_engine failed: {e}")
    try:
        get_tts_service()
        # Trigger one tiny remote synth to avoid first-turn TTS cold-start.
        if _truthy_env("TTS_REMOTE_PREWARM_ENABLED", "1"):
            tts = get_tts_service()
            try:
                _ = tts.synthesize(
                    "准备好了",
                    emotion="neutral",
                    speaker_profile={"tts_role": "diana", "gender": "female", "age_group": "adult"},
                )
                print(f"[Warmup] tts_remote_prewarm_ms={int(getattr(tts, 'last_latency_ms', 0) or 0)}")
            except Exception as tts_warm_err:
                print(f"[Warmup] tts_remote_prewarm failed: {tts_warm_err}")
    except Exception as e:
        print(f"[Warmup] get_tts_service failed: {e}")
    print(f"[Warmup] startup prewarm done cost_ms={int((time.perf_counter()-t0)*1000)}")


class ChatReq(BaseModel):
    session_id: str
    message: str = ""
    chat_history: Optional[List[Dict]] = []
    multimodal: Optional[Dict] = None
    scenario_id: Optional[str] = None
    scene_name: Optional[str] = None
    scene_description: Optional[str] = None
    characters: Optional[List[Dict]] = None


class SessionReq(BaseModel):
    scenario_id: str = "shandong_dinner"
    scene_name: str = "家庭饭桌试炼"
    characters: Optional[List[Dict]] = []
    scene_description: Optional[str] = ""
    user_info: Optional[Dict] = None
    pressure_tags: Optional[List[str]] = []
    pressure_value: Optional[int] = 5
    drinking_capacity: Optional[int] = 0


class MMReq(BaseModel):
    text: str
    emotion_features: Optional[Dict] = None
    voice_features: Optional[Dict] = None


class ClientLogReq(BaseModel):
    level: str = "info"
    message: str
    payload: Optional[Dict] = None


class ScenarioGenerateReq(BaseModel):
    scene_type: str = "shandong_dinner"
    scene_name: str = "家庭饭桌试炼"
    only_characters: bool = False
    banquet_level: Optional[str] = None


class InterviewQuestionReq(BaseModel):
    industry: str
    position: str


class ContentOptimizeReq(BaseModel):
    content: str
    scene_type: str


class TTSReq(BaseModel):
    text: str
    emotion: Optional[str] = "neutral"
    speaker: Optional[str] = None
    speaker_profile: Optional[Dict] = None


class ShareReportImageReq(BaseModel):
    image_data: str


def _rehydrate_session_from_request(eng, req: ChatReq) -> bool:
    """Rebuild a minimal session from client payload when server memory is missing."""
    try:
        if not hasattr(eng, "sessions") or not req.session_id:
            return False
        if req.session_id in eng.sessions:
            return True
        raw_history = req.chat_history or []
        if not isinstance(raw_history, list) or not raw_history:
            return False

        scenario_id = (req.scenario_id or "shandong_dinner").strip() or "shandong_dinner"
        scenario_map = getattr(eng, "scenarios", {}) or {}
        scenario = copy.deepcopy(
            scenario_map.get(scenario_id) or scenario_map.get("shandong_dinner") or {}
        )
        if not isinstance(scenario, dict):
            scenario = {}
        if isinstance(req.characters, list) and req.characters:
            scenario["characters"] = req.characters
        if req.scene_description:
            scenario["description"] = str(req.scene_description)

        unified_history: List[Dict] = []
        history: List[Dict] = []
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            text = str(item.get("content") or item.get("text") or "").strip()
            if not text:
                continue
            role = str(item.get("role") or "").strip().lower()
            is_user = role in {"user", "human", "用户"}
            speaker = str(item.get("speaker") or item.get("npc_id") or "").strip()
            if not speaker:
                speaker = "用户" if is_user else "NPC"
            unified_history.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "is_user": is_user,
                    "timestamp_ms": int(time.time() * 1000),
                }
            )
            history.append({"speaker": speaker, "text": text})

        if not unified_history:
            return False

        eng.sessions[req.session_id] = {
            "scenario_id": scenario_id,
            "scenario": scenario,
            "scene_name": req.scene_name or scenario.get("name", "本轮会话"),
            "turn_count": 0,
            "dominance": {"user": 50, "ai": 50},
            "history": history,
            "scores_history": [],
            "unified_history": unified_history,
            "chat_history": raw_history,
            "pressure_tags": [],
            "pressure_value": 5,
            "drinking_capacity": 0,
        }
        _save_session_snapshot(eng, req.session_id)
        print(
            f"[SessionRehydrate] rebuilt sid={req.session_id} "
            f"history={len(unified_history)} scenario_id={scenario_id}"
        )
        return True
    except Exception as e:
        print(f"[SessionRehydrate] failed sid={getattr(req, 'session_id', '')}: {e}")
        return False


def _load_scene_preset_prompts() -> Dict[str, Dict[str, str]]:
    default_map: Dict[str, Dict[str, str]] = {
        "shandong_dinner": {
            "主持人": "各位先入座，先碰个杯，咱们按礼数慢慢聊。",
            "主陪": "先走一个，咱今天讲究气氛，但话也要讲明白。",
            "长辈": "别急，先把你的想法说完整，礼数和分寸都照顾到。",
            "大舅": "来，先碰一下，你这阵子的打算给大家交个底。",
            "大妗子": "你慢慢说，把细节落到实处，大家好帮你拿主意。",
            "表哥": "我先热个场，咱们实话实说，别把话题聊散了。",
        },
        "business_dinner": {
            "主持人": "先对齐目标，再聊合作边界和落地节奏。",
            "甲方负责人": "我们先看结果目标，资源和时间窗口后面细谈。",
            "乙方商务": "我先给可落地方案，再讲成本与排期。",
            "风险顾问": "我先把风险点摆出来，避免后面返工。",
        },
        "interview": {
            "面试官": "我们直接进入问题，请你先做一分钟自我介绍。",
            "hr": "先别紧张，回答尽量结论先行、结构清晰。",
            "竞争者": "我先给一个思路框架，后面欢迎你补充反驳。",
        },
        "debate": {
            "主持人": "先明确议题定义，再进入立论和反驳环节。",
            "正方辩手": "我先立论，核心观点与证据链如下。",
            "反方辩手": "我先指出前提漏洞，再给反证。",
        },
        "_default": {
            "主持人": "我们开始吧，先把关键问题摆到桌面上。",
            "引导者": "先热个场，你先说最核心的一点。",
            "同事": "我先补充背景，方便大家对齐。",
        },
    }
    if not SCENE_PRESET_PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Preset prompts file not found: {SCENE_PRESET_PROMPTS_PATH}")
    try:
        raw = SCENE_PRESET_PROMPTS_PATH.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Preset prompts load failed: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError("Preset prompts JSON must be an object keyed by scenario_id.")
    normalized: Dict[str, Dict[str, str]] = {}
    for scene_id, scene_map in data.items():
        if isinstance(scene_map, dict):
            normalized[str(scene_id)] = {
                str(k): str(v) for k, v in scene_map.items() if str(k).strip() and str(v).strip()
            }
    if not normalized:
        raise RuntimeError("Preset prompts JSON is empty after normalization.")
    return normalized


SCENE_ROLE_OPENING_TEMPLATES: Dict[str, Dict[str, str]] = _load_scene_preset_prompts()
def _character_name(char: Dict) -> str:
    return str(char.get("name") or char.get("n") or "").strip()


def _character_role(char: Dict) -> str:
    return str(char.get("role") or char.get("r") or "").strip()


def _infer_identity_tag(char: Dict) -> str:
    explicit = str(char.get("identity") or char.get("identity_tag") or "").strip().lower()
    if explicit:
        return explicit
    raw = " ".join(
        [
            _character_name(char),
            _character_role(char),
            str(char.get("personality") or char.get("p") or ""),
            str(char.get("background") or char.get("b") or ""),
        ]
    ).lower()
    if any(k in raw for k in ["主陪", "长辈", "领导", "负责人", "面试官", "主持人", "总"]):
        return "senior"
    if any(k in raw for k in ["顾问", "风控", "法务", "评审", "点评"]):
        return "advisor"
    if any(k in raw for k in ["商务", "销售", "合作", "客户", "甲方", "乙方"]):
        return "business"
    if any(k in raw for k in ["竞争者", "晚辈", "新人", "学生", "表哥", "表姐", "表弟", "表妹"]):
        return "junior"
    if any(k in raw for k in ["辩手", "技术", "工程", "产品"]):
        return "specialist"
    return "neutral"


def _moss_voice_aliases() -> List[str]:
    return ["alex", "anna", "bella", "benjamin", "charles", "claire", "david", "diana"]


def _normalize_tts_role_alias(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if ":" in raw:
        raw = raw.split(":")[-1]
    raw_l = raw.lower()
    return raw_l if raw_l in _moss_voice_aliases() else ""


def _infer_gender_age(char: Dict) -> Dict[str, str]:
    gender = normalize_gender(str(char.get("gender") or char.get("sex") or ""))
    age_group = str(char.get("age_group") or char.get("ageGroup") or "").strip().lower() or infer_age_group(char.get("age"))
    if gender and age_group:
        return {"gender": gender, "age_group": age_group}
    hint_text = " ".join(
        [
            _character_name(char),
            _character_role(char),
            str(char.get("personality") or char.get("p") or ""),
            str(char.get("background") or char.get("b") or ""),
        ]
    ).strip()
    inferred = infer_demographics_from_text(hint_text)
    return {
        "gender": gender or inferred.get("gender", ""),
        "age_group": age_group or inferred.get("age_group", "adult"),
    }


def _pick_tts_role_by_profile(gender: str, age_group: str, identity: str) -> str:
    g = (gender or "").strip().lower()
    a = (age_group or "").strip().lower()
    i = (identity or "").strip().lower()

    # 基线：性别 -> 年龄
    if g == "male":
        if a == "child":
            base = "benjamin"
        elif a == "youth":
            base = "david"
        elif a == "elder":
            base = "charles"
        else:
            base = "alex"
    elif g == "female":
        if a == "child":
            base = "anna"
        elif a == "youth":
            base = "bella"
        elif a == "elder":
            base = "claire"
        else:
            base = "diana"
    else:
        base = "diana"

    # 第三层：身份微调
    identity_override = {
        "senior": "charles" if g == "male" else "claire",
        "advisor": "benjamin" if g == "male" else "claire",
        "business": "alex" if g == "male" else "diana",
        "junior": "david" if g == "male" else "bella",
        "specialist": "alex" if g == "male" else "anna",
    }
    return identity_override.get(i, base)


def _infer_tts_role_for_character(char: Dict) -> str:
    explicit = _normalize_tts_role_alias(
        str(char.get("tts_role") or char.get("ttsRole") or char.get("tts角色") or "")
    )
    if explicit:
        return explicit
    ga = _infer_gender_age(char)
    identity = _infer_identity_tag(char)
    return _pick_tts_role_by_profile(ga.get("gender", ""), ga.get("age_group", ""), identity)


def _voice_from_tts_role_alias(tts_role: str, model: str = "") -> str:
    alias = _normalize_tts_role_alias(tts_role)
    if not alias:
        return ""
    m = str(model or SILICONFLOW_CONFIG.tts_model_default).strip()
    return f"{m}:{alias}"


def _normalize_characters_tts_fields(characters: List[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for c in (characters or []):
        if not isinstance(c, dict):
            continue
        item = dict(c)
        ga = _infer_gender_age(item)
        if ga.get("gender") and not item.get("gender"):
            item["gender"] = ga.get("gender")
        if ga.get("age_group") and not item.get("age_group"):
            item["age_group"] = ga.get("age_group")
        if not item.get("identity"):
            item["identity"] = _infer_identity_tag(item)
        tts_role = _infer_tts_role_for_character(item)
        item["tts_role"] = tts_role
        item["tts角色"] = tts_role
        voice = str(item.get("tts_voice") or "").strip() or _voice_from_tts_role_alias(tts_role, SILICONFLOW_CONFIG.tts_model_default)
        if not voice and ga.get("gender") and ga.get("age_group"):
            voice = str(SILICONFLOW_CONFIG.tts_role_voice_map.get(f"{ga.get('gender')}:{ga.get('age_group')}") or "").strip()
        item["tts_voice"] = voice
        normalized.append(item)
    return normalized


def _build_preset_opening_line(char: Dict, scenario_id: str = "") -> str:
    name = _character_name(char)
    role = _character_role(char)
    name_l = name.lower()
    role_l = role.lower()
    scene_templates = SCENE_ROLE_OPENING_TEMPLATES.get(scenario_id, {})
    merged_templates = {**SCENE_ROLE_OPENING_TEMPLATES.get("_default", {}), **scene_templates}

    for key, text in merged_templates.items():
        key_l = key.lower()
        if key_l and key_l in name_l:
            return text

    for key, text in merged_templates.items():
        key_l = key.lower()
        if key_l and key_l in role_l:
            return text

    if role:
        return f"我是{name or role}，我先开个头：我们先把重点摆清楚再往下聊。"
    if name:
        return f"我是{name}，我先说一句：先把真实情况讲明白，后面才好推进。"
    return "我们开始吧，先把当下最关键的问题摆到桌面上。"

def _build_preset_opening_utterances(characters: List[Dict], scenario_id: str = "") -> List[Dict]:
    utterances: List[Dict] = []
    for i, c in enumerate(characters or []):
        speaker = _character_name(c)
        if not speaker:
            continue
        utterances.append(
            {
                "npc_id": speaker,
                "text": _build_preset_opening_line(c, scenario_id=scenario_id),
                "emotion": "neutral",
                "delay_ms": 320 + i * 120,
            }
        )
    return utterances


def _is_fixed_tts_phrase(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    tn = "".join(t.split())
    for scene_map in SCENE_ROLE_OPENING_TEMPLATES.values():
        for v in scene_map.values():
            if tn == "".join(str(v).split()):
                return True
    return False


def _tts_cache_key(req: TTSReq, service) -> str:
    # Fixed phrases should share cache aggressively to reduce cold-start latency.
    payload = {
        "model": getattr(service, "model", ""),
        "format": getattr(service, "response_format", "wav"),
        "text": (req.text or "").strip(),
    }
    digest = hashlib.sha1(str(payload).encode("utf-8")).hexdigest()[:20]
    return f"fixed_{digest}"


def _fixed_mp3_filename(req: TTSReq, service) -> str:
    voice_key = ""
    try:
        if hasattr(service, "_resolve_voice"):
            voice_key = str(
                service._resolve_voice(
                    emotion=req.emotion or "neutral",
                    speaker_profile=req.speaker_profile,
                )
                or ""
            ).strip()
    except Exception:
        voice_key = ""
    payload = {
        "model": getattr(service, "model", ""),
        "voice": voice_key,
        "text": (req.text or "").strip(),
    }
    digest = hashlib.sha1(str(payload).encode("utf-8")).hexdigest()[:24]
    return f"fixed_{digest}.mp3"


def _ensure_fixed_mp3(req: TTSReq, service) -> str:
    filename = _fixed_mp3_filename(req, service)
    out_path = AUDIO_OUTPUT_DIR / filename
    if out_path.exists():
        return f"/audio/{filename}"

    original_format = str(getattr(service, "response_format", "wav") or "wav")
    try:
        if hasattr(service, "response_format"):
            service.response_format = "mp3"
        audio_bytes = service.synthesize(
            req.text,
            req.emotion or "neutral",
            speaker_profile=req.speaker_profile,
        )
    finally:
        if hasattr(service, "response_format"):
            service.response_format = original_format
    if not audio_bytes:
        return ""

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    return f"/audio/{filename}"


def _find_character_profile(characters: List[Dict], speaker: str) -> Optional[Dict]:
    target = (speaker or "").strip()
    if not target:
        return None
    for c in (characters or []):
        name = str(c.get("name") or c.get("n") or "").strip()
        alias = str(c.get("alias") or c.get("a") or "").strip()
        if target == name or (alias and target == alias):
            return c
    return None


def _tts_url_from_req(req: TTSReq, service) -> str:
    cacheable = _is_fixed_tts_phrase(req.text)
    if cacheable:
        fixed_url = _ensure_fixed_mp3(req, service)
        if fixed_url:
            return fixed_url

    cache_key = _tts_cache_key(req, service) if cacheable else ""
    if cache_key and cache_key in TTS_FIXED_AUDIO_CACHE:
        cached_file = TTS_FIXED_AUDIO_CACHE.get(cache_key, "")
        if cached_file and (AUDIO_OUTPUT_DIR / cached_file).exists():
            return f"/audio/{cached_file}"

    audio_bytes = service.synthesize(req.text, req.emotion or "neutral", speaker_profile=req.speaker_profile)
    if not audio_bytes:
        return ""

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"tts_{int(time.time() * 1000)}.wav"
    out_path = AUDIO_OUTPUT_DIR / filename
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    if cache_key:
        TTS_FIXED_AUDIO_CACHE[cache_key] = filename
        TTS_FIXED_AUDIO_CACHE.move_to_end(cache_key)
        while len(TTS_FIXED_AUDIO_CACHE) > TTS_FIXED_CACHE_MAX_ITEMS:
            TTS_FIXED_AUDIO_CACHE.popitem(last=False)
    return f"/audio/{filename}"


def _async_warmup_tts(utterances: List[Dict], characters: List[Dict]) -> None:
    try:
        service = get_tts_service()
    except Exception as e:
        print(f"[PresetTTS] warmup init skipped: {e}")
        return

    for u in (utterances or []):
        text = str(u.get("text") or "").strip()
        speaker = str(u.get("npc_id") or "").strip()
        if not text or not _is_fixed_tts_phrase(text):
            continue
        profile = _find_character_profile(characters or [], speaker)
        req = TTSReq(
            text=text,
            emotion=str(u.get("emotion") or "neutral"),
            speaker=speaker or None,
            speaker_profile=profile,
        )
        try:
            _tts_url_from_req(req, service)
        except Exception as e:
            print(f"[PresetTTS] warmup failed for speaker={speaker}: {e}")


def _prebuild_utterance_tts(
    utterances: List[Dict],
    characters: List[Dict],
    force_generate_fixed: bool = False,
) -> List[Dict]:
    if not utterances:
        return utterances
    t0 = time.perf_counter()
    try:
        service = get_tts_service()
    except Exception as e:
        print(f"[PresetTTS] init skipped: {e}")
        return utterances

    attached_count = 0
    generated_count = 0
    miss_count = 0
    for u in utterances:
        text = str(u.get("text") or "").strip()
        speaker = str(u.get("npc_id") or "").strip()
        if not text:
            continue
        profile = _find_character_profile(characters or [], speaker)
        req = TTSReq(
            text=text,
            emotion=str(u.get("emotion") or "neutral"),
            speaker=speaker or None,
            speaker_profile=profile,
        )
        try:
            cacheable = _is_fixed_tts_phrase(text)
            url = ""
            if cacheable:
                fixed_name = _fixed_mp3_filename(req, service)
                fixed_path = AUDIO_OUTPUT_DIR / fixed_name
                if fixed_path.exists():
                    url = f"/audio/{fixed_name}"
                elif force_generate_fixed:
                    url = _ensure_fixed_mp3(req, service)
                    if url:
                        generated_count += 1
            if url:
                u["tts_url"] = url
                attached_count += 1
            else:
                miss_count += 1
        except Exception as e:
            print(f"[PresetTTS] synth failed for speaker={speaker}: {e}")
    # Warm up fixed phrases in background; do not block API latency.
    try:
        th = threading.Thread(
            target=_async_warmup_tts,
            args=(utterances, characters),
            daemon=True,
        )
        th.start()
    except Exception as e:
        print(f"[PresetTTS] warmup thread start failed: {e}")
    total_ms = int((time.perf_counter() - t0) * 1000)
    print(
        "[PresetTTS] prebuild "
        f"force_generate_fixed={force_generate_fixed} "
        f"utterances={len(utterances)} "
        f"attached={attached_count} generated={generated_count} miss={miss_count} "
        f"cost_ms={total_ms}"
    )
    return utterances


def _siliconflow_chat_generate(prompt: str, system_prompt: str = "你是高情商救场助手。", temperature: float = 0.7, max_tokens: int = 220) -> str:
    api_key = _siliconflow_api_key()
    base_url = _siliconflow_base_url()
    model = os.getenv("SILICONFLOW_LLM_MODEL", SILICONFLOW_CONFIG.llm_model_default).strip()
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"chat/completions failed: HTTP {resp.status_code} {resp.text[:220]}")
    data = resp.json() or {}
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


def _build_rescue_context_from_session(session: Dict) -> Dict[str, str]:
    scenario_id = session.get("scenario_id", "shandong_dinner")
    scene_name = session.get("scene_name", "场景")
    dominance = session.get("dominance", {"user": 50, "ai": 50}) or {"user": 50, "ai": 50}
    user_dominance = int(dominance.get("user", 50) or 50)
    ai_dominance = int(dominance.get("ai", 50) or 50)

    npc_list = (session.get("scenario") or {}).get("characters", []) or []
    ai_name = (npc_list[0] or {}).get("name") if npc_list else ""
    if not ai_name and npc_list:
        ai_name = (npc_list[0] or {}).get("n", "")
    ai_name = ai_name or "AI"

    lines: List[str] = []
    for t in (session.get("history") or [])[-6:]:
        if not isinstance(t, dict):
            continue
        if t.get("user"):
            lines.append(f"用户：{t.get('user')}")
        if t.get("ai"):
            lines.append(f"{t.get('speaker') or 'NPC'}：{t.get('ai')}")
        if t.get("text"):
            lines.append(f"{t.get('speaker') or 'NPC'}：{t.get('text')}")
    if not lines:
        for t in (session.get("unified_history") or [])[-8:]:
            if not isinstance(t, dict):
                continue
            speaker = t.get("speaker") or ("用户" if t.get("is_user") else "NPC")
            text = t.get("text") or ""
            if text:
                lines.append(f"{speaker}：{text}")

    return {
        "scenario_id": scenario_id,
        "scene_name": scene_name,
        "ai_name": ai_name,
        "user_dominance": user_dominance,
        "ai_dominance": ai_dominance,
        "context": "\n".join(lines).strip(),
    }


def _save_share_report_image(image_data: str) -> Dict[str, str]:
    image_bytes = _decode_png_data_url(image_data)

    SHARE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    token_seed = f"{time.time_ns()}:{len(image_bytes)}:{hashlib.sha1(image_bytes).hexdigest()}"
    token = hashlib.sha1(token_seed.encode("utf-8")).hexdigest()[:20]
    filename = f"{token}.png"
    out = SHARE_REPORT_DIR / filename
    with open(out, "wb") as f:
        f.write(image_bytes)

    SHARE_REPORT_INDEX[token] = filename
    SHARE_REPORT_INDEX.move_to_end(token)
    while len(SHARE_REPORT_INDEX) > SHARE_REPORT_MAX_ITEMS:
        old_token, old_filename = SHARE_REPORT_INDEX.popitem(last=False)
        old_path = SHARE_REPORT_DIR / old_filename
        try:
            if old_path.exists():
                old_path.unlink()
        except Exception as cleanup_err:
            print(f"[Share] cleanup failed token={old_token}: {cleanup_err}")
    public_url = _upload_report_image_public(image_bytes)
    share_path = f"/share/report/{token}"
    public_base = _public_base_url()
    share_url = f"{public_base}{share_path}" if public_base else share_path
    return {
        "token": token,
        "image_url": f"/api/share/report-image/{filename}",
        "share_url": share_url,
        "public_url": public_url,
    }


@app.get("/favicon.ico")
async def favicon():
    return Response(
        base64.b64decode(
            "AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wD///8AiaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/////wD///8A////AP///wD///8A////AP///wD///8AiaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v////8A////AP///wD///8A////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/////AP///wD///8A////AP///wD///8A////AP///wD///8AiaT6/4mk+v+JpPr/iaT6/4mk+v+JpPr/iaT6/////wD///8A////AP///wD///8A////AP///wD///8A////AP///wCJpPr/iaT6/4mk+v+JpPr/iaT6/4mk+v////8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8AiaT6/4mk+v+JpPr/////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A//8AAP//AAD//wAA//8AAOAfAADADwAAwAcAAMAHAADgBwAA8A8AAOAfAADADwAAwA8AAOAfAAD//wAA//8AAA=="
        ),
        media_type="image/x-icon",
    )


@app.get("/")
async def index():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "features": ["multi-agent", "rag", "decision-engine", "anti-hallucination"],
    }


@app.post("/api/session/start")
async def start_session(req: SessionReq):
    req_t0 = time.perf_counter()
    print(
        "[SessionStart] begin "
        f"scenario_id={req.scenario_id} scene_name={req.scene_name} "
        f"chars={len(req.characters or [])}"
    )
    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}

    try:
        step_t0 = time.perf_counter()
        try:
            session_id = eng.start_session(
                scenario_id=req.scenario_id,
                characters=req.characters or [],
                scene_name=req.scene_name,
                scene_description=req.scene_description,
                user_info=req.user_info,
                pressure_tags=req.pressure_tags or [],
                pressure_value=req.pressure_value or 5,
                drinking_capacity=req.drinking_capacity or 0,
            )
        except TypeError:
            # Backward compatibility for lightweight test/mocked engines.
            session_id = eng.start_session(
                scenario_id=req.scenario_id,
                characters=req.characters or [],
                scene_name=req.scene_name,
                scene_description=req.scene_description,
                user_info=req.user_info,
            )
        start_cost_ms = int((time.perf_counter() - step_t0) * 1000)
        print(f"[SessionStart] engine.start_session cost_ms={start_cost_ms} session_id={session_id}")

        if hasattr(eng, 'use_unified_agent') and eng.use_unified_agent:
            step_t0 = time.perf_counter()
            preset_utterances = _build_preset_opening_utterances(req.characters or [], scenario_id=req.scenario_id)
            build_cost_ms = int((time.perf_counter() - step_t0) * 1000)
            step_t0 = time.perf_counter()
            preset_utterances = _prebuild_utterance_tts(
                preset_utterances,
                req.characters or [],
                force_generate_fixed=True,
            )
            tts_cost_ms = int((time.perf_counter() - step_t0) * 1000)
            with_url = sum(1 for u in (preset_utterances or []) if str(u.get("tts_url") or "").strip())
            print(
                "[SessionStart] unified preset "
                f"build_cost_ms={build_cost_ms} tts_cost_ms={tts_cost_ms} "
                f"utterances={len(preset_utterances or [])} with_tts_url={with_url}"
            )
            if preset_utterances:
                _save_session_snapshot(eng, session_id)
                return {
                    "success": True,
                    "data": {
                        "session_id": session_id,
                        "is_unified_agent": True,
                        "utterances": preset_utterances,
                        "should_await_user": True,
                    },
                    "meta": {"latency_ms": int((time.perf_counter() - req_t0) * 1000), "stage": "session_start"},
                }
            print("[SessionStart] unified preset empty, fallback to process_turn")
            for result in eng.process_turn(session_id, "", is_interrupt=False):
                if result.stage == "complete":
                    utterances = result.data.get("utterances", [])
                    with_url = sum(1 for u in (utterances or []) if str(u.get("tts_url") or "").strip())
                    print(
                        "[SessionStart] unified process_turn complete "
                        f"utterances={len(utterances or [])} with_tts_url={with_url}"
                    )
                    _save_session_snapshot(eng, session_id)
                    return {
                        "success": True,
                        "data": {
                            "session_id": session_id,
                            "is_unified_agent": True,
                            "utterances": utterances,
                            "should_await_user": result.data.get("should_await_user", True),
                        },
                        "meta": {"latency_ms": int((time.perf_counter() - req_t0) * 1000), "stage": "session_start"},
                    }
        else:
            session = eng.sessions[session_id]
            opening_utterances = _build_preset_opening_utterances(
                req.characters or session["scenario"].get("characters", []),
                scenario_id=req.scenario_id,
            )
            opening_utterances = _prebuild_utterance_tts(
                opening_utterances,
                req.characters or session["scenario"].get("characters", []),
                force_generate_fixed=True,
            )
            with_url = sum(1 for u in (opening_utterances or []) if str(u.get("tts_url") or "").strip())
            print(
                "[SessionStart] classic opening "
                f"utterances={len(opening_utterances or [])} with_tts_url={with_url}"
            )
            opening = None
            if not opening_utterances:
                opening = eng.multi_agent.agents_list[0].think(
                    {
                        "scenario_id": req.scenario_id,
                        "characters": req.characters or session["scenario"].get("characters", []),
                        "user_input": "",
                        "turn_count": 0,
                        "dominance": {"user": 50, "ai": 50},
                        "scene_description": req.scene_description,
                        "user_info": req.user_info,
                    }
                )

            _save_session_snapshot(eng, session_id)
            return {
                "success": True,
                "data": {
                    "session_id": session_id,
                    "opening_utterances": opening_utterances,
                    "opening": opening.content if opening else "",
                    "opening_speaker": opening.metadata.get("speaker") if opening else "",
                    "user_dominance": 50,
                    "ai_dominance": 50,
                    "features": {"multi_agent": True, "rag": True, "decision_engine": True},
                    "is_unified_agent": False,
                },
                "meta": {"latency_ms": int((time.perf_counter() - req_t0) * 1000), "stage": "session_start"},
            }

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        print(f"[SessionStart] end total_cost_ms={int((time.perf_counter() - req_t0) * 1000)}")


@app.post("/api/chat/send")
async def send_msg(req: ChatReq):
    req_t0 = time.perf_counter()
    if not req.session_id:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        _try_restore_session_snapshot(eng, req.session_id)
    if req.session_id not in eng.sessions:
        print(f"[SessionLookup] miss sid={req.session_id} chat_history_len={len(req.chat_history or [])}")
        _rehydrate_session_from_request(eng, req)
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        multimodal = req.multimodal or {}
        llm_t0 = time.perf_counter()
        mm_result = None
        mm_latency_ms = 0
        if multimodal:
            try:
                mm_t0 = time.perf_counter()
                mm_result = get_mm_analyzer().process_turn(req.message, multimodal)
                mm_latency_ms = int((time.perf_counter() - mm_t0) * 1000)
            except Exception:
                mm_result = None
        print(f"[API] 收到多模态数据: {multimodal}")
        try:
            turn_iter = eng.process_turn(
                req.session_id, req.message, multimodal, is_interrupt=False
            )
        except TypeError:
            turn_iter = eng.process_turn(req.session_id, req.message, multimodal)
        for result in turn_iter:
            if result.stage == "complete":
                payload = result.data or {}
                session = eng.sessions.get(req.session_id, {}) if hasattr(eng, "sessions") else {}
                turn_idx = int(session.get("turn_count", session.get("turn", 0)) or 0)
                if isinstance(payload.get("utterances"), list) and turn_idx <= 4:
                    chars = (session.get("scenario") or {}).get("characters", []) if isinstance(session, dict) else []
                    payload["utterances"] = _prebuild_utterance_tts(payload.get("utterances") or [], chars or [])
                _save_session_snapshot(eng, req.session_id)
                utterances = payload.get("utterances") if isinstance(payload, dict) else []
                utter_count = len(utterances) if isinstance(utterances, list) else 0
                with_tts_url = 0
                if isinstance(utterances, list):
                    with_tts_url = sum(1 for u in utterances if isinstance(u, dict) and str(u.get("tts_url") or "").strip())
                print(
                    "[ChatSend] complete "
                    f"sid={req.session_id} utterances={utter_count} with_tts_url={with_tts_url} "
                    f"llm_total_ms={int((time.perf_counter() - llm_t0) * 1000)} "
                    f"api_total_ms={int((time.perf_counter() - req_t0) * 1000)} "
                    f"mm_ms={mm_latency_ms}"
                )
                if mm_result:
                    payload["multimodal_analysis"] = mm_result
                return {
                    "success": True,
                    "data": payload,
                    "meta": {
                        "latency_ms": int((time.perf_counter() - req_t0) * 1000),
                        "mm_latency_ms": mm_latency_ms,
                        "stage": "chat_send",
                    },
                }

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chat/interrupt")
async def interrupt_chat(req: ChatReq):
    if not req.session_id:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        _try_restore_session_snapshot(eng, req.session_id)
    if req.session_id not in eng.sessions:
        print(f"[SessionLookup] miss sid={req.session_id} chat_history_len={len(req.chat_history or [])}")
        _rehydrate_session_from_request(eng, req)
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        for result in eng.process_turn(req.session_id, "", is_interrupt=True):
            if result.stage == "complete":
                payload = result.data or {}
                session = eng.sessions.get(req.session_id, {}) if hasattr(eng, "sessions") else {}
                turn_idx = int(session.get("turn_count", session.get("turn", 0)) or 0)
                if isinstance(payload.get("utterances"), list) and turn_idx <= 4:
                    chars = (session.get("scenario") or {}).get("characters", []) if isinstance(session, dict) else []
                    payload["utterances"] = _prebuild_utterance_tts(payload.get("utterances") or [], chars or [])
                _save_session_snapshot(eng, req.session_id)
                return {"success": True, "data": payload}

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chat/continue")
async def continue_chat(req: ChatReq):
    if not req.session_id:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        _try_restore_session_snapshot(eng, req.session_id)
    if req.session_id not in eng.sessions:
        print(f"[SessionLookup] miss sid={req.session_id} chat_history_len={len(req.chat_history or [])}")
        _rehydrate_session_from_request(eng, req)
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        for result in eng.process_turn(req.session_id, "", is_interrupt=False):
            if result.stage == "complete":
                payload = result.data or {}
                session = eng.sessions.get(req.session_id, {}) if hasattr(eng, "sessions") else {}
                turn_idx = int(session.get("turn_count", session.get("turn", 0)) or 0)
                if isinstance(payload.get("utterances"), list) and turn_idx <= 4:
                    chars = (session.get("scenario") or {}).get("characters", []) if isinstance(session, dict) else []
                    payload["utterances"] = _prebuild_utterance_tts(payload.get("utterances") or [], chars or [])
                _save_session_snapshot(eng, req.session_id)
                return {"success": True, "data": payload}

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chat/rescue")
async def rescue(req: ChatReq):
    if not req.session_id:
        return {"success": False, "error": "无效会话"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        _try_restore_session_snapshot(eng, req.session_id)
    if req.session_id not in eng.sessions:
        print(f"[SessionLookup] miss sid={req.session_id} chat_history_len={len(req.chat_history or [])}")
        _rehydrate_session_from_request(eng, req)
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        from core.prompts.registry import get_rescue_master_prompt

        session = eng.sessions.get(req.session_id, {}) if hasattr(eng, "sessions") else {}
        ctx = _build_rescue_context_from_session(session if isinstance(session, dict) else {})
        prompt = get_rescue_master_prompt(
            scenario_id=ctx["scenario_id"],
            scene_name=ctx["scene_name"],
            ai_name=ctx["ai_name"],
            user_dominance=ctx["user_dominance"],
            ai_dominance=ctx["ai_dominance"],
            context=ctx["context"],
        )
        try:
            suggestion = _siliconflow_chat_generate(prompt, system_prompt="你是高情商救场助手，请给可直接说出口的一段回复。")
            if suggestion:
                return {"success": True, "data": {"suggestion": suggestion}}
        except Exception as llm_err:
            raise RuntimeError(f"Rescue direct llm failed: {llm_err}") from llm_err
        raise RuntimeError("Rescue generated empty response; fallback disabled.")
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/session/end")
async def end_session(req: ChatReq):
    if not req.session_id:
        return {"success": False, "error": "无效会话"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        _try_restore_session_snapshot(eng, req.session_id)
    if req.session_id not in eng.sessions:
        print(f"[SessionLookup] miss sid={req.session_id} chat_history_len={len(req.chat_history or [])}")
        _rehydrate_session_from_request(eng, req)
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        print(f"[ClientAPI] api_end_session_start session_id={req.session_id}")
        report = eng.end_session(req.session_id)
        print(
            "[ClientAPI] api_end_session_success "
            f"session_id={req.session_id} "
            f"scene={report.get('scene_name')} "
            f"score_keys={sorted((report.get('scores') or {}).keys())} "
            f"npc_os_count={len(report.get('npc_os_list') or [])}"
        )
        _delete_session_snapshot(req.session_id)
        return {"success": True, "data": report}
    except Exception as e:
        print(f"[ClientAPI] api_end_session_failed session_id={req.session_id} error={e}")
        logger.exception("api_end_session_failed session_id=%s", req.session_id)
        return {"success": False, "error": str(e)}


@app.post("/api/client-log")
async def client_log(req: ClientLogReq):
    payload = req.payload or {}
    print(
        f"[ClientLog] level={req.level} message={req.message} payload={payload}"
    )
    return {"success": True}


@app.post("/api/multimodal/analyze")
async def mm_analyze(req: MMReq):
    try:
        analyzer = get_mm_analyzer()
        result = analyzer.analyze_multimodal(
            req.text, req.emotion_features, req.voice_features
        )
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


if MULTIPART_AVAILABLE:
    @app.post("/api/stt")
    async def stt(file: UploadFile = File(...)):
        req_t0 = time.perf_counter()
        try:
            audio_bytes = await file.read()
            service = get_stt_service()
            result = service.transcribe(audio_bytes)
            stt_latency_ms = int(getattr(service, "last_latency_ms", 0) or 0)

            mm_result = {}
            mm_latency_ms = 0
            try:
                analyzer = get_mm_analyzer()
                mm_t0 = time.perf_counter()
                mm_result = analyzer.analyze_multimodal(
                    text=result.get("text", ""),
                    emotion_features=None,
                    voice_features=result.get("voice_features"),
                ) or {}
                mm_latency_ms = int((time.perf_counter() - mm_t0) * 1000)
            except Exception as mm_err:
                # STT text should still be usable even when multimodal dependencies are unavailable.
                print(f"[STT] multimodal analyze skipped: {mm_err}")
            return {
                "success": True,
                "data": {
                    "text": result.get("text", ""),
                    "voice_features": result.get("voice_features", {}),
                    "emotion_state": mm_result.get("emotion_state"),
                    "behavior_cues": mm_result.get("behavior_cues"),
                },
                "meta": {
                    "latency_ms": int((time.perf_counter() - req_t0) * 1000),
                    "stt_latency_ms": stt_latency_ms,
                    "mm_latency_ms": mm_latency_ms,
                    "stage": "stt",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
else:
    @app.post("/api/stt")
    async def stt_unavailable():
        return {
            "success": False,
            "error": "STT is unavailable: python-multipart is not installed in this deployment.",
        }


@app.post("/api/tts")
async def tts(req: TTSReq):
    req_t0 = time.perf_counter()
    try:
        service = get_tts_service()
        cacheable = _is_fixed_tts_phrase(req.text)
        cache_hit = False
        if cacheable:
            fixed_name = _fixed_mp3_filename(req, service)
            if (AUDIO_OUTPUT_DIR / fixed_name).exists():
                cache_hit = True
        url = _tts_url_from_req(req, service)
        if not url:
            print(
                "[TTSAPI] failed "
                f"speaker={req.speaker or ''} cacheable={cacheable} "
                f"text_len={len(req.text or '')}"
            )
            return {"success": False, "error": "TTS failed"}
        total_ms = int((time.perf_counter() - req_t0) * 1000)
        print(
            "[TTSAPI] ok "
            f"speaker={req.speaker or ''} cacheable={cacheable} cache_hit={cache_hit} "
            f"text_len={len(req.text or '')} url={url} "
            f"total_ms={total_ms} remote_ms={int(getattr(service, 'last_latency_ms', 0) or 0)}"
        )

        return {
            "success": True,
            "data": {"url": url, "cache_hit": cache_hit},
            "meta": {
                "latency_ms": total_ms,
                "remote_latency_ms": int(getattr(service, "last_latency_ms", 0) or 0),
                "stage": "tts",
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/tts/voices")
async def tts_voices():
    return {
        "success": True,
        "data": {
            "default_model": SILICONFLOW_CONFIG.tts_model_default,
            "default_voice": SILICONFLOW_CONFIG.tts_voice_default,
            "official_role_aliases": _moss_voice_aliases(),
            "official_voices": SILICONFLOW_CONFIG.tts_official_voices,
            "preset_role_voice_map": SILICONFLOW_CONFIG.preset_role_voice_map,
            "demographic_voice_map": SILICONFLOW_CONFIG.tts_role_voice_map,
            "emotion_voice_map": SILICONFLOW_CONFIG.tts_emotion_voice_map,
        },
    }


@app.post("/api/share/report-image")
async def create_share_report_image(req: ShareReportImageReq):
    try:
        saved = _save_share_report_image(req.image_data)
        return {"success": True, "data": saved}
    except HTTPException as e:
        return {"success": False, "error": str(e.detail)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/share/report-image/{filename}")
async def get_share_report_image(filename: str):
    safe = str(filename or "").strip()
    if not safe.endswith(".png") or "/" in safe or "\\" in safe:
        raise HTTPException(status_code=404, detail="not found")
    path = SHARE_REPORT_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png")


@app.get("/share/report/{token}", response_class=HTMLResponse)
async def view_share_report(token: str):
    t = str(token or "").strip()
    filename = SHARE_REPORT_INDEX.get(t, "")
    if not filename:
        filename = f"{t}.png"
    path = SHARE_REPORT_DIR / filename
    if not path.exists():
        return HTMLResponse("<h2>报告不存在或已过期</h2>", status_code=404)
    image_url = f"/api/share/report-image/{filename}"
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>TalkArena 报告分享</title>
  <style>
    body{{margin:0;padding:24px;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;justify-content:center;}}
    .card{{max-width:960px;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(15,23,42,.08);}}
    h1{{margin:0 0 12px;font-size:22px;color:#0f172a;}}
    img{{width:100%;height:auto;border-radius:12px;border:1px solid #e2e8f0;}}
  </style>
</head>
<body>
  <div class="card">
    <h1>TalkArena 复盘报告</h1>
    <img src="{image_url}" alt="report image" />
  </div>
</body>
</html>
"""
    )


@app.get("/api/knowledge/search")
async def search_knowledge(query: str):
    """RAG知识库搜索"""
    from core.rag.knowledge_base import ShandongDinnerKnowledgeBase

    kb = ShandongDinnerKnowledgeBase()
    entries = kb.retrieve(query, top_k=5)
    return {
        "success": True,
        "data": [
            {
                "title": e.title,
                "category": e.category,
                "content": e.content,
                "score": e.relevance_score,
            }
            for e in entries
        ],
    }


@app.post("/api/scenario/generate")
async def generate_scenario(req: ScenarioGenerateReq):
    """AI生成场景和成员信息"""
    try:
        # 获取LLM实例
        eng = get_engine()
        llm = eng.llm
        
        # 根据场景类型生成不同的prompt
        if req.scene_type in ("shandong_dinner", "business_dinner"):
            if req.only_characters:
                prompt = f"""
请为一场山东饭桌场景生成3个饭桌成员的详细信息，每个成员包括：
- 姓名
- 角色（如：长辈、晚辈、同事等）
- 性格特点
- 背景故事
- 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合山东酒桌文化特点，角色设定合理，背景故事生动。
同时，请为用户指定一个身份，用户身份应符合年轻人群体，例如：晚辈、年轻人、刚工作的新人等。

请以JSON格式输出，包含以下字段：
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
- user_identity: 用户身份信息，包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
"""
            else:
                prompt = f"""
请为一场山东饭桌场景生成以下内容：
1. 详细的场景背景描述（2-3句话），包括时间、地点、目的和氛围
2. 3个饭桌成员的详细信息，每个成员包括：
   - 姓名
   - 角色（如：长辈、晚辈、同事等）
   - 性格特点
   - 背景故事
   - 适合的emoji头像
3. 用户身份信息，用户身份应符合年轻人群体，例如：晚辈、年轻人、刚工作的新人等

当前场景名称：{req.scene_name}
请确保生成的内容符合山东酒桌文化特点，角色设定合理，背景故事生动。

请以JSON格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
- user_identity: 用户身份信息，包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
"""
        elif req.scene_type == "interview":
            if req.only_characters:
                prompt = f"""
请为一场群面竞争场景生成2-3个面试竞争者的详细信息，每个角色包括：
- 姓名
- 角色（都是面试竞争者）
- 性格特点
- 背景故事
- 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合群面竞争场景，角色设定鲜明，是与用户同台竞争的应聘者。

请以JSON格式输出，包含以下字段：
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
"""
            else:
                prompt = f"""
请为一场群面竞争场景生成以下内容：
1. 详细的场景背景描述（2-3句话），包括公司类型、面试岗位、群面形式
2. 2-3个面试竞争者的详细信息，每个角色包括：
   - 姓名
   - 角色（都是面试竞争者）
   - 性格特点
   - 背景故事
   - 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合群面竞争场景，角色设定鲜明，是与用户同台竞争的应聘者。

请以JSON格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
"""
        elif req.scene_type == "debate":
            if req.only_characters:
                prompt = f"""
请为一场辩论场景生成3个辩论相关角色的详细信息，每个角色包括：
- 姓名
- 角色（如：正方辩手、反方辩手、主持人等）
- 性格特点
- 背景故事
- 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合辩论场景特点，角色设定鲜明，背景故事合理。

请以JSON格式输出，包含以下字段：
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
"""
            else:
                prompt = f"""
请为一场辩论场景生成以下内容：
1. 详细的场景背景描述（2-3句话），包括辩论主题、辩论形式、参与人员
2. 3个辩论相关角色的详细信息，每个角色包括：
   - 姓名
   - 角色（如：正方辩手、反方辩手、主持人等）
   - 性格特点
   - 背景故事
   - 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合辩论场景特点，角色设定鲜明，背景故事合理。

请以JSON格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段、visualTraits字段（DiceBear捏脸参数，包含style与options）
"""
        else:
            return {"success": False, "error": "不支持的场景类型"}
        
        prompt += """

额外硬性要求：
- 每个 characters 成员必须新增字段 `tts_role`（TTS 角色槽位，按“性别 -> 年龄 -> 身份”推导）。
- 建议同时提供 `gender`、`age_group`、`identity` 三个字段以支持稳定选声。
- `tts_role` 只能在官方可选中取值：alex/anna/bella/benjamin/charles/claire/david/diana。
- 可选同时提供 `tts_voice`（如不提供由系统按 tts_role 自动映射）。
- 输出中不要遗漏原有字段。
"""

        # 调用LLM生成内容
        response = llm.generate(prompt, max_new_tokens=1500, temperature=0.8)
        
        # 尝试解析JSON响应
        import json
        try:
            # 清理响应，只保留JSON部分
            # 查找JSON的开始和结束位置
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # 确保返回的数据结构正确
                if req.only_characters:
                    # 只需要characters字段和可选的user_identity字段
                    if "characters" in result:
                        response_data = {"characters": _normalize_characters_tts_fields(result["characters"])}
                        if "user_identity" in result:
                            response_data["user_identity"] = result["user_identity"]
                        return {
                            "success": True,
                            "data": response_data
                        }
                    else:
                        return {"success": False, "error": "生成的内容格式不正确，缺少characters字段"}
                else:
                    # 需要description、characters和可选的user_identity字段
                    if "description" in result and "characters" in result:
                        response_data = {
                            "description": result["description"],
                            "characters": _normalize_characters_tts_fields(result["characters"])
                        }
                        if "user_identity" in result:
                            response_data["user_identity"] = result["user_identity"]
                        return {
                            "success": True,
                            "data": response_data
                        }
                    else:
                        return {"success": False, "error": "生成的内容格式不正确"}
            else:
                return {"success": False, "error": "无法找到JSON内容"}
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response[:500]}...")
            return {"success": False, "error": "无法解析生成的内容"}
        except Exception as e:
            print(f"解析过程中出现错误: {e}")
            return {"success": False, "error": "解析过程中出现错误"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/scenario/regenerate")
async def regenerate_scenario(req: ScenarioGenerateReq):
    """重新生成场景设定（覆盖当前编辑内容）"""
    try:
        # 获取 LLM 实例
        eng = get_engine()
        llm = eng.llm
        
        # 使用与 generate_scenario 相同的逻辑，但添加随机性确保不同结果
        if req.scene_type in ("shandong_dinner", "business_dinner"):
            # 根据酒局等级构建不同的prompt
            banquet_level_context = ""
            if req.scene_type == "business_dinner" and req.banquet_level:
                if req.banquet_level == "formal":
                    banquet_level_context = """
酒局等级：正式商务宴请
特点：高端酒店包间，精心布置的餐桌，双方高层悉数到场。这是礼仪性资源展示，信任建立的前置仪式。着装正式，举止得体，言谈谨慎而有分量。酒过三巡后才逐渐进入正题，重点在建立关系、展示诚意，为后续合作铺路。
"""
                elif req.banquet_level == "informal":
                    banquet_level_context = """
酒局等级：非正式摸底
特点：装修雅致的私房菜餐厅，氛围相对轻松。双方试探，话里有话。看似随意的闲聊中暗藏机锋，每一个话题都可能是在打探底线。不需要过于正式，但要时刻保持警觉，听懂弦外之音，同时巧妙地传递自己的立场。
"""
                elif req.banquet_level == "truth":
                    banquet_level_context = """
酒局等级：酒后吐真言
特点：酒过数巡，氛围变得热烈而直接。高压下的情感博弈，测试忠诚度。酒精卸下了部分伪装，话语开始变得尖锐和真实。这是考验彼此信任和底线的时刻，需要在保持清醒的同时，应对各种情感和利益的考验。
"""
                elif req.banquet_level == "street":
                    banquet_level_context = """
酒局等级：深夜大排档
特点：霓虹灯闪烁的街头，塑料板凳，冰镇啤酒。卸下伪装，进行最后的利益交换。没有了办公室的繁文缛节，大家都露出了最真实的一面。这是敲定最终细节的时刻，直接、务实、不绕弯子，但也要守住自己的核心利益。
"""
            
            if req.scene_type == "business_dinner" and req.banquet_level:
                prompt = f"""
请为一场商务饭局谈判场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括时间、地点、目的和氛围
2. 3 个饭桌成员的详细信息，每个成员包括：
   - 姓名
   - 角色（如：甲方负责人、乙方商务、风险顾问等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像
3. 用户身份信息，用户身份应符合商务场景，例如：部门新人、项目经理、商务代表等

{banquet_level_context}

当前场景名称：{req.scene_name}
请确保生成的内容完全符合上述酒局等级的特点，角色设定合理，背景故事生动。
注意：请生成与之前不同的场景设定，包括不同的场景描述和角色配置。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段、visualTraits 字段（DiceBear捏脸参数，包含 style 与 options）
- user_identity: 用户身份信息，包含 name、role、personality、background、avatar 字段、visualTraits 字段（DiceBear捏脸参数，包含 style 与 options）
"""
            else:
                prompt = f"""
请为一场山东饭桌场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括时间、地点、目的和氛围
2. 3 个饭桌成员的详细信息，每个成员包括：
   - 姓名
   - 角色（如：长辈、晚辈、同事等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像
3. 用户身份信息，用户身份应符合年轻人群体，例如：晚辈、年轻人、刚工作的新人等

当前场景名称：{req.scene_name}
请确保生成的内容符合山东酒桌文化特点，角色设定合理，背景故事生动。
注意：请生成与之前不同的场景设定，包括不同的场景描述和角色配置。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段、visualTraits 字段（DiceBear捏脸参数，包含 style 与 options）
- user_identity: 用户身份信息，包含 name、role、personality、background、avatar 字段、visualTraits 字段（DiceBear捏脸参数，包含 style 与 options）
"""
        elif req.scene_type == "interview":
            prompt = f"""
请为一场面试场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括公司类型、面试岗位、面试目的
2. 2-3 个面试相关角色的详细信息，每个角色包括：
   - 姓名
   - 角色（如：面试官、HR、竞争者等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像

当前场景名称：{req.scene_name}
请确保生成的内容符合职场面试场景，角色设定专业，背景故事合理。
注意：请生成与之前不同的场景设定。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段、visualTraits 字段（DiceBear捏脸参数，包含 style 与 options）
"""
        elif req.scene_type == "debate":
            prompt = f"""
请为一场辩论场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括辩论主题、辩论形式、参与人员
2. 3 个辩论相关角色的详细信息，每个角色包括：
   - 姓名
   - 角色（如：正方辩手、反方辩手、主持人等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像

当前场景名称：{req.scene_name}
请确保生成的内容符合辩论场景特点，角色设定鲜明，背景故事合理。
注意：请生成与之前不同的场景设定。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段、visualTraits 字段（DiceBear捏脸参数，包含 style 与 options）
"""
        else:
            return {"success": False, "error": "不支持的场景类型"}
        
        prompt += """

额外硬性要求：
- 每个 characters 成员必须新增字段 `tts_role`（TTS 角色槽位，按“性别 -> 年龄 -> 身份”推导）。
- 建议同时提供 `gender`、`age_group`、`identity` 三个字段以支持稳定选声。
- `tts_role` 只能在官方可选中取值：alex/anna/bella/benjamin/charles/claire/david/diana。
- 可选同时提供 `tts_voice`（如不提供由系统按 tts_role 自动映射）。
- 输出中不要遗漏原有字段。
"""

        # 调用 LLM 生成内容，使用更高的 temperature 增加多样性
        response = llm.generate(prompt, max_new_tokens=1500, temperature=0.9)
        
        # 解析 JSON 响应
        import json
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                if "description" in result and "characters" in result:
                    normalized_chars = _normalize_characters_tts_fields(result["characters"])
                    response_data = {
                        "description": result["description"],
                        "characters": normalized_chars
                    }
                    if "user_identity" in result:
                        response_data["user_identity"] = result["user_identity"]
                    return {
                        "success": True,
                        "data": response_data
                    }
                else:
                    return {"success": False, "error": "生成的内容格式不正确"}
            else:
                return {"success": False, "error": "无法找到 JSON 内容"}
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误：{e}")
            return {"success": False, "error": "无法解析生成的内容"}
        except Exception as e:
            print(f"解析过程中出现错误：{e}")
            return {"success": False, "error": "解析过程中出现错误"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/interview/generate_question")
async def generate_interview_question(req: InterviewQuestionReq):
    """根据行业和岗位生成专业的群面问题"""
    try:
        eng = get_engine()
        llm = eng.llm
        
        prompt = f"""请为{req.industry}行业的{req.position}岗位设计一个专业的群面面试问题。

要求：
1. 问题应该真实、专业，符合该行业和岗位的实际工作场景
2. 问题应该具有一定的挑战性，能够考察候选人的综合能力
3. 问题应该适合群面环境，让多个候选人可以同时回答并展开讨论
4. 问题长度控制在100-200字之间
5. 不要包含任何格式说明或额外解释，只输出问题本身

请直接输出面试问题："""
        
        response = llm.generate(prompt, max_new_tokens=300, temperature=0.8)
        
        question = response.strip()
        if not question:
            question = f"请分享一个你在{req.industry}行业做{req.position}相关工作的经历，说明你遇到的最大挑战和解决方案。"
        
        return {"success": True, "data": {"question": question}}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/content/optimize")
async def optimize_content(req: ContentOptimizeReq):
    """AI优化场景描述或面试问题"""
    try:
        eng = get_engine()
        llm = eng.llm
        
        if req.scene_type == "family":
            prompt = f"""请优化以下家庭饭桌场景的背景描述：

{req.content}

要求：
1. 保持原有的核心信息和场景设定
2. 让描述更加生动、具体、有画面感
3. 增加一些细节，让场景更加真实可信
4. 语言要自然流畅，符合中文表达习惯
5. 优化后的内容长度在150-300字之间
6. 只输出优化后的内容，不要包含任何解释或格式说明"""
        elif req.scene_type == "business":
            prompt = f"""请优化以下商务饭局场景的背景描述：

{req.content}

要求：
1. 保持原有的核心信息和场景设定
2. 让描述更加专业、商务、有氛围感
3. 增加一些商务细节，让场景更加真实可信
4. 语言要正式得体，符合商务场合
5. 优化后的内容长度在150-300字之间
6. 只输出优化后的内容，不要包含任何解释或格式说明"""
        elif req.scene_type == "interview":
            prompt = f"""请优化以下群面面试问题：

{req.content}

要求：
1. 保持原有的核心问题和考察点
2. 让问题更加清晰、专业、有挑战性
3. 问题应该适合群面环境，让多个候选人可以同时回答
4. 语言要专业、准确、有条理
5. 优化后的问题长度在100-250字之间
6. 只输出优化后的问题，不要包含任何解释或格式说明"""
        else:
            prompt = f"""请优化以下内容：

{req.content}

要求：
1. 保持原有的核心信息
2. 让描述更加生动、具体
3. 语言要自然流畅
4. 只输出优化后的内容"""
        
        response = llm.generate(prompt, max_new_tokens=400, temperature=0.7)
        
        optimized_content = response.strip()
        if not optimized_content:
            optimized_content = req.content
        
        return {"success": True, "data": {"optimized_content": optimized_content}}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/scenarios/list")
async def list_scenarios():
    """获取可用场景列表（精简为4个高质量场景）"""
    scenarios = [
        {
            "id": "shandong_dinner",
            "name": "家庭饭桌试炼",
            "category": "dinner",
            "description": "春节家宴中的高压社交，练边界与分寸",
            "icon": "🍜",
            "sub_scenes": ["家庭饭桌试炼"],
        },
        {
            "id": "business_dinner",
            "name": "商务饭局谈判",
            "category": "dinner",
            "description": "在敬酒和寒暄里推进合作，不失礼也不失守",
            "icon": "🤝",
            "sub_scenes": ["商务饭局谈判"],
        },
        {
            "id": "interview",
            "name": "群面竞争场",
            "category": "interview",
            "description": "同台竞技、展现优势、团队合作",
            "icon": "💼",
            "sub_scenes": ["群面竞争场"],
        },
        {
            "id": "debate",
            "name": "立场攻防辩论",
            "category": "debate",
            "description": "定义清晰、证据对撞、精准反驳",
            "icon": "⚔️",
            "sub_scenes": ["立场攻防辩论"],
        },
    ]
    return {"success": True, "data": scenarios}


try:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/audio", StaticFiles(directory=str(AUDIO_OUTPUT_DIR)), name="audio")
except Exception as e:
    print(f"[Audio] static mount skipped: {e}")
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")


def _load_html_template() -> str:
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


HTML_TEMPLATE = _load_html_template()


if __name__ == "__main__":
    import uvicorn

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    if "formatters" in log_config:
        if "default" in log_config["formatters"]:
            log_config["formatters"]["default"]["fmt"] = "%(asctime)s | %(levelprefix)s %(message)s"
            log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        if "access" in log_config["formatters"]:
            log_config["formatters"]["access"]["fmt"] = "%(asctime)s | %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
            log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    uvicorn.run(app, host="127.0.0.1", port=7860, log_config=log_config)


