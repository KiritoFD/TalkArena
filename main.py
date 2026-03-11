"""
TalkArena FastAPI 服务端
整合 Multi-Agent、RAG、决策引擎、防幻觉机制
"""

import sys
import os
import importlib.util
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import base64
from pydantic import BaseModel
from typing import List, Optional, Dict

app = FastAPI(title="TalkArena")
MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None

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
        self.sessions[session_id] = {
            "scenario": {"characters": characters or [{"name": "主持人", "role": "引导者"}]},
            "turn": 0,
            "scene_name": scene_name,
        }
        return session_id

    def process_turn(self, session_id: str, message: str, multimodal: Dict):
        session = self.sessions[session_id]
        session["turn"] += 1
        turn = session["turn"]
        chars = session["scenario"].get("characters", [])
        speaker = chars[turn % len(chars)].get("name", "主持人") if chars else "主持人"

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
            "ai_text": f"{speaker}：我听到你说“{message}”。请继续展开一个具体案例。",
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
        yield _MockResult("complete", payload)

    def get_rescue_suggestion(self, session_id: str):
        return "救场建议：先复述问题，再按STAR（情境-任务-行动-结果）给出30秒结构化回答。"

    def end_session(self, session_id: str):
        return {
            "scene_name": self.sessions.get(session_id, {}).get("scene_name", "模拟对话"),
            "medal": "🥇",
            "scores": {"emotional": 82, "reaction": 79, "total": 81},
            "summary": "你在多NPC环境中维持了稳定表达，能在追问下保持结构。",
            "suggestion": "下一轮提升点：减少重复句，增加结果数字与复盘反思。",
        }


def get_engine():
    global engine
    if engine is None:
        try:
            from model_loader import LLMLoader
            from core.engine import TalkArenaEngine

            llm = LLMLoader()
            llm.load()
            engine = TalkArenaEngine(llm, enable_tts=True, use_unified_agent=True)
        except Exception as e:
            print(f"[Engine] 使用回退引擎，原因: {e}")
            engine = FallbackTalkArenaEngine()
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
        from core.stt_local import LocalSTTService

        stt_service = LocalSTTService()
    return stt_service


def get_tts_service():
    global tts_service
    if tts_service is None:
        from core.tts_local import LocalTTSService

        tts_service = LocalTTSService()
    return tts_service


class ChatReq(BaseModel):
    session_id: str
    message: str = ""
    chat_history: Optional[List[Dict]] = []
    multimodal: Optional[Dict] = None


class SessionReq(BaseModel):
    scenario_id: str = "shandong_dinner"
    scene_name: str = "家庭饭桌试炼"
    characters: Optional[List[Dict]] = []
    scene_description: Optional[str] = ""
    user_info: Optional[Dict] = None


class MMReq(BaseModel):
    text: str
    emotion_features: Optional[Dict] = None
    voice_features: Optional[Dict] = None


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
    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}

    try:
        session_id = eng.start_session(
            scenario_id=req.scenario_id,
            characters=req.characters or [],
            scene_name=req.scene_name,
            scene_description=req.scene_description,
            user_info=req.user_info,
        )

        if hasattr(eng, 'use_unified_agent') and eng.use_unified_agent:
            for result in eng.process_turn(session_id, "", is_interrupt=False):
                if result.stage == "complete":
                    return {
                        "success": True,
                        "data": {
                            "session_id": session_id,
                            "is_unified_agent": True,
                            "utterances": result.data.get("utterances", []),
                            "should_await_user": result.data.get("should_await_user", True),
                        },
                    }
        else:
            session = eng.sessions[session_id]
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

            return {
                "success": True,
                "data": {
                    "session_id": session_id,
                    "opening": opening.content if opening else "",
                    "opening_speaker": opening.metadata.get("speaker") if opening else "",
                    "user_dominance": 50,
                    "ai_dominance": 50,
                    "features": {"multi_agent": True, "rag": True, "decision_engine": True},
                    "is_unified_agent": False,
                },
            }

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chat/send")
async def send_msg(req: ChatReq):
    if not req.session_id:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        multimodal = req.multimodal or {}
        mm_result = None
        if multimodal:
            try:
                mm_result = get_mm_analyzer().process_turn(req.message, multimodal)
            except Exception:
                mm_result = None
        print(f"[API] 收到多模态数据: {multimodal}")
        for result in eng.process_turn(req.session_id, req.message, multimodal, is_interrupt=False):
            if result.stage == "complete":
                payload = result.data or {}
                if mm_result:
                    payload["multimodal_analysis"] = mm_result
                return {"success": True, "data": payload}

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chat/interrupt")
async def interrupt_chat(req: InterruptReq):
    if not req.session_id:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        for result in eng.process_turn(req.session_id, "", is_interrupt=True):
            if result.stage == "complete":
                return {"success": True, "data": result.data}

        return {"success": False, "error": "处理失败"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chat/continue")
async def continue_chat(req: InterruptReq):
    if not req.session_id:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        for result in eng.process_turn(req.session_id, "", is_interrupt=False):
            if result.stage == "complete":
                return {"success": True, "data": result.data}

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
        return {"success": False, "error": "会话不存在"}

    try:
        suggestion = eng.get_rescue_suggestion(req.session_id)
        return {"success": True, "data": {"suggestion": suggestion}}
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
        return {"success": False, "error": "会话不存在"}

    try:
        report = eng.end_session(req.session_id)
        return {"success": True, "data": report}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
        try:
            audio_bytes = await file.read()
            service = get_stt_service()
            result = service.transcribe(audio_bytes)
            analyzer = get_mm_analyzer()
            mm_result = analyzer.analyze_multimodal(
                text=result.get("text", ""),
                emotion_features=None,
                voice_features=result.get("voice_features"),
            )
            return {
                "success": True,
                "data": {
                    "text": result.get("text", ""),
                    "voice_features": result.get("voice_features", {}),
                    "emotion_state": mm_result.get("emotion_state"),
                    "behavior_cues": mm_result.get("behavior_cues"),
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
    try:
        service = get_tts_service()
        audio_bytes = service.synthesize(req.text, req.emotion or "neutral")
        if not audio_bytes:
            return {"success": False, "error": "TTS failed"}
        filename = f"tts_{int(__import__('time').time() * 1000)}.wav"
        AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = AUDIO_OUTPUT_DIR / filename
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return {"success": True, "data": {"url": f"/audio/{filename}"}}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
                        response_data = {"characters": result["characters"]}
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
                            "characters": result["characters"]
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
                    response_data = {
                        "description": result["description"],
                        "characters": result["characters"]
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


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TalkArena - 酒桌情商训练平台</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#E6F0FF;min-height:100vh}
.page{display:none;width:100%;min-height:100vh}
.page.active{display:flex;flex-direction:column}

#p1{justify-content:center;align-items:center;padding:20px}
.hero{background:#fff;border:4px solid #C8102E;border-radius:20px;padding:40px 60px;max-width:700px;box-shadow:0 10px 40px rgba(200,16,46,.2);text-align:center}
.logo{font-size:48px;margin-bottom:10px}
.title{color:#C8102E;font-size:36px;font-weight:900;letter-spacing:3px}
.sub{color:#8B0000;font-size:16px;margin:10px 0 25px}

.tech-badges{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:20px 0}
.badge{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600}
.badge.rag{background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%)}
.badge.decision{background:linear-gradient(135deg,#ee0979 0%,#ff6a00 100%)}
.badge.validator{background:linear-gradient(135deg,#4776E6 0%,#8E54E9 100%)}

.features{text-align:left;margin:25px 0;background:#f8f9fa;padding:20px;border-radius:12px}
.fi{margin:12px 0;padding-left:15px;border-left:3px solid #C8102E;font-size:14px;line-height:1.6}
.fi b{color:#C8102E}

.btn1{background:#C8102E;color:#fff;border:none;padding:18px 60px;font-size:22px;font-weight:bold;border-radius:14px;cursor:pointer;box-shadow:0 8px 25px rgba(200,16,46,.4);transition:all .3s}
.btn1:hover{transform:translateY(-3px);box-shadow:0 12px 35px rgba(200,16,46,.5)}

#p2{padding:30px;max-width:900px;margin:0 auto}
.cfg-title{color:#C8102E;font-size:28px;font-weight:900;text-align:center}
.cfg-sub{color:#8B0000;font-size:14px;text-align:center;margin:10px 0 30px}

.section-l{font-size:17px;font-weight:bold;color:#333;margin:25px 0 15px;display:flex;align-items:center;gap:10px}
.ai-tag{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:11px;padding:4px 10px;border-radius:10px;font-weight:600}

.sg{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.sc{flex:1;min-width:110px;padding:14px;background:#fff;border:2px solid #E5E7EB;border-radius:10px;cursor:pointer;text-align:center;transition:all .2s}
.sc:hover{border-color:#C8102E;transform:translateY(-2px)}
.sc.on{border-color:#C8102E;background:#FFE6E6}

.mg{display:flex;gap:18px;margin-bottom:20px}
.mc{flex:1;padding:20px;background:#fff;border:2px solid #E5E7EB;border-radius:12px;text-align:center;transition:all .2s;min-height:200px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.mc:hover{border-color:#C8102E;transform:translateY(-2px);box-shadow:0 4px 15px rgba(0,0,0,.1)}
.ma{width:64px;height:64px;border-radius:18px;background:linear-gradient(160deg,#f4f7ff,#e6efff);display:flex;align-items:center;justify-content:center;margin-bottom:10px;overflow:hidden;box-shadow:inset 0 -6px 12px rgba(66,104,174,.12)}
.ma img{width:100%;height:100%;object-fit:cover}
.ma.avatar-emoji{font-size:40px;line-height:1}
.mn{font-weight:bold;font-size:16px;margin-bottom:8px;color:#333}
.mr{font-size:13px;color:#666;margin-top:5px}
.mc-tooltip{position:relative}
#customTooltip{position:fixed;background:#333;color:#fff;padding:12px 16px;border-radius:8px;font-size:13px;white-space:pre-wrap;max-width:400px;z-index:9999;box-shadow:0 4px 20px rgba(0,0,0,0.25);pointer-events:none;display:none;line-height:1.6}
#customTooltip.visible{display:block}


.ab{display:flex;gap:18px;justify-content:center;margin-top:35px}
.btn2{padding:12px 25px;background:#fff;border:2px solid #E5E7EB;border-radius:10px;cursor:pointer;font-size:14px;transition:all .2s}
.btn2:hover{border-color:#999}
.btn3{padding:15px 45px;background:#C8102E;color:#fff;border:none;border-radius:12px;cursor:pointer;font-size:18px;font-weight:bold;box-shadow:0 6px 20px rgba(200,16,46,.3);transition:all .3s}
.btn3:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(200,16,46,.4)}
.mb2{background:#fff;border:2px solid #E5E7EB;border-radius:8px;padding:8px 12px;cursor:pointer;font-size:14px;transition:all .2s}
.mb2:hover{border-color:#667eea}
.mb2.on{background:#667eea;color:#fff;border-color:#667eea}
select{background:#fff;border:2px solid #E5E7EB;border-radius:8px;padding:10px;font-size:14px;cursor:pointer}
select:focus{outline:none;border-color:#667eea}

#p3{background:#F8FAFC;height:100vh}
.ch{background:#fff;padding:14px 20px;border-bottom:1px solid #E2E8F0;display:flex;justify-content:space-between;align-items:center;position:relative;z-index:200}
.hl{display:flex;align-items:center;gap:25px}
.bb{padding:8px 16px;background:#4A90E2;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600}
.sd{display:flex;gap:25px;background:#f8f9fa;padding:10px 20px;border-radius:10px}
.si{text-align:center}
.sla{font-size:11px;color:#666}
.sv{font-size:22px;font-weight:bold}
.sv.u{color:#4A90E2}
.sv.a{color:#C62828}
.hr{display:flex;gap:10px}
.rb{padding:8px 16px;background:#5B6BF9;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600}
.eb{padding:8px 16px;background:#D32F2F;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600}
.ib{padding:8px 16px;background:#FF9800;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600}

.rescue-fab{position:fixed;right:30px;bottom:120px;width:70px;height:70px;border-radius:50%;background:linear-gradient(135deg,#5B6BF9,#7C8CFF);color:#fff;border:none;cursor:pointer;font-size:14px;font-weight:700;box-shadow:0 8px 25px rgba(91,107,249,.4);z-index:1000;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;transition:all .3s}
.rescue-fab:hover{transform:translateY(-3px) scale(1.05);box-shadow:0 12px 35px rgba(91,107,249,.5)}
.rescue-fab:active{transform:translateY(-1px) scale(1.02)}

.voice-toggle{padding:8px 14px;border-radius:10px;border:2px solid #e2e8f0;background:#fff;color:#0f172a;font-weight:700;cursor:pointer;transition:all .2s}
.voice-toggle.on{background:#111827;border-color:#111827;color:#fff;box-shadow:0 8px 18px rgba(17,24,39,.25)}
.voice-toggle:hover{transform:translateY(-1px)}

.profile-wrap{max-width:1100px;margin:0 auto;padding:28px}
.profile-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}
.profile-title{font-size:24px;font-weight:900;color:#111827}
.profile-sub{font-size:13px;color:#6b7280}
.profile-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:14px 0 20px}
.profile-stat{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px;box-shadow:0 8px 20px rgba(15,23,42,.06)}
.profile-stat h4{margin:0 0 8px 0;font-size:13px;color:#64748b}
.profile-stat .stat-val{font-size:22px;font-weight:900;color:#111827}
.profile-tabs{display:flex;gap:10px;overflow-x:auto;padding:6px 2px;margin-bottom:14px}
.profile-tab{white-space:nowrap;padding:8px 14px;border-radius:999px;border:1px solid #e2e8f0;background:#fff;color:#334155;font-weight:700;cursor:pointer}
.profile-tab.active{background:#2563eb;border-color:#2563eb;color:#fff;box-shadow:0 8px 18px rgba(37,99,235,.3)}
.report-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.report-card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px;box-shadow:0 8px 18px rgba(15,23,42,.06);display:flex;flex-direction:column;gap:10px}
.report-card h5{margin:0;font-size:15px;color:#111827}
.report-card p{margin:0;font-size:13px;line-height:1.5;color:#4b5563}
.report-card .view-btn{margin-top:auto;padding:8px 10px;border-radius:10px;border:none;background:#111827;color:#fff;font-weight:700;cursor:pointer}
.view-btn{padding:8px 12px;border-radius:10px;border:none;background:#111827;color:#fff;font-weight:700;cursor:pointer}
.dist-row{display:flex;justify-content:space-between;font-size:12px;color:#475569;padding:4px 0;border-bottom:1px dashed #e5e7eb}
.dist-row:last-child{border-bottom:none}
.report-empty{background:#f8fafc;border:1px dashed #cbd5f5;border-radius:14px;padding:22px;color:#6b7280;text-align:center}

.interrupt-fab{position:fixed;right:30px;bottom:210px;width:70px;height:70px;border-radius:50%;background:linear-gradient(135deg,#FF9800,#FFB74D);color:#fff;border:none;cursor:pointer;font-size:14px;font-weight:700;box-shadow:0 8px 25px rgba(255,152,0,.4);z-index:1000;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;transition:all .3s}
.interrupt-fab:hover{transform:translateY(-3px) scale(1.05);box-shadow:0 12px 35px rgba(255,152,0,.5)}
.interrupt-fab:active{transform:translateY(-1px) scale(1.02)}

.cm{flex:1;display:flex;overflow:hidden}

.sp{width:200px;background:linear-gradient(180deg,#E6F0FF 0%,#FFF 100%);border-right:1px solid #E2E8F0;padding:18px;display:flex;flex-direction:column}
.st{font-size:14px;color:#666;margin-bottom:18px;text-align:center;font-weight:600}
.ci{display:flex;align-items:center;gap:12px;padding:12px;background:#fff;border-radius:10px;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,.05);transition:all .2s;position:relative;overflow:hidden}
.ci.talk{border:2px solid #C8102E;box-shadow:0 4px 15px rgba(200,16,46,.2)}
.ci::after{content:'';position:absolute;inset:auto -40% -60% -40%;height:60%;background:radial-gradient(circle at center,rgba(74,144,226,.08),transparent 70%);pointer-events:none;opacity:0;transition:opacity .2s}
.ci.talk::after{opacity:1}
.ca{font-size:18px;line-height:1}
.cn{font-weight:bold;font-size:14px;color:#333}
.head{width:44px;height:44px;border-radius:14px;background:linear-gradient(145deg,#fff,#f4f7ff);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 -4px 8px rgba(74,144,226,.12),0 4px 10px rgba(0,0,0,.08);position:relative;flex-shrink:0;transition:transform .2s ease}
.head-face{width:36px;height:36px;position:relative}
.eyes{position:absolute;top:10px;left:6px;right:6px;display:flex;justify-content:space-between}
.eye{width:7px;height:8px;border-radius:50%;background:#222;transition:transform .08s,height .08s}
.mouth{position:absolute;left:50%;bottom:6px;transform:translateX(-50%);width:14px;height:4px;border-radius:8px;background:#b35f5f;transition:width .08s,height .08s,border-radius .08s,background .12s}
.ci.state-speaking .head{transform:translateY(-1px) scale(1.03)}
.ci.state-speaking .mouth{width:16px;height:10px;border-radius:8px;background:#c44b4b;animation:talkMouth .12s infinite alternate}
.ci.state-reacting .head{animation:nod 1.6s ease-in-out infinite}
.ci.state-listening .mouth{background:#8a6f6f;width:12px}
.ci.state-idle .head{filter:saturate(.9)}
.ci.blink .eye{height:2px;transform:translateY(3px)}
.ci.look-user .head-face{transform:translateX(-1px)}
.ci.look-speaker .head-face{transform:translateX(1px)}
.ci .backchannel{position:absolute;top:4px;right:8px;background:#eef5ff;color:#4A90E2;border:1px solid #dbe9ff;padding:1px 6px;border-radius:10px;font-size:10px;opacity:0;transform:translateY(-4px);transition:all .18s}
.ci.has-backchannel .backchannel{opacity:1;transform:translateY(0)}
@keyframes talkMouth{from{height:6px;width:12px}to{height:11px;width:18px}}
@keyframes nod{0%,100%{transform:translateY(0)}50%{transform:translateY(1.5px)}}

.cc{flex:1;display:flex;flex-direction:column;padding:18px;overflow:hidden}
.mc2{flex:1;overflow-y:auto;padding:12px;background:#fff;border-radius:12px;border:1px solid #E2E8F0;margin-bottom:12px}
.msg{max-width:75%;margin:10px 0;padding:14px 18px;border-radius:12px;animation:fadeIn .3s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.msg.u{margin-left:auto;background:#E3F2FD;border-left:4px solid #2196F3}
.msg.b{background:linear-gradient(135deg,#FFF9F0 0%,#FFEFD5 100%);border-left:4px solid #F5A623}
.msg-emo{margin-left:8px;font-size:18px;animation:pulse 0.5s}
@keyframes pulse{0%{transform:scale(0.8)}50%{transform:scale(1.2)}100%{transform:scale(1)}}
.ca{transition:transform 0.3s}
.ms{font-weight:bold;color:#D48806;font-size:15px;margin-bottom:6px}
.mco{line-height:1.6;color:#333;font-size:14px}

.cb{background:linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);border-radius:10px;padding:14px 18px;margin:12px 0;border-left:4px solid #667eea;display:flex;align-items:center;gap:10px}
.cb-icon{font-size:24px}
.ct2{font-size:14px;color:#333}

.ia{background:#6495ED;border-radius:28px;padding:10px 18px;display:flex;align-items:center;gap:12px;box-shadow:0 4px 15px rgba(100,149,237,.3)}
.mb{background:transparent;border:none;font-size:22px;cursor:pointer;transition:all .2s;position:relative;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center}
.mb:hover{transform:scale(1.1);background:rgba(255,255,255,.2)}
.mb.active{background:#C8102E !important;box-shadow:0 0 15px rgba(200,16,46,.5)}

.ci2{flex:1;background:transparent;border:none;color:#fff;font-size:16px;outline:none}
.ci2::placeholder{color:rgba(255,255,255,.7)}
.sb{background:#fff;color:#6495ED;border:none;padding:8px 22px;border-radius:16px;cursor:pointer;font-weight:bold;font-size:14px;transition:all .2s}
.sb:hover{transform:scale(1.05)}

.sp{width:200px;background:linear-gradient(180deg,#E6F0FF 0%,#FFF 100%);border-right:1px solid #E2E8F0;padding:18px;display:flex;flex-direction:column}
.st{font-size:14px;color:#666;margin-bottom:18px;text-align:center;font-weight:600}
.ci{display:flex;align-items:center;gap:12px;padding:12px;background:#fff;border-radius:10px;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,.05);transition:all .2s;position:relative;overflow:hidden}
.ci.talk{border:2px solid #C8102E;box-shadow:0 4px 15px rgba(200,16,46,.2)}
.ci::after{content:'';position:absolute;inset:auto -40% -60% -40%;height:60%;background:radial-gradient(circle at center,rgba(74,144,226,.08),transparent 70%);pointer-events:none;opacity:0;transition:opacity .2s}
.ci.talk::after{opacity:1}
.ca{font-size:18px;line-height:1;transition:transform 0.3s}
.cn{font-weight:bold;font-size:14px;color:#333}
.head{width:44px;height:44px;border-radius:14px;background:linear-gradient(145deg,#fff,#f4f7ff);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 -4px 8px rgba(74,144,226,.12),0 4px 10px rgba(0,0,0,.08);position:relative;flex-shrink:0;transition:transform .2s ease}
.head-face{width:36px;height:36px;position:relative;transition:transform .16s}
.eyes{position:absolute;top:10px;left:6px;right:6px;display:flex;justify-content:space-between}
.eye{width:7px;height:8px;border-radius:50%;background:#222;transition:transform .08s,height .08s}
.mouth{position:absolute;left:50%;bottom:6px;transform:translateX(-50%);width:14px;height:4px;border-radius:8px;background:#b35f5f;transition:width .08s,height .08s,border-radius .08s,background .12s}
.ci.state-speaking .head{transform:translateY(-1px) scale(1.03)}
.ci.state-speaking .mouth{width:16px;height:10px;border-radius:8px;background:#c44b4b;animation:talkMouth .12s infinite alternate}
.ci.state-reacting .head{animation:nod 1.6s ease-in-out infinite}
.ci.state-listening .mouth{background:#8a6f6f;width:12px}
.ci.state-idle .head{filter:saturate(.9)}
.ci.blink .eye{height:2px;transform:translateY(3px)}
.ci.look-user .head-face{transform:translateX(-1px)}
.ci.look-speaker .head-face{transform:translateX(1px)}
.ci .backchannel{position:absolute;top:4px;right:8px;background:#eef5ff;color:#4A90E2;border:1px solid #dbe9ff;padding:1px 6px;border-radius:10px;font-size:10px;opacity:0;transform:translateY(-4px);transition:all .18s}
.ci.has-backchannel .backchannel{opacity:1;transform:translateY(0)}
@keyframes talkMouth{from{height:6px;width:12px}to{height:11px;width:18px}}
@keyframes nod{0%,100%{transform:translateY(0)}50%{transform:translateY(1.5px)}}

.sp-metrics{flex:1;margin-top:20px;overflow-y:auto}
.sp-metrics .mt{font-size:12px;color:#666;font-weight:600;margin-bottom:10px;text-align:center}
.sp-metric{background:#fff;border-radius:8px;padding:10px;margin-bottom:8px;border:1px solid #E2E8F0}
.sp-metric .mlabel{font-size:11px;color:#666;display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.sp-metric .mlabel span:first-child{font-weight:600}
.bar-bg{width:100%;height:8px;background:#E2E8F0;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:4px;transition:width .3s}

.mp{width:200px;background:#fff;border-left:1px solid #E2E8F0;padding:18px;display:flex;flex-direction:column;position:absolute;right:0;top:68px;bottom:0;transform:translateX(100%);transition:transform .3s ease;z-index:50}
.mp.visible{transform:translateX(0)}
.mp .mt{font-size:13px;color:#333;font-weight:bold;text-align:center;margin-bottom:12px}
.mp .cam-preview{width:100%;aspect-ratio:4/3;background:#1a1a1a;border-radius:10px;overflow:hidden;margin-bottom:12px;position:relative}
.mp .cam-preview video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1);display:none}
.mp .cam-placeholder{position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#666;font-size:11px}
.mp .cam-placeholder{width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#666;font-size:11px}
.mp select{width:100%;padding:8px;font-size:12px;border:1px solid #E5E7EB;border-radius:6px;margin-bottom:8px;background:#fff}
.mp button{width:100%;padding:10px;font-size:13px;border:1px solid #E5E7EB;border-radius:8px;background:#fff;cursor:pointer;transition:all .2s;margin-bottom:8px}
.mp button:hover{border-color:#667eea}
.mp button.on{background:#667eea;color:#fff;border-color:#667eea}
.mp .vol-bar{width:100%;height:20px;background:#E2E8F0;border-radius:4px;overflow:hidden;margin-bottom:8px;display:flex;padding:2px}
.mp .vol-fill{height:100%;background:#667eea;border-radius:2px;transition:width .1s;margin-right:2px}
.mp .vol-fill:last-child{margin-right:0}
.mp .vol-segment{flex:1;height:100%;background:#E2E8F0;border-radius:3px;margin-right:4px}
.mp .vol-segment:last-child{margin-right:0}
.mp .vol-segment.active{background:#22c55e;box-shadow:0 0 8px #22c55e}
.mp .vol-label{font-size:11px;color:#666;text-align:center}

.pressure-section-wrapper{margin-top:10px;margin-bottom:10px}
.pressure-header-row{display:flex;align-items:center;justify-content:space-between;margin-top:25px;margin-bottom:10px}
.pressure-container{display:flex;gap:20px;align-items:flex-start;margin-top:10px;margin-bottom:10px}
.pressure-tags{display:flex;gap:10px;flex-wrap:wrap;flex:1}
.pressure-tag{padding:12px 20px;background:#fff;border:2px solid #E2E8F0;border-radius:16px;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:6px;font-size:14px;font-weight:600;color:#333}
.pressure-tag:hover{border-color:#667eea;background:#f0f3ff}
.pressure-tag.selected{border-color:#667eea;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;box-shadow:0 4px 15px rgba(102,126,234,.3)}

.pressure-slider::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);cursor:pointer;box-shadow:0 2px 8px rgba(102,126,234,.4);transition:transform .2s}
.pressure-slider::-webkit-slider-thumb:hover{transform:scale(1.1)}
.pressure-slider::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);cursor:pointer;border:none;box-shadow:0 2px 8px rgba(102,126,234,.4)}

.banquet-level-wrapper{margin-top:10px;margin-bottom:10px}
.banquet-level-tags{display:flex;gap:10px;flex-wrap:wrap}
.banquet-level-tag{padding:14px 24px;background:#fff;border:2px solid #E2E8F0;border-radius:16px;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:8px;font-size:15px;font-weight:600;color:#333;position:relative}
.banquet-level-tag:hover{border-color:#667eea;background:#f0f3ff;transform:translateY(-2px);box-shadow:0 4px 12px rgba(102,126,234,.15)}
.banquet-level-tag.selected{border-color:#667eea;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;box-shadow:0 4px 15px rgba(102,126,234,.3);transform:translateY(-2px)}

.drinking-capacity-stars{display:flex;gap:8px;align-items:center}
.drinking-capacity-stars .star{font-size:40px;cursor:pointer;transition:all .15s;color:#ddd;user-select:none;line-height:1}
.drinking-capacity-stars .star:hover{transform:scale(1.15)}
.drinking-capacity-stars .star.filled{color:#ffd700;text-shadow:0 2px 8px rgba(255,215,0,.4)}

.interview-info-wrapper{margin-top:10px;margin-bottom:10px}
.interview-tags{display:flex;gap:10px;flex-wrap:wrap}
.interview-tag{padding:12px 22px;background:#fff;border:2px solid #E2E8F0;border-radius:14px;cursor:pointer;transition:all .2s;font-size:14px;font-weight:600;color:#333}
.interview-tag:hover{border-color:#667eea;background:#f0f3ff}
.interview-tag.selected{border-color:#667eea;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;box-shadow:0 4px 15px rgba(102,126,234,.3)}

.confirm-edit-btn{padding:10px 24px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:10px;cursor:pointer;font-size:14px;font-weight:600;transition:all .2s;box-shadow:0 4px 12px rgba(102,126,234,.3)}
.confirm-edit-btn:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(102,126,234,.4)}
.confirm-edit-btn:disabled{opacity:.6;cursor:not-allowed;transform:none}

.ai-optimize-btn{padding:10px 20px;background:transparent;color:#333;border:2px solid #E2E8F0;border-radius:10px;cursor:pointer;font-size:14px;font-weight:600;transition:all .2s}
.ai-optimize-btn:hover{border-color:#667eea;background:#f0f3ff;color:#667eea}
.ai-optimize-btn:disabled{opacity:.6;cursor:not-allowed}

.interview-question-box{background:#fff;border:2px solid #667eea;border-radius:16px;padding:16px;margin-bottom:16px;box-shadow:0 4px 12px rgba(102,126,234,.15)}
.interview-question-content{font-size:15px;color:#333;line-height:1.6;max-height:96px;overflow-y:auto;transition:max-height .3s ease}
.interview-question-content.collapsed{max-height:24px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.interview-question-toggle{display:flex;justify-content:center;align-items:center;margin-top:12px;cursor:pointer}
.interview-question-toggle span{display:inline-block;font-size:18px;color:#667eea;transition:transform .3s ease}
.interview-question-toggle:hover span{transform:scale(1.2)}
.interview-question-toggle.collapsed span{transform:rotate(180deg)}
.interview-question-toggle.collapsed:hover span{transform:rotate(180deg) scale(1.2)}

#p4{background:#2c313c;padding:40px;justify-content:center;align-items:center}
.rc{background:#fff;border-radius:20px;padding:40px;max-width:1400px;width:100%;box-shadow:0 20px 60px rgba(0,0,0,.3)}
.rt{text-align:center;font-size:26px;font-weight:bold;margin-bottom:20px;color:#333}
.md{text-align:center;font-size:80px;margin:20px 0}
.sg2{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:25px 0}
.sb2{background:#f8f9fa;padding:18px;border-radius:12px;text-align:center}
.sbl{font-size:12px;color:#666;margin-bottom:5px}
.sbv{font-size:28px;font-weight:bold;color:#667eea}
.rs{background:#f8f9fa;padding:20px;border-radius:12px;line-height:1.8;font-size:15px;color:#333}
.rss{background:#fff9e6;padding:15px;border-radius:12px;border-left:4px solid #F5A623;margin-top:20px;font-size:14px;color:#333}
.rb2{display:flex;gap:18px;justify-content:center;margin-top:30px}

.loading{display:flex;align-items:center;gap:15px;padding:20px}
.spinner{width:30px;height:30px;border:3px solid #e9ecef;border-top-color:#667eea;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Rebuilt NPC portrait panel: high contrast, state-driven expressions */
#p3 .sp{width:280px;background:linear-gradient(180deg,#dbe7ff 0%,#f4f8ff 60%,#fff 100%);border-right:1px solid #cddcf5;padding:18px}
#p3 .st{display:none}
#p3 #cl{display:flex;flex-direction:column;gap:12px}
#p3 .ci{display:flex;align-items:center;gap:14px;padding:14px;background:linear-gradient(160deg,#ffffff 0%,#f5f9ff 100%);border:1px solid #d9e5fb;border-radius:18px;box-shadow:0 6px 20px rgba(39,76,153,.12);position:relative;overflow:hidden}
#p3 .ci::before{content:'';position:absolute;left:0;top:0;bottom:0;width:5px;background:#b8c9ea}
#p3 .ci.state-speaking{border-color:#C8102E;box-shadow:0 10px 30px rgba(200,16,46,.35),0 0 0 3px rgba(200,16,46,.15)}
#p3 .ci.state-speaking::before{background:#C8102E;width:6px}
#p3 .ci.state-speaking .avatar-main{transform:scale(1.08);filter:drop-shadow(0 3px 5px rgba(200,16,46,.35))}
#p3 .ci.state-reacting::before{background:#5e9dff}
#p3 .ci.state-listening::before{background:#39a96b}
#p3 .ci .head{width:68px;height:68px;border-radius:20px;background:linear-gradient(160deg,#ecf2ff 0%,#dae8ff 100%);border:1px solid #c5d9ff;box-shadow:inset 0 -8px 14px rgba(66,104,174,.18),0 5px 12px rgba(15,23,42,.12);display:flex;align-items:center;justify-content:center;flex-shrink:0;position:relative}
#p3 .ci .avatar-main{width:44px;height:44px;border-radius:14px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#fff;filter:drop-shadow(0 2px 2px rgba(0,0,0,.2));transition:transform .2s ease,filter .2s ease}
#p3 .ci .avatar-main img{width:100%;height:100%;object-fit:cover}
#p3 .ci .avatar-main.avatar-emoji{font-size:36px;line-height:1}
#p3 .ci .avatar-exp{position:absolute;right:-5px;bottom:-5px;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#fff;border:1px solid #d9e6ff;box-shadow:0 3px 8px rgba(0,0,0,.15);font-size:14px}
#p3 .ci .npc-meta{display:flex;flex-direction:column;gap:2px}
#p3 .ci .cn{font-size:20px;font-weight:900;color:#252932;line-height:1}
#p3 .ci .role{font-size:16px;color:#54657f;line-height:1.2}
#p3 .ci .mood-pill{display:inline-flex;align-items:center;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;color:#2f3f5d;background:#e5eefc;border:1px solid #c8d9f8;width:fit-content;margin-top:4px}
#p3 .ci .backchannel{position:absolute;top:8px;right:10px;font-size:11px;font-weight:700;color:#2759a5;background:#e6f0ff;border:1px solid #bfd4ff;border-radius:999px;padding:2px 7px;opacity:0;transform:translateY(-4px);transition:all .16s}
#p3 .ci.has-backchannel .backchannel{opacity:1;transform:translateY(0)}
#p3 .ci.blink .avatar-main{transform:scaleY(.84)}
#p3 .ci.look-speaker .avatar-main{transform:translateX(1px)}
#p3 .ci.look-user .avatar-main{transform:translateX(-1px)}
#p3 .ci.expr-neutral .mood-pill{background:#e6ecf7;color:#43506a;border-color:#c9d4ea}
#p3 .ci.expr-calm .avatar-exp{background:#e5f4ea;border-color:#bfe1cb}
#p3 .ci.expr-calm .mood-pill{background:#e5f4ea;color:#226a43;border-color:#bfe1cb}
#p3 .ci.expr-focused .avatar-main{filter:drop-shadow(0 2px 3px rgba(29,95,183,.45))}
#p3 .ci.expr-focused .avatar-exp{background:#e5f0ff;border-color:#bfd8ff}
#p3 .ci.expr-focused .mood-pill{background:#e5f0ff;color:#1d5fb7;border-color:#bfd8ff}
#p3 .ci.expr-engaged .head{box-shadow:inset 0 -8px 14px rgba(151,93,25,.22),0 0 0 2px rgba(255,139,0,.25),0 8px 16px rgba(255,139,0,.25)}
#p3 .ci.expr-engaged .avatar-main{animation:talkBob .18s infinite alternate}
#p3 .ci.expr-engaged .avatar-exp{background:#fff2df;border-color:#ffd3a7}
#p3 .ci.expr-engaged .mood-pill{background:#fff2df;color:#a75000;border-color:#ffd3a7}
#p3 .ci.expr-warm .avatar-exp{background:#ffe8d9;border-color:#ffc7ad}
#p3 .ci.expr-warm .mood-pill{background:#ffe8d9;color:#a34521;border-color:#ffc7ad}
#p3 .ci.expr-warm .avatar-main{transform:translateY(-1px) scale(1.03)}
@keyframes talkBob{from{transform:translateY(-1px) scale(1.02)}to{transform:translateY(1px) scale(1.07)}}
</style>
</head>
<body>
<!-- 登录/注册按钮 -->
<button id="loginBtn" class="login-btn" onclick="openAuthModal()" style="position:absolute;top:20px;right:20px;padding:10px 20px;background:#C8102E;color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer;z-index:9999;">登录/注册</button>
<div id="p1" class="page active">
<div class="hero">
<div class="logo">🎯</div>
<div class="title">表达训练营</div>
<div class="sub">AI驱动的口语表达实战训练平台</div>
<div class="tech-badges">
<span class="badge">Multi-Agent协同</span>
<span class="badge rag">RAG知识增强</span>
<span class="badge decision">决策引擎</span>
<span class="badge validator">防幻觉机制</span>
</div>
<div class="features">
<div class="fi"><b>核心玩法</b> - 在山东酒桌文化的情商高压测试中生存，掌握应对技巧</div>
<div class="fi"><b>技术亮点</b> - 多Agent协同决策、知识库增强生成、智能任务规划</div>
<div class="fi"><b>训练价值</b> - 实时多模态分析、高情商回复建议、详细复盘报告</div>
<div class="fi"><b>场景丰富</b> - 4种高质量场景，覆盖饭桌、商务、面试与辩论</div>
</div>
<button class="btn1" onclick="goCfg()">开始挑战</button>
</div>
</div>

<div id="p2" class="page">
<div class="cfg-title">表达训练营</div>
<div class="cfg-sub">选择你的训练场</div>
<div class="section-l">选择场景</div>
<div class="sg" id="sg"></div>
<div class="ab" style="margin-top:20px;margin-bottom:20px;">
<button class="btn2" id="sceneGenBtn" onclick="regenerateScene()" disabled>🔄 重新生成场景设定</button>
</div>

<div class="banquet-level-wrapper" id="banquetLevelWrapper" style="display:none;">
  <div class="banquet-level-row" style="display:flex;align-items:flex-end;justify-content:space-between;gap:30px;">
    <div class="banquet-level-section" style="flex:1;">
      <div class="section-l" style="margin-top:25px;margin-bottom:10px;">酒局等级</div>
      <div class="banquet-level-tags" id="banquetLevelTags">
        <div class="banquet-level-tag mc-tooltip selected" data-level="formal" onclick="selectBanquetLevel(this)" data-tooltip="礼仪性资源展示，信任建立的前置仪式">
          🍽️ 正式商务宴请
        </div>
        <div class="banquet-level-tag mc-tooltip" data-level="informal" onclick="selectBanquetLevel(this)" data-tooltip="双方试探，话里有话">
          🤝 非正式摸底
        </div>
        <div class="banquet-level-tag mc-tooltip" data-level="truth" onclick="selectBanquetLevel(this)" data-tooltip="高压下的情感博弈，测试忠诚度">
          🍻 酒后吐真言
        </div>
        <div class="banquet-level-tag mc-tooltip" data-level="street" onclick="selectBanquetLevel(this)" data-tooltip="卸下伪装，进行最后的利益交换">
          🍜 深夜大排档
        </div>
      </div>
    </div>
    <div class="drinking-capacity-section" style="display:flex;flex-direction:column;align-items:flex-end;">
      <div class="section-l" style="margin-top:25px;margin-bottom:10px;">我的酒量</div>
      <div class="drinking-capacity-stars" id="drinkingCapacityStars">
        <span class="star" data-index="0" onclick="setDrinkingCapacity(1)">☆</span>
        <span class="star" data-index="1" onclick="setDrinkingCapacity(2)">☆</span>
        <span class="star" data-index="2" onclick="setDrinkingCapacity(3)">☆</span>
      </div>
    </div>
  </div>
</div>

<div class="interview-info-wrapper" id="interviewInfoWrapper" style="display:none;">
  <div class="section-l" style="margin-top:25px;margin-bottom:10px;">面试岗位</div>
  
  <div class="interview-row" style="display:flex;align-items:center;gap:15px;margin-bottom:15px;">
    <span style="font-size:14px;font-weight:600;color:#333;min-width:50px;">行业</span>
    <div class="interview-tags" id="industryTags">
      <div class="interview-tag selected" data-value="互联网" onclick="selectIndustry(this)">互联网</div>
      <div class="interview-tag" data-value="金融" onclick="selectIndustry(this)">金融</div>
      <div class="interview-tag" data-value="快消" onclick="selectIndustry(this)">快消</div>
      <div class="interview-tag" data-value="咨询" onclick="selectIndustry(this)">咨询</div>
      <div class="interview-tag" data-value="自定义" onclick="selectIndustry(this)">自定义</div>
    </div>
    <input type="text" id="customIndustryInput" placeholder="请输入行业" oninput="updateInterviewQuestion()" style="display:none;width:180px;padding:10px 15px;border:2px solid #667eea;border-radius:12px;font-size:14px;outline:none;">
  </div>
  
  <div class="interview-row" style="display:flex;align-items:center;gap:15px;">
    <span style="font-size:14px;font-weight:600;color:#333;min-width:50px;">岗位</span>
    <div class="interview-tags" id="positionTags">
      <div class="interview-tag selected" data-value="产品" onclick="selectPosition(this)">产品</div>
      <div class="interview-tag" data-value="销售" onclick="selectPosition(this)">销售</div>
      <div class="interview-tag" data-value="市场" onclick="selectPosition(this)">市场</div>
      <div class="interview-tag" data-value="分析" onclick="selectPosition(this)">分析</div>
      <div class="interview-tag" data-value="自定义" onclick="selectPosition(this)">自定义</div>
    </div>
    <input type="text" id="customPositionInput" placeholder="请输入岗位" oninput="updateInterviewQuestion()" style="display:none;width:180px;padding:10px 15px;border:2px solid #667eea;border-radius:12px;font-size:14px;outline:none;">
  </div>
</div>

<div class="section-l" id="sceneInfoSection" style="display:none;">
  <span id="sceneInfoTitle">背景信息</span>
  <span style="font-size:12px;color:#667eea;cursor:pointer;" onclick="toggleSceneEdit()">✏️ 编辑</span>
  <button class="btn2" id="aiCustomizeBtn" onclick="generateCustomInterviewQuestion()" style="display:none;padding:8px 16px;font-size:13px;float:right;">🤖 AI定制</button>
</div>
<div class="scene-description" id="sceneDescription" style="display:none;background:#f8f9fa;border-radius:10px;padding:15px;margin:10px 0;border-left:4px solid #667eea;">
  <div id="sceneDescriptionText" style="font-size:14px;color:#333;line-height:1.5;"></div>
  <div id="sceneEditWrapper" style="display:none;">
    <textarea id="sceneDescriptionEdit" style="width:100%;min-height:100px;border:1px solid #ddd;border-radius:5px;padding:10px;font-size:14px;color:#333;line-height:1.5;resize:vertical;"></textarea>
    <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:10px 0 0 0;">
      <button class="ai-optimize-btn" id="aiOptimizeBtn" onclick="aiOptimizeContent()">🌟 AI优化</button>
      <button class="confirm-edit-btn" id="confirmEditBtn" onclick="confirmSceneEdit()">确认</button>
    </div>
  </div>
</div>
<div class="pressure-section-wrapper" id="pressureSectionWrapper" style="display:none;">
  <div class="pressure-header-row" style="display:flex;align-items:center;justify-content:space-between;margin-top:25px;margin-bottom:10px;">
    <div class="section-l" style="margin:0;">压力敏感区 <span style="font-size:12px;color:#667eea;">可选填</span></div>
    <div class="pressure-value-box" id="pressureValueBox" style="display:flex;align-items:center;gap:10px;background:#f8f9fa;border-radius:12px;padding:8px 15px;border:1px solid #E2E8F0;">
      <span style="font-size:13px;font-weight:600;color:#333;white-space:nowrap;">压力值</span>
      <span style="font-size:12px;color:#666;font-weight:600;">0</span>
      <input type="range" id="pressureSlider" min="0" max="10" value="5" class="pressure-slider" oninput="updatePressureValue(this.value)" style="width:120px;height:6px;border-radius:3px;background:#E2E8F0;outline:none;-webkit-appearance:none;">
      <span style="font-size:12px;color:#666;font-weight:600;">10</span>
      <span id="pressureDisplay" style="font-size:22px;font-weight:bold;color:#667eea;min-width:28px;text-align:center;">5</span>
    </div>
  </div>
  <div class="pressure-container" id="pressureContainer" style="display:flex;">
    <div class="pressure-tags" id="pressureTags">
      <div class="pressure-tag" data-tag="催婚" onclick="selectPressureTag(this)">💒 催婚</div>
      <div class="pressure-tag" data-tag="催育" onclick="selectPressureTag(this)">👶 催育</div>
      <div class="pressure-tag" data-tag="工作" onclick="selectPressureTag(this)">💼 工作</div>
      <div class="pressure-tag" data-tag="学业" onclick="selectPressureTag(this)">📚 学业</div>
      <div class="pressure-tag" data-tag="自定义" onclick="selectPressureTag(this)">✏️ 自定义</div>
    </div>
  </div>
</div>
<div class="custom-pressure-input" id="customPressureInput" style="display:none;margin-top:10px;">
  <input type="text" id="customPressureText" placeholder="请输入自定义压力话题..." style="width:100%;padding:10px;border:1px solid #ddd;border-radius:8px;font-size:14px;">
</div>

<div class="section-l" id="memberSection" style="display:none;">饭局成员 <span class="ai-tag">AI智能分配</span></div>
<div class="mg" id="mg" style="display:none;"></div>
<div class="ab" id="actionButtons" style="display:none;">
<button class="btn2" onclick="randMem()">随机换人</button>
<button class="btn3" onclick="start()">入席开整</button>
</div>
</div>

<div id="p3" class="page">
<div class="ch">
<div class="hl">
<button class="bb" onclick="goBackFromGame()">返回</button>
<div class="sd">
<div class="si"><span class="sla">你的气场</span><span class="sv u" id="us">50</span></div>
<div class="si"><span class="sla">AI气场</span><span class="sv a" id="as">50</span></div>
</div>
</div>
<div class="hr">
<button class="voice-toggle" id="npcVoiceToggle" onclick="toggleNpcVoice()">NPC语音: 关</button>
<button class="eb" onclick="end()">结束</button>
</div>
</div>
<div class="cm">
<div class="sp">
<div class="st"></div>
<div id="cl"></div>
<div class="sp-metrics">
<div class="mt" style="color:#C8102E;font-weight:bold;">🎭 实时情感分析</div>
<div class="sp-metric"><div class="mlabel"><span>😎 自信度</span><span id="val-confidence">0</span></div><div class="bar-bg"><div class="bar-fill" id="bar-confidence" style="width:0%;background:#22c55e"></div></div></div>
<div class="sp-metric"><div class="mlabel"><span>😐 平静度</span><span id="val-calm">0</span></div><div class="bar-bg"><div class="bar-fill" id="bar-calm" style="width:0%;background:#3b82f6"></div></div></div>
<div class="sp-metric"><div class="mlabel"><span>😰 紧张度</span><span id="val-nervous">0</span></div><div class="bar-bg"><div class="bar-fill" id="bar-nervous" style="width:0%;background:#ef4444"></div></div></div>
<div class="sp-metric"><div class="mlabel"><span>🤔 专注度</span><span id="val-focus">0</span></div><div class="bar-bg"><div class="bar-fill" id="bar-focus" style="width:0%;background:#f59e0b"></div></div></div>
</div>
<div class="mt" style="margin-top:15px;color:#667eea;font-weight:bold;">📊 AI综合评分</div>
<div class="sp-metric" style="background:linear-gradient(135deg,#f0f3ff,#e0e7ff);border-color:#667eea">
<div class="mlabel" style="font-size:14px"><span>总分</span><span id="val-score" style="font-size:20px;font-weight:bold;color:#C8102E">--</span></div>
<div class="bar-bg" style="height:12px"><div class="bar-fill" id="bar-score" style="width:0%;background:linear-gradient(90deg,#667eea,#764ba2);height:100%"></div></div>
</div>
</div>
<div class="cc">
<div id="interviewQuestionBox" class="interview-question-box" style="display:none;">
  <div id="interviewQuestionContent" class="interview-question-content"></div>
  <div class="interview-question-toggle" onclick="toggleInterviewQuestionBox()">
    <span id="toggleArrow">▼</span>
  </div>
</div>
<div class="mc2" id="mc2"></div>
<div class="cb" id="cb" style="display:none"><span class="cb-icon">💡</span><span class="ct2" id="ct2"></span></div>
<div class="ia">
<button class="mb" onclick="toggleCameraPanel()">📷</button>
<button class="mb" id="micInputBtn" onclick="toggleM()">🎙️</button>
<input class="ci2" id="ci2" placeholder="输入消息..." onkeypress="if(event.key==='Enter')send()" autocomplete="off" name="message_input">
<button class="sb" onclick="send()">发送</button>
</div>
</div>
<div class="mp" id="monitorPanel">
<div class="mt">🎥 摄像头监控</div>
<div class="cam-preview" id="camPreview">
<div class="cam-placeholder" id="camPlaceholder">摄像头未开启</div>
<video id="camVideo" autoplay muted playsinline"></video>
</div>
<select id="camSelect"><option value="">📷 选择摄像头</option></select>
<button id="cmb" onclick="toggleC()">📷 开启摄像头</button>
<div style="display:flex;flex-direction:column;gap:8px;margin-top:10px">
<div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:8px"><span id="ei">❓</span><div style="font-size:10px;color:#666;margin-top:2px">表情</div><div id="et" style="font-size:11px;color:#333">未检测</div></div>
</div>
</div>
<div style="margin-top:14px;">
    <div class="mt">🎤 麦克风</div>
    <select id="micSelect"><option value="">🎤 选择麦克风</option></select>
    <button id="mmb" onclick="toggleM()">🎤 开始录音</button>
    <div id="micStatus" style="margin-top:8px;font-size:12px;color:#666;">未录音</div>
    <div id="voiceEmotion" style="margin-top:6px;font-size:12px;color:#333;">语音情感: --</div>
</div>
<button class="interrupt-fab" id="interruptBtn" onclick="interrupt()" style="display:none;">⏸️ 打断</button>
<button class="rescue-fab" onclick="rescue()">💡 救场</button>
</div>

<div id="p4" class="page"><div class="rc" id="rc"></div></div>
<div id="p5" class="page">
    <div class="profile-wrap">
        <div class="profile-head">
            <button class="bb" onclick="goBackFromProfile()">返回</button>
            <div>
                <div class="profile-title">个人中心</div>
                <div class="profile-sub" id="profileSub">历史对话报告</div>
            </div>
            <div></div>
        </div>
        <div id="profileContent"></div>
    </div>
</div>

<script>
let sid=null,scene='家庭饭桌试炼',mems=[],chars=[],hist=[],cam=null,mic=null,isC=0,isM=0;
let selectedScenarioId='shandong_dinner';
let selectedPressureTags=[];
let pressureValue=5;
let emotionData={confidence:50,calm:50,nervous:20,focus:50};
let emotionInterval=null;
let talkingHeadTimer=null,lastVoiceLevel=0,lastSpeaker='';
const npcRenderState={};
let isFirstCameraClick=true;
let isFirstMicClick=true;
let isNPCSpeaking=false;
let pendingUtterances=[];
let shouldAwaitUser=true;
let profileSceneActive=null;
let lastPageBeforeProfile='p1';
const pool={
"家庭饭桌试炼":{id:"shandong_dinner",icon:"🍜",members:[
{a:"👴",n:"大舅",r:"主陪·长辈",b:"看重礼数与体面，善于在热闹中施压"},
{a:"👵",n:"大妗子",r:"观察者",b:"温和追问细节，擅长把话题落到现实"},
{a:"🧑",n:"表哥",r:"气氛组",b:"会替长辈推进节奏，也会给你台阶"}
]},
"商务饭局谈判":{id:"business_dinner",icon:"🤝",members:[
{a:"👨‍💼",n:"王总",r:"甲方负责人",b:"注重结果与执行，关心合作确定性"},
{a:"👔",n:"李总",r:"乙方商务",b:"善于铺垫关系，强调互利与长期合作"},
{a:"🧠",n:"周顾问",r:"风险顾问",b:"盯条款边界和落地风险，追问很尖锐"}
]},
"群面竞争场":{id:"interview",icon:"💼",members:[
{a:"👩‍💼",n:"竞争者A",r:"面试对手",b:"自信强势，善于表现自己"},
{a:"🧑‍💼",n:"竞争者B",r:"面试对手",b:"沉稳细致，回答有条理"},
{a:"👨‍💼",n:"竞争者C",r:"面试对手",b:"思维活跃，常有创新观点"}
]},
"立场攻防辩论":{id:"debate",icon:"⚔️",members:[
{a:"🟦",n:"正方辩手",r:"主张方",b:"强调收益、效率与可行性"},
{a:"🟥",n:"反方辩手",r:"质疑方",b:"强调代价、风险与外部性"},
{a:"🧑‍⚖️",n:"点评席",r:"评审",b:"专抓逻辑漏洞，追问证据来源"}
]}
};
const presetSceneDescriptions={
"家庭饭桌试炼":"春节返乡家宴，长辈主导节奏，话题围绕工作进展、婚恋与人情往来。你需要稳住礼貌、边界与表达力度。",
"商务饭局谈判":"合作签约前夜的商务晚宴，重点在信任、利益边界与合作节奏。语言要有分寸，既给面子也守底线。",
"群面竞争场":"多人面试现场，与其他应聘者同台竞争。主面试官在旁观察，你需要在竞争中展现优势，答案需结论先行、证据支撑。",
"立场攻防辩论":"围绕公共议题展开攻防，强调定义清晰、证据质量和反驳针对性。避免空泛口号。"
};

const banquetLevelDescriptions={
"formal":"正式商务宴请：高端酒店包间，精心布置的餐桌，双方高层悉数到场。这是礼仪性资源展示，信任建立的前置仪式。着装正式，举止得体，言谈谨慎而有分量。酒过三巡后才逐渐进入正题，重点在建立关系、展示诚意，为后续合作铺路。",
"informal":"非正式摸底：装修雅致的私房菜餐厅，氛围相对轻松。双方试探，话里有话。看似随意的闲聊中暗藏机锋，每一个话题都可能是在打探底线。不需要过于正式，但要时刻保持警觉，听懂弦外之音，同时巧妙地传递自己的立场。",
"truth":"酒后吐真言：酒过数巡，氛围变得热烈而直接。高压下的情感博弈，测试忠诚度。酒精卸下了部分伪装，话语开始变得尖锐和真实。这是考验彼此信任和底线的时刻，需要在保持清醒的同时，应对各种情感和利益的考验。",
"street":"深夜大排档：霓虹灯闪烁的街头，塑料板凳，冰镇啤酒。卸下伪装，进行最后的利益交换。没有了办公室的繁文缛节，大家都露出了最真实的一面。这是敲定最终细节的时刻，直接、务实、不绕弯子，但也要守住自己的核心利益。"
};

const interviewQuestions={
"互联网":{
"产品":"请分享你最近使用的一个产品，分析它的核心需求、用户痛点和你的改进建议。",
"销售":"假设我们要推出一款新的SaaS产品，目标客户是中小企业，你会如何设计销售策略？",
"市场":"请为我们即将上线的社交产品设计一个冷启动的营销策略，预算50万。",
"分析":"如果给你一份用户行为数据，你会从哪些维度进行分析，来提升产品的用户留存？"
},
"金融":{
"产品":"请设计一款面向年轻人群的理财产品，说明它的核心卖点和风控机制。",
"销售":"如何向高净值客户推荐我们的财富管理服务？请模拟一个销售场景。",
"市场":"请为我们银行的信用卡业务设计一个年度营销方案，目标是提升年轻客群的活跃度。",
"分析":"如果发现某款理财产品的赎回率突然上升，你会如何分析原因并给出建议？"
},
"快消":{
"产品":"请为我们的新品奶茶设计一个产品概念，包括口味、包装和定价策略。",
"销售":"如何在3个月内将一款新零食打进本地的连锁超市渠道？",
"市场":"请为我们的品牌设计一个双11的营销活动，目标是提升销售额30%。",
"分析":"如果发现某款产品在某个区域的销量下滑，你会如何分析原因并给出改进建议？"
},
"咨询":{
"产品":"请分享你做过的一个产品咨询项目，说明你的分析框架和最终成果。",
"销售":"如何向一家传统企业销售数字化转型咨询服务？请说明你的销售流程。",
"市场":"请为一家新成立的咨询公司设计品牌定位和市场推广策略。",
"分析":"如果客户说他们的利润率在下降，你会如何进行分析并给出建议？"
}
};

let selectedBanquetLevel='formal';
let drinkingCapacity=0;
const scenes=Object.keys(pool);
function $(id){return document.getElementById(id)}

const dicebearStylePool=['avataaars','pixel-art','lorelei','notionists'];
const dicebearOptionAllow=new Set(['top','accessories','facialHair','clothing','eyes','eyebrows','mouth','skinColor','hairColor','facialHairColor','accessoriesColor','clothingColor','hatColor']);
const useExternalDicebear=false;

function hashSeed(str='npc'){
    let h=0;
    for(let i=0;i<str.length;i++){h=((h<<5)-h)+str.charCodeAt(i);h|=0}
    return Math.abs(h);
}
function pickStyle(seed){
    const idx=hashSeed(seed)%dicebearStylePool.length;
    return dicebearStylePool[idx];
}
function pickBySeed(seed,list,offset=0){
    if(!Array.isArray(list)||list.length===0)return '';
    return list[(hashSeed(seed)+offset)%list.length];
}
function inferGender(member){
    const raw=String(
        `${member?.gender||member?.sex||''} ${member?.n||member?.name||''} ${member?.r||member?.role||''} ${member?.b||member?.background||''} ${member?.p||member?.personality||''}`
    ).toLowerCase();
    if(/female|woman|girl|lady|女|女生|女性|妈妈|阿姨|姐姐|妹妹|大妗子|婶|嫂/.test(raw))return 'female';
    if(/male|man|boy|gentleman|男|男生|男性|叔|伯|爷|哥哥|弟弟|大舅|表哥|主陪/.test(raw))return 'male';
    return 'unknown';
}
function inferAgeGroup(member){
    const explicit=Number(member?.age);
    if(Number.isFinite(explicit)&&explicit>0){
        if(explicit>=55)return 'senior';
        if(explicit>=35)return 'middle';
        return 'young';
    }
    const raw=String(
        `${member?.n||member?.name||''} ${member?.r||member?.role||''} ${member?.b||member?.background||''} ${member?.p||member?.personality||''}`
    ).toLowerCase();
    if(/爷|奶|伯|叔|姑父|舅|妗|长辈|senior|elder|主陪/.test(raw))return 'senior';
    if(/新人|晚辈|学生|实习|候选|junior|intern|student|candidate|表弟|表妹/.test(raw))return 'young';
    return 'middle';
}
function inferIdentity(member){
    const raw=String(`${member?.r||member?.role||''} ${member?.b||member?.background||''} ${member?.p||member?.personality||''}`).toLowerCase();
    if(/老板|总|领导|主任|长辈|面试官|评委|甲方|boss|leader|manager|director|主陪|长者/.test(raw))return 'senior';
    if(/顾问|法务|风控|财务|老师|consult|advisor|legal|risk|观察者/.test(raw))return 'advisor';
    if(/商务|销售|客户|运营|bd|sales|business|client|operation/.test(raw))return 'business';
    if(/新人|晚辈|候选|实习|学生|junior|candidate|intern|student|气氛组/.test(raw))return 'junior';
    if(/技术|研发|程序|工程师|产品|engineer|developer|tech|product/.test(raw))return 'tech';
    return 'neutral';
}
function getSceneProfileOverrides(member){
    const name=String(member?.n||member?.name||'');
    const role=String(member?.r||member?.role||'');
    if(name.includes('大舅')||role.includes('主陪')){
        return {gender:'male',ageGroup:'senior',identity:'senior',mustache:true,glasses:false,smile:0,hairType:'short'};
    }
    if(name.includes('大妗子')||role.includes('观察者')){
        return {gender:'female',ageGroup:'middle',identity:'advisor',mustache:false,glasses:true,smile:0,hairType:'long'};
    }
    if(name.includes('表哥')||role.includes('气氛组')){
        return {gender:'male',ageGroup:'young',identity:'junior',mustache:false,glasses:false,smile:1,hairType:'short'};
    }
    return null;
}
function buildIdentityTraits(member,seed){
    const stableSeed=seed||member?.n||member?.name||'npc';
    const gender=inferGender(member);
    const identity=inferIdentity(member);
    const base={
        style:'avataaars',
        options:{
            top:pickBySeed(stableSeed,['shortHairShortFlat','shortHairTheCaesar','shortHairFrizzle','longHairStraight2','longHairMiaWallace']),
            accessories:pickBySeed(stableSeed,['none','prescription01','prescription02','round'],3),
            facialHair:'none',
            clothing:pickBySeed(stableSeed,['shirtCrewNeck','blazerShirt','blazerSweater','hoodie'],5),
            eyes:pickBySeed(stableSeed,['default','happy','side','squint'],7),
            eyebrows:pickBySeed(stableSeed,['default','upDown','raisedExcited','raisedExcitedNatural'],11),
            mouth:pickBySeed(stableSeed,['default','smile','serious'],13),
        }
    };
    if(gender==='female'){
        base.options.top=pickBySeed(stableSeed,['longHairStraight2','longHairMiaWallace','longHairBob','longHairCurly']);
        base.options.facialHair='none';
    }else if(gender==='male'){
        base.options.top=pickBySeed(stableSeed,['shortHairShortFlat','shortHairTheCaesar','shortHairDreads02','shortHairShortWaved']);
        base.options.facialHair=pickBySeed(stableSeed,['none','beardLight','moustacheFancy'],17);
    }
    if(identity==='senior'){
        base.style='avataaars';
        base.options.clothing=pickBySeed(stableSeed,['blazerShirt','blazerSweater','shirtCrewNeck'],19);
        base.options.accessories=pickBySeed(stableSeed,['prescription02','prescription01','none'],23);
        base.options.mouth=pickBySeed(stableSeed,['serious','default','smile'],29);
    }else if(identity==='advisor'){
        base.style='avataaars';
        base.options.accessories=pickBySeed(stableSeed,['prescription01','prescription02','round'],31);
        base.options.clothing=pickBySeed(stableSeed,['shirtCrewNeck','blazerShirt','shirtScoopNeck'],37);
    }else if(identity==='business'){
        base.style='avataaars';
        base.options.clothing=pickBySeed(stableSeed,['blazerShirt','shirtVNeck','shirtCrewNeck'],41);
        base.options.eyebrows=pickBySeed(stableSeed,['default','raisedExcited','upDown'],43);
    }else if(identity==='junior'){
        base.style='notionists';
        base.options.clothing=pickBySeed(stableSeed,['hoodie','graphicShirt','shirtCrewNeck'],47);
        base.options.mouth=pickBySeed(stableSeed,['smile','default','twinkle'],53);
    }else if(identity==='tech'){
        base.style='pixel-art';
        base.options.clothing=pickBySeed(stableSeed,['hoodie','graphicShirt','shirtCrewNeck'],59);
        base.options.accessories=pickBySeed(stableSeed,['none','round','prescription01'],61);
    }else{
        base.style=pickStyle(stableSeed);
    }
    return base;
}
function normalizeVisualTraits(traits,seed,member){
    const normalized=buildIdentityTraits(member,seed);
    if(traits&&typeof traits==='object'){
        if(traits.style)normalized.style=String(traits.style);
        const opts=traits.options||traits.params||{};
        Object.keys(opts||{}).forEach(k=>{
            if(!dicebearOptionAllow.has(k))return;
            const v=opts[k];
            if(v!==null&&v!==undefined&&String(v).trim()!==''){
                normalized.options[k]=String(v);
            }
        });
    }
    return normalized;
}
function buildDicebearUrl(traits,seed,member){
    const safeSeed=seed||'npc';
    const normalized=normalizeVisualTraits(traits,safeSeed,member);
    const params=new URLSearchParams();
    params.set('seed',safeSeed);
    Object.entries(normalized.options).forEach(([k,v])=>params.set(k,v));
    return `https://api.dicebear.com/7.x/${normalized.style}/svg?${params.toString()}`;
}
function resolveAvatarUrl(member){
    if(!member)return null;
    if(member.avatarUrl)return member.avatarUrl;
    const raw=member.avatar||member.a;
    if(raw&&/^https?:\/\//.test(raw))return raw;
    if(useExternalDicebear){
        const traits=member.visualTraits||member.visual_traits||null;
        return buildDicebearUrl(traits,member.n||member.name||'npc',member);
    }
    return buildFallbackAvatarDataUrl(member);
}
function getAvatarInitials(name){
    const text=String(name||'NPC').trim();
    if(!text)return 'NP';
    if(/[\u4e00-\u9fa5]/.test(text)){
        return text.slice(0,2);
    }
    const parts=text.split(/\s+/).filter(Boolean);
    if(parts.length>=2){
        return (parts[0][0]+parts[1][0]).toUpperCase();
    }
    return text.slice(0,2).toUpperCase();
}
function buildFallbackAvatarDataUrl(member){
    const seed=(member?.n||member?.name||'npc');
    const profileOverride=getSceneProfileOverrides(member)||{};
    const identity=profileOverride.identity||inferIdentity(member);
    const gender=profileOverride.gender||inferGender(member);
    const ageGroup=profileOverride.ageGroup||inferAgeGroup(member);
    const h=hashSeed(`${seed}|${identity}|${gender}|${ageGroup}`);
    const bgPalette=[
        ['#dbeafe','#bfdbfe'],
        ['#dcfce7','#bbf7d0'],
        ['#fef3c7','#fde68a'],
        ['#f3e8ff','#e9d5ff'],
        ['#ffe4e6','#fecdd3']
    ];
    const skinPalette=['#F4C7A1','#EAB38F','#D79A74','#C68662'];
    const hairMale=['#1f2937','#374151','#111827','#4b5563'];
    const hairFemale=['#111827','#3f3f46','#78350f','#4c1d95'];
    const clothByIdentity={
        senior:['#334155','#0f172a'],
        advisor:['#1d4ed8','#2563eb'],
        business:['#0f766e','#115e59'],
        junior:['#7c3aed','#6d28d9'],
        tech:['#0f766e','#0f172a'],
        neutral:['#475569','#334155']
    };
    const bg=bgPalette[h%bgPalette.length];
    const skin=skinPalette[(h+1)%skinPalette.length];
    const hair=(gender==='female'?hairFemale:hairMale)[(h+2)%4];
    const cloth=(clothByIdentity[identity]||clothByIdentity.neutral)[(h+3)%2];
    const eyeColor=['#111827','#1f2937','#0f172a'][(h+4)%3];
    const useGlasses=(profileOverride.glasses===true)||((profileOverride.glasses!==false)&&(identity==='advisor'||identity==='senior')&&(h%2===0));
    const hasBeard=(profileOverride.mustache===true)||((profileOverride.mustache!==false)&&(gender==='male')&&(ageGroup!=='young')&&(h%3===0));
    const smileLevel=profileOverride.smile===1?1:(profileOverride.smile===0?0:((identity==='business'||identity==='junior')?1:0));
    const ageWrinkle=ageGroup==='senior';
    const preferLong=profileOverride.hairType==='long' || (profileOverride.hairType!=='short' && gender==='female');
    const hairTopPath = preferLong
        ? "M18,34 C20,14 34,8 48,8 C64,8 78,16 80,34 L80,42 C74,34 66,30 48,30 C32,30 24,34 18,42 Z"
        : "M20,35 C22,20 34,12 48,12 C62,12 74,20 76,35 L76,40 C69,34 60,32 48,32 C36,32 27,34 20,40 Z";
    const hairSideFemale = preferLong
        ? "<path d='M18,40 C16,52 18,66 24,76 L30,76 C24,64 24,52 26,42 Z' fill='"+hair+"' opacity='0.95'/><path d='M78,40 C80,52 78,66 72,76 L66,76 C72,64 72,52 70,42 Z' fill='"+hair+"' opacity='0.95'/>"
        : "";
    const mouthPath = smileLevel
        ? "M40,62 C44,66 52,66 56,62"
        : "M40,64 C44,62 52,62 56,64";
    const glassesSvg = useGlasses
        ? "<rect x='33' y='48' width='11' height='8' rx='2' fill='none' stroke='#334155' stroke-width='1.5'/><rect x='52' y='48' width='11' height='8' rx='2' fill='none' stroke='#334155' stroke-width='1.5'/><line x1='44' y1='52' x2='52' y2='52' stroke='#334155' stroke-width='1.2'/>"
        : "";
    const beardSvg = hasBeard
        ? "<path d='M37,66 C40,74 56,74 59,66 C57,71 39,71 37,66 Z' fill='#374151' opacity='0.85'/>"
        : "";
    const wrinkleSvg = ageWrinkle
        ? "<line x1='36' y1='46' x2='43' y2='46' stroke='#b08968' stroke-width='0.8' opacity='0.6'/><line x1='53' y1='46' x2='60' y2='46' stroke='#b08968' stroke-width='0.8' opacity='0.6'/>"
        : "";
    const svg=`<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 96 96'>
<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='${bg[0]}'/><stop offset='100%' stop-color='${bg[1]}'/></linearGradient></defs>
<rect x='2' y='2' width='92' height='92' rx='20' fill='url(#bg)'/>
<ellipse cx='48' cy='94' rx='38' ry='20' fill='${cloth}' opacity='0.18'/>
<path d='M18,96 C20,78 30,72 48,72 C66,72 76,78 78,96 Z' fill='${cloth}'/>
<rect x='43' y='66' width='10' height='9' rx='4' fill='${skin}'/>
<circle cx='27' cy='51' r='4' fill='${skin}'/>
<circle cx='69' cy='51' r='4' fill='${skin}'/>
<ellipse cx='48' cy='50' rx='22' ry='24' fill='${skin}'/>
<path d='${hairTopPath}' fill='${hair}'/>
${hairSideFemale}
<circle cx='40' cy='52' r='2.1' fill='${eyeColor}'/>
<circle cx='56' cy='52' r='2.1' fill='${eyeColor}'/>
<path d='M36,47 C38,46 42,46 44,47' stroke='#374151' stroke-width='1.4' fill='none' stroke-linecap='round'/>
<path d='M52,47 C54,46 58,46 60,47' stroke='#374151' stroke-width='1.4' fill='none' stroke-linecap='round'/>
<path d='${mouthPath}' stroke='#7f1d1d' stroke-width='1.5' fill='none' stroke-linecap='round'/>
${glassesSvg}
${beardSvg}
${wrinkleSvg}
</svg>`;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}
function renderAvatarMarkup(member,wrapperClass){
    const url=resolveAvatarUrl(member);
    if(url){
        return `<div class="${wrapperClass}"><img src="${url}" alt="${member?.n||member?.name||'avatar'}"></div>`;
    }
    const emoji=member?.a||member?.avatar||'🙂';
    return `<div class="${wrapperClass} avatar-emoji">${emoji}</div>`;
}
function renderInlineAvatar(avatar,name){
    if(avatar&&/^(https?:\/\/|data:image\/)/.test(avatar)){
        return `<img src="${avatar}" alt="${name||'avatar'}" style="width:24px;height:24px;border-radius:50%;object-fit:cover;">`;
    }
    return `<span>${avatar||'👤'}</span>`;
}
function ensureMemberVisuals(member){
    if(!member)return member;
    if(!member.visualTraits){
        member.visualTraits=buildIdentityTraits(member,member.n||member.name||'npc');
    }
    if(!member.avatar&&member.a)member.avatar=member.a;
    member.avatarUrl=resolveAvatarUrl(member);
    return member;
}
function mapAICharacter(c){
    const member={
        a:c.avatar||'👤',
        avatar:c.avatar||'👤',
        n:c.name||'NPC',
        r:c.role||'',
        b:c.background||c.personality||'未知',
        p:c.personality||'',
        gender:c.gender||c.sex||'',
        visualTraits:c.visualTraits||c.visual_traits||null
    };
    return ensureMemberVisuals(member);
}

let npcVoiceEnabled=false;
let npcAudioQueue=[];
let npcAudioPlaying=false;
let npcAudio=null;
function updateNpcVoiceButton(){
    const btn=$('npcVoiceToggle');
    if(!btn)return;
    btn.classList.toggle('on',npcVoiceEnabled);
    btn.textContent=npcVoiceEnabled?'NPC语音: 开':'NPC语音: 关';
}
function stopNpcVoice(){
    npcAudioQueue=[];
    npcAudioPlaying=false;
    if(npcAudio){
        npcAudio.pause();
        npcAudio.currentTime=0;
    }
}
function toggleNpcVoice(){
    npcVoiceEnabled=!npcVoiceEnabled;
    if(!npcVoiceEnabled)stopNpcVoice();
    updateNpcVoiceButton();
}
function queueNpcAudio(url){
    if(!url)return;
    npcAudioQueue.push(url);
    if(!npcAudioPlaying)playNextNpcAudio();
}
function playNextNpcAudio(){
    if(!npcVoiceEnabled){npcAudioQueue=[];return;}
    const url=npcAudioQueue.shift();
    if(!url){npcAudioPlaying=false;return;}
    npcAudioPlaying=true;
    npcAudio=new Audio(url);
    npcAudio.onended=()=>{npcAudioPlaying=false;playNextNpcAudio()};
    npcAudio.onerror=()=>{npcAudioPlaying=false;playNextNpcAudio()};
    npcAudio.play().catch(()=>{npcAudioPlaying=false;playNextNpcAudio()});
}
async function speakNPC(text,emotion){
    if(!npcVoiceEnabled)return;
    const clean=String(text||'').replace(/^\s*[^：:]{1,6}[：:]\s*/,'').trim();
    if(!clean)return;
    try{
        const r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:clean,emotion:emotion||'neutral'})});
        const d=await r.json();
        if(d.success&&d.data&&d.data.url){
            queueNpcAudio(d.data.url);
        }
    }catch(e){}
}

function setDrinkingCapacity(score){
    drinkingCapacity=score;
    const stars=document.querySelectorAll('#drinkingCapacityStars .star');
    stars.forEach((star,index)=>{
        if(index<score){
            star.textContent='★';
            star.classList.add('filled');
        } else {
            star.textContent='☆';
            star.classList.remove('filled');
        }
    });
}

let selectedIndustry='互联网';
let selectedPosition='产品';

function updateInterviewQuestion(){
    if(scene !== '群面竞争场') return;
    
    let question = '';
    const customIndustryVal = $('customIndustryInput').value;
    const customPositionVal = $('customPositionInput').value;
    
    const industry = selectedIndustry === '自定义' ? customIndustryVal : selectedIndustry;
    const position = selectedPosition === '自定义' ? customPositionVal : selectedPosition;
    
    if(interviewQuestions[selectedIndustry] && interviewQuestions[selectedIndustry][selectedPosition]){
        question = interviewQuestions[selectedIndustry][selectedPosition];
    } else if(industry && position){
        question = `请分享一个你在${industry}行业做${position}相关工作的经历，说明你遇到的最大挑战和解决方案。`;
    } else {
        question = '请介绍你自己，并分享一个最能体现你能力的项目经历。';
    }
    
    applySceneInfo(question);
}

async function generateCustomInterviewQuestion(){
    const btn = document.getElementById('aiCustomizeBtn');
    const originalText = btn.textContent;
    
    const customIndustryVal = $('customIndustryInput').value;
    const customPositionVal = $('customPositionInput').value;
    
    const industry = selectedIndustry === '自定义' ? customIndustryVal : selectedIndustry;
    const position = selectedPosition === '自定义' ? customPositionVal : selectedPosition;
    
    if(!industry || !position){
        alert('请先选择或输入行业和岗位');
        return;
    }
    
    try {
        btn.disabled = true;
        btn.textContent = '生成中...';
        
        const r = await fetch('/api/interview/generate_question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                industry: industry,
                position: position
            })
        });
        
        const d = await r.json();
        if (d.success && d.data && d.data.question) {
            applySceneInfo(d.data.question);
        } else {
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        alert('生成失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function selectIndustry(el){
    selectedIndustry=el.dataset.value;
    document.querySelectorAll('#industryTags .interview-tag').forEach(t=>t.classList.remove('selected'));
    el.classList.add('selected');
    
    const customInput=$('customIndustryInput');
    if(selectedIndustry==='自定义'){
        customInput.style.display='block';
        customInput.focus();
    } else {
        customInput.style.display='none';
    }
    
    updateInterviewQuestion();
}

function selectPosition(el){
    selectedPosition=el.dataset.value;
    document.querySelectorAll('#positionTags .interview-tag').forEach(t=>t.classList.remove('selected'));
    el.classList.add('selected');
    
    const customInput=$('customPositionInput');
    if(selectedPosition==='自定义'){
        customInput.style.display='block';
        customInput.focus();
    } else {
        customInput.style.display='none';
    }
    
    updateInterviewQuestion();
}

function isPresetScene(){return !!pool[scene]}
function applySceneInfo(description){
    const sceneDescText=document.getElementById('sceneDescriptionText');
    const sceneDescEdit=document.getElementById('sceneDescriptionEdit');
    sceneDescText.innerText=description||'';
    sceneDescEdit.value=description||'';
    document.getElementById('sceneInfoSection').style.display=description?'block':'none';
    document.getElementById('sceneDescription').style.display=description?'block':'none';
}
function refreshSceneInfoForSelection(){
    const btn=document.getElementById('sceneGenBtn');
    const sceneInfoTitle=document.getElementById('sceneInfoTitle');
    const aiCustomizeBtn=document.getElementById('aiCustomizeBtn');
    const preset=isPresetScene();
    if(preset){
        // 根据场景更新标题和按钮显示
        if(scene === '群面竞争场'){
            sceneInfoTitle.textContent='面试问题';
            aiCustomizeBtn.style.display='inline-block';
        } else {
            sceneInfoTitle.textContent='背景信息';
            aiCustomizeBtn.style.display='none';
        }
        
        // 如果是商务饭局谈判场景，使用酒局等级对应的描述
        if(scene === '商务饭局谈判'){
            applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
            btn.style.display='block';
        } else if(scene === '群面竞争场'){
            updateInterviewQuestion();
            btn.style.display='none';
        } else {
            applySceneInfo(presetSceneDescriptions[scene]||`${scene}场景，角色和背景已预置。`);
            btn.style.display='block';
        }
        document.getElementById('memberSection').style.display='block';
        document.getElementById('mg').style.display='flex';
        document.getElementById('actionButtons').style.display='flex';
        btn.disabled=false;
        btn.textContent='🔄 重新生成场景设定';
        return;
    }
    applySceneInfo('');
    btn.disabled=false;
    btn.textContent='🔄 重新生成场景设定';
    btn.style.display='block';
    sceneInfoTitle.textContent='背景信息';
    aiCustomizeBtn.style.display='none';
}
function detectEmotionLabel(t){
    if(!t)return'neutral';
    if(/[哈哈|高兴|开心|好|不错]/i.test(t))return'happy';
    if(/[尴尬|不好意思|抱歉|难过]/i.test(t))return'sad';
    if(/[不行|不能|不喝|生气|别闹]/i.test(t))return'angry';
    return'neutral';
}
function detectEmotion(t){if(!t)return'😐';if(/[哈哈|高兴|开心|好|不错]/i.test(t))return'😊';if(/[谢谢|感谢|感激]/i.test(t))return'🙏';if(/[尴尬|不好意思|抱歉]/i.test(t))return'😳';if(/[不行|不能|不喝]/i.test(t))return'😤';if(/[干|喝|走一个]/i.test(t))return'🍺';return'😐'}
function buildHeadCard(c){const m=ensureMemberVisuals(c);return `<div class="ci state-idle look-user expr-neutral" data-n="${m.n}"><div class="head">${renderAvatarMarkup(m,'avatar-main')}<span class="avatar-exp">😐</span></div><div class="npc-meta"><div class="cn">${m.n}</div><div class="role">${m.r||''}</div><div class="mood-pill">平静</div></div></div>`}
function setRenderState(name,patch={}){if(!npcRenderState[name])npcRenderState[name]={state:'idle',look:'user',backchannel:''};Object.assign(npcRenderState[name],patch)}
function _resolveExpression(st){
    if(st.state==='speaking')return {key:'engaged',label:'发言中',emoji:'🗣️'};
    if(st.state==='reacting')return {key:'warm',label:'有回应',emoji:'🙂'};
    if(st.state==='listening'&&st.look==='speaker')return {key:'focused',label:'在专注',emoji:'🤔'};
    if(st.state==='listening')return {key:'calm',label:'在聆听',emoji:'😌'};
    return {key:'neutral',label:'平静',emoji:'😐'};
}
function applyRenderState(name){const card=document.querySelector(`.ci[data-n="${name}"]`);if(!card)return;const st=npcRenderState[name]||{state:'idle',look:'user',backchannel:''};card.classList.remove('state-idle','state-listening','state-reacting','state-speaking','look-user','look-speaker','has-backchannel','expr-neutral','expr-calm','expr-focused','expr-engaged','expr-warm');card.classList.add(`state-${st.state}`);card.classList.add(`look-${st.look||'user'}`);const expr=_resolveExpression(st);card.classList.add(`expr-${expr.key}`);const mood=card.querySelector('.mood-pill');if(mood)mood.textContent=expr.label;const exp=card.querySelector('.avatar-exp');if(exp)exp.textContent=expr.emoji}
function blinkRandom(){document.querySelectorAll('#cl .ci').forEach(card=>{if(card.classList.contains('state-speaking'))return;if(Math.random()<0.05){card.classList.add('blink');setTimeout(()=>card.classList.remove('blink'),120)}})}
function renderConversationState(mode,speaker=''){
    const names=chars.map(c=>c.n);
    if(!names.length)return;
    names.forEach((name,idx)=>{
        if(mode==='npc_speaking'){
            const isSpeaker=name===speaker;
            setRenderState(name,{state:isSpeaker?'speaking':'listening',look:isSpeaker?'user':'speaker'});
        }else if(mode==='after_npc'){
            const isSpeaker=name===speaker;
            setRenderState(name,{state:isSpeaker?'reacting':'listening',look:'user'});
        }else if(mode==='user_speaking'){
            setRenderState(name,{state:'listening',look:'user'});
        }else{
            setRenderState(name,{state:'listening',look:'user'});
        }
        applyRenderState(name);
    });
}
function inferBeat(){const confusion=Math.max(0,Math.min(100,(100-emotionData.focus+emotionData.nervous)/2));const stress=Math.max(0,Math.min(100,(emotionData.nervous+(100-emotionData.calm))/2));if(stress>66||confusion>70)return 'controlled_rescue';if(scene.includes('面试')||selectedScenarioId==='interview')return 'pressure_check';return 'table_banter'}
function runNonverbalLoop(){if(talkingHeadTimer)clearInterval(talkingHeadTimer);talkingHeadTimer=setInterval(()=>{if(!$('p3').classList.contains('active'))return;blinkRandom()},1400)}
function goBackFromGame(){
    if(confirm('返回将清除当前对话记录，确定要返回吗？')){
        stopNpcVoice();
        sid=null;
        hist=[];
        pendingUtterances=[];
        shouldAwaitUser=true;
        isNPCSpeaking=false;
        lastSpeaker='';
        
        $('mc2').innerHTML='';
        $('cl').innerHTML='';
        $('cb').style.display='none';
        $('interruptBtn').style.display='none';
        updScr(50,50);
        renderConversationState('idle');
        
        show('p2');
    }
}
function show(p){document.querySelectorAll('.page').forEach(e=>e.classList.remove('active'));$(p).classList.add('active');const loginBtn=document.getElementById('loginBtn');if(p==='p3'&&!currentUser){loginBtn.style.display='none'}else if(!currentUser){loginBtn.style.display='block'}}
function selectPressureTag(el){
    const tag = el.dataset.tag;
    console.log('点击标签:', tag);
    console.log('当前选中标签:', selectedPressureTags);
    
    // 切换选中状态
    if(el.classList.contains('selected')){
        // 取消选中
        el.classList.remove('selected');
        selectedPressureTags = selectedPressureTags.filter(t => t !== tag);
    } else {
        // 选中
        el.classList.add('selected');
        if(!selectedPressureTags.includes(tag)){
            selectedPressureTags.push(tag);
        }
    }
    
    console.log('更新后选中标签:', selectedPressureTags);
    
    // 处理自定义输入
    const customPressureInput = $('customPressureInput');
    if(selectedPressureTags.includes('自定义')){
        customPressureInput.style.display = 'block';
    } else {
        customPressureInput.style.display = 'none';
    }
    
    // 暂时始终显示压力值滑块，用于调试
    const pressureValueBox = $('pressureValueBox');
    console.log('压力值盒子元素:', pressureValueBox);
    pressureValueBox.style.display = 'flex';
}

function updatePressureValue(value){
    pressureValue = parseInt(value);
    $('pressureDisplay').textContent = value;
}

function selectBanquetLevel(el){
    const level = el.dataset.level;
    selectedBanquetLevel = level;
    
    // 取消所有选中状态
    document.querySelectorAll('.banquet-level-tag').forEach(t=>t.classList.remove('selected'));
    // 选中当前标签
    el.classList.add('selected');
    
    // 更新场景信息
    applySceneInfo(banquetLevelDescriptions[level]);
}

function refreshBanquetLevelInfo(){
    if(scene === '商务饭局谈判'){
        applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
    }
}
function goCfg(){show('p2')}
function selScene(el){
    document.querySelectorAll('.sc').forEach(e=>e.classList.remove('on'));
    el.classList.add('on');
    scene=el.dataset.s;
    const p=pool[scene];
    selectedScenarioId=p?p.id:'shandong_dinner';
    
    // 只在家庭饭桌场景显示压力敏感区
    const pressureSectionWrapper = $('pressureSectionWrapper');
    const customPressureInput = $('customPressureInput');
    const pressureValueBox = $('pressureValueBox');
    
    if(scene.includes('家庭')){
        pressureSectionWrapper.style.display = 'block';
    } else {
        pressureSectionWrapper.style.display = 'none';
        customPressureInput.style.display = 'none';
        pressureValueBox.style.display = 'none';
        // 清除选择
        document.querySelectorAll('.pressure-tag').forEach(t=>t.classList.remove('selected'));
        selectedPressureTags = [];
    }
    
    // 只在商务饭局谈判场景显示酒局等级
    const banquetLevelWrapper = $('banquetLevelWrapper');
    
    if(scene === '商务饭局谈判'){
        banquetLevelWrapper.style.display = 'block';
        // 应用选中的酒局等级对应的场景信息
        applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
    } else {
        banquetLevelWrapper.style.display = 'none';
    }
    
    // 只在群面竞争场场景显示面试信息
    const interviewInfoWrapper = $('interviewInfoWrapper');
    
    if(scene === '群面竞争场'){
        interviewInfoWrapper.style.display = 'block';
    } else {
        interviewInfoWrapper.style.display = 'none';
    }
    
    genMems();
}
function genMems(){
    const p=pool[scene];
    if(p){
        mems=p.members.slice(0,3);
        selectedScenarioId=p.id;
        
        // 设置默认用户身份，根据场景调整
        let userRole = '参与者';
        let userBackground = '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。';
        
        if(scene.includes('家庭')){
            userRole = '晚辈';
            userBackground = '作为家中的晚辈，你需要在长辈面前展现礼貌和尊重，同时巧妙应对长辈的各种关怀和询问。';
        } else if(scene.includes('商务') || scene.includes('客户')){
            userRole = '部门新人';
            userBackground = '作为公司的新人，你需要在商务宴请中展示专业素养，学会得体应对客户的各种话题和敬酒。';
        } else if(scene.includes('面试')){
            userRole = '候选人';
            userBackground = '你需要结论先行、证据支撑，面对追问保持稳定和可验证性。';
        } else if(scene.includes('辩论')){
            userRole = '辩手';
            userBackground = '你需要定义清晰、证据充分，并对对方核心论点做针对性反驳。';
        }
        
        window.userInfo = {
            a: '👨‍💼',
            n: '你',
            r: userRole,
            b: userBackground
        };
    }else{
        mems=pool['家庭饭桌试炼'].members.slice(0,3);
        selectedScenarioId='shandong_dinner';
        
        // 默认用户信息
        window.userInfo = {
            a: '👨‍💼',
            n: '你',
            r: '参与者',
            b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
        };
    }
    renderMems();
    renderScenes();
    refreshSceneInfoForSelection();
}
function renderScenes(){$('sg').innerHTML=scenes.map(s=>{
    const isDebate = s === "立场攻防辩论";
    const displayName = isDebate ? "日常纠纷化解" : s;
    const disabledClass = isDebate ? " disabled" : "";
    const extraStyle = isDebate ? 'style="opacity:0.5;pointer-events:none;cursor:not-allowed;"' : '';
    const extraContent = isDebate ? '<div style="font-size:12px;color:#888;margin-top:4px;">（开发中）</div>' : '';
    return `<div class="sc${s===scene?' on':''}${disabledClass}" data-s="${s}" ${isDebate ? '' : `onclick="selScene(this)"`} ${extraStyle}><div style="font-size:24px">${pool[s].icon}</div><div>${displayName}</div>${extraContent}</div>`;
}).join('')}
function renderMems(){
    // 使用动态用户信息，如果未设置则使用默认值
    const userInfo = ensureMemberVisuals(window.userInfo || {
        a: '👨‍💼',
        n: '你',
        r: '参与者',
        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
    });
    window.userInfo=userInfo;
    
    const userMember = `<div class="mc mc-tooltip" style="border:2px solid #4A90E2;background:#E3F2FD;position:relative;cursor:pointer" data-tooltip="${userInfo.b}">
        <div style="position:absolute;top:-10px;right:-10px;width:60px;height:60px;background:#2196F3;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;transform:rotate(15deg);box-shadow:0 2px 5px rgba(0,0,0,0.2);z-index:10;">你的角色</div>
        <div style="position:absolute;top:5px;right:5px;cursor:pointer;font-size:16px;" onclick="editMember('user')">✏️</div>
        ${renderAvatarMarkup(userInfo,'ma')}
        <div class="mn" style="color:#2196F3;">${userInfo.n}</div>
        <div style="background:#2196F3;color:#fff;padding:4px 8px;border-radius:10px;font-size:12px;margin:5px 0;">${userInfo.r}</div>
        <div style="font-size:13px;color:#666;line-height:1.4;">${userInfo.b.substring(0, 50)}${userInfo.b.length > 50 ? '...' : ''}</div>
    </div>`;
    
    $('mg').innerHTML=mems.map((m,i)=>{
        const member=ensureMemberVisuals(m);
        return `
        <div class="mc mc-tooltip" style="position:relative;cursor:pointer" data-tooltip="${member.b || member.personality || '无详细信息'}">
            <div style="position:absolute;top:5px;right:5px;cursor:pointer;font-size:16px;" onclick="editMember(${i})">✏️</div>
            ${renderAvatarMarkup(member,'ma')}
            <div class="mn">${member.n}</div>
            <div style="background:#E3F2FD;color:#2196F3;padding:4px 8px;border-radius:10px;font-size:12px;margin:5px 0;">${member.r}</div>
            <div style="font-size:13px;color:#666;line-height:1.4;">${(member.b || member.personality || '无详细信息').substring(0, 50)}${(member.b || member.personality || '').length > 50 ? '...' : ''}</div>
        </div>
    `}).join('') + userMember;
}

function toggleSceneEdit() {
    const textDiv = document.getElementById('sceneDescriptionText');
    const editWrapper = document.getElementById('sceneEditWrapper');
    const editArea = document.getElementById('sceneDescriptionEdit');
    
    if (editWrapper.style.display === 'none') {
        // 切换到编辑模式
        editArea.value = textDiv.innerText;
        textDiv.style.display = 'none';
        editWrapper.style.display = 'block';
        editArea.focus();
    } else {
        // 切换回显示模式
        textDiv.innerText = editArea.value;
        textDiv.style.display = 'block';
        editWrapper.style.display = 'none';
    }
}

function confirmSceneEdit() {
    const textDiv = document.getElementById('sceneDescriptionText');
    const editWrapper = document.getElementById('sceneEditWrapper');
    const editArea = document.getElementById('sceneDescriptionEdit');
    
    textDiv.innerText = editArea.value;
    textDiv.style.display = 'block';
    editWrapper.style.display = 'none';
}

async function aiOptimizeContent() {
    const btn = document.getElementById('aiOptimizeBtn');
    const originalText = btn.textContent;
    const editArea = document.getElementById('sceneDescriptionEdit');
    const currentContent = editArea.value.trim();
    
    if (!currentContent) {
        alert('请先输入一些内容再进行优化');
        return;
    }
    
    // 确定场景类型
    let sceneType = 'general';
    if (scene === '家庭饭桌试炼') {
        sceneType = 'family';
    } else if (scene === '商务饭局谈判') {
        sceneType = 'business';
    } else if (scene === '群面竞争场') {
        sceneType = 'interview';
    }
    
    try {
        btn.disabled = true;
        btn.textContent = '优化中...';
        
        const r = await fetch('/api/content/optimize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                content: currentContent,
                scene_type: sceneType
            })
        });
        
        const d = await r.json();
        if (d.success && d.data && d.data.optimized_content) {
            editArea.value = d.data.optimized_content;
        } else {
            alert('优化失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        alert('优化失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function editMember(index) {
    let member;
    if (index === 'user') {
        member = window.userInfo || {
            a: '👨‍💼',
            n: '你',
            r: '参与者',
            b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
        };
    } else {
        member = mems[index];
    }
    
    const modal = document.createElement('div');
    modal.id = 'editModal';
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:1000;';
    
    modal.innerHTML = `
        <div style="background:white;border-radius:10px;padding:20px;width:90%;max-width:500px;max-height:80vh;overflow-y:auto;">
            <h3 style="margin:0 0 15px 0;color:#333;">编辑成员信息</h3>
            <div style="margin-bottom:15px;">
                <label style="display:block;margin-bottom:5px;font-weight:bold;color:#555;">姓名</label>
                <input type="text" id="editName" value="${member.n}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:5px;font-size:14px;">
            </div>
            <div style="margin-bottom:15px;">
                <label style="display:block;margin-bottom:5px;font-weight:bold;color:#555;">角色</label>
                <input type="text" id="editRole" value="${member.r}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:5px;font-size:14px;">
            </div>
            <div style="margin-bottom:15px;">
                <label style="display:block;margin-bottom:5px;font-weight:bold;color:#555;">背景故事</label>
                <textarea id="editBackground" style="width:100%;min-height:100px;padding:8px;border:1px solid #ddd;border-radius:5px;font-size:14px;resize:vertical;">${member.b}</textarea>
            </div>
            <div style="display:flex;gap:10px;justify-content:flex-end;">
                <button onclick="closeEditModal()" style="padding:8px 16px;border:1px solid #ddd;background:white;border-radius:5px;cursor:pointer;">取消</button>
                <button onclick="saveMemberEdit(${index})" style="padding:8px 16px;border:none;background:#2196F3;color:white;border-radius:5px;cursor:pointer;">保存</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

function closeEditModal() {
    const modal = document.getElementById('editModal');
    if (modal) {
        modal.remove();
    }
}

function saveMemberEdit(index) {
    const name = document.getElementById('editName').value;
    const role = document.getElementById('editRole').value;
    const background = document.getElementById('editBackground').value;
    
    if (index === 'user') {
        window.userInfo.n = name;
        window.userInfo.r = role;
        window.userInfo.b = background;
    } else {
        mems[index].n = name;
        mems[index].r = role;
        mems[index].b = background;
    }
    
    renderMems();
    closeEditModal();
}

async function randMem() {
    try {
        const b = document.querySelector('button[onclick="randMem()"]');
        const originalText = b.textContent;
        
        // 更改按钮文本为动态加载文案
        const loadingMessages = ['正在重新设计人物...', '正在构建新的人物关系...', '正在生成新角色...', '即将完成...'];
        let currentIndex = 0;
        let intervalId;
        
        // 显示加载文案，显示完后停留在最后一个文案
        intervalId = setInterval(() => {
            if (currentIndex < loadingMessages.length) {
                b.textContent = loadingMessages[currentIndex];
                currentIndex++;
            } else {
                // 已经显示完所有文案，停止定时器并保持在最后一个文案
                clearInterval(intervalId);
            }
        }, 1000);
        
        b.disabled = true;
        
        const r = await fetch('/api/scenario/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                scene_type: selectedScenarioId, 
                scene_name: scene,
                only_characters: true // 只生成成员信息
            })
        });
        
        clearInterval(intervalId);
        
        const d = await r.json();
        if (d.success) {
            // 更新成员信息
            if (d.data.characters && d.data.characters.length > 0) {
                // 只取前3个作为NPC
                mems = d.data.characters.slice(0, 3).map(mapAICharacter);
                
                // 如果AI提供了用户身份信息，则更新全局用户身份
                if (d.data.user_identity) {
                    window.userInfo = ensureMemberVisuals({
                        a: d.data.user_identity.avatar || '👤',
                        avatar: d.data.user_identity.avatar || '👤',
                        n: d.data.user_identity.name || '你',
                        r: d.data.user_identity.role || '参与者',
                        b: d.data.user_identity.background || d.data.user_identity.personality || '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。',
                        p: d.data.user_identity.personality || '',
                        gender: d.data.user_identity.gender || d.data.user_identity.sex || '',
                        visualTraits: d.data.user_identity.visualTraits || d.data.user_identity.visual_traits || null
                    });
                } else {
                    // 默认用户信息
                    window.userInfo = ensureMemberVisuals({
                        a: '👨‍💼',
                        n: '你',
                        r: '参与者',
                        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                    });
                }
                
                renderMems();
            }
        } else {
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        console.error('生成成员时出错:', e);
        const b = document.querySelector('button[onclick="randMem()"]');
        b.textContent = '随机换人';
        alert('生成成员时出错，请稍后再试');
    } finally {
        const b = document.querySelector('button[onclick="randMem()"]');
        b.textContent = '随机换人';
        b.disabled = false;
    }
}

async function regenerateScene() {
    // 显示确认框
    if (!confirm('⚠️ 重新生成将覆盖当前编辑内容，确定继续？')) {
        return;
    }
    
    try {
        const b = document.getElementById('sceneGenBtn');
        const originalText = b.textContent;
        
        // 更改按钮文本为动态加载文案
        const loadingMessages = ['正在重新设计场景...', '正在重构人物关系...', '正在生成新设定...', '即将完成...'];
        let currentIndex = 0;
        let intervalId;
        
        // 显示加载文案，显示完后停留在最后一个文案
        intervalId = setInterval(() => {
            if (currentIndex < loadingMessages.length) {
                b.textContent = loadingMessages[currentIndex];
                currentIndex++;
            } else {
                // 已经显示完所有文案，停止定时器并保持在最后一个文案
                clearInterval(intervalId);
            }
        }, 1000);
        
        b.disabled = true;
        
        const r = await fetch('/api/scenario/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                scene_type: selectedScenarioId, 
                scene_name: scene,
                banquet_level: scene === '商务饭局谈判' ? selectedBanquetLevel : null
            })
        });
        
        clearInterval(intervalId);
        
        const d = await r.json();
        if (d.success) {
            // 更新场景描述
            if (d.data.description) {
                const sceneDescText = document.getElementById('sceneDescriptionText');
                const sceneDescEdit = document.getElementById('sceneDescriptionEdit');
                sceneDescText.innerText = d.data.description;
                sceneDescEdit.value = d.data.description;
                
                // 显示场景信息部分
                document.getElementById('sceneInfoSection').style.display = 'block';
                document.getElementById('sceneDescription').style.display = 'block';
            }
            
            // 更新成员信息
            if (d.data.characters && d.data.characters.length > 0) {
                // 只取前3个作为NPC
                mems = d.data.characters.slice(0, 3).map(mapAICharacter);
                
                // 如果AI提供了用户身份信息，则更新全局用户身份
                if (d.data.user_identity) {
                        window.userInfo = ensureMemberVisuals({
                            a: d.data.user_identity.avatar || '👤',
                            avatar: d.data.user_identity.avatar || '👤',
                            n: d.data.user_identity.name || '你',
                            r: d.data.user_identity.role || '参与者',
                            b: d.data.user_identity.background || d.data.user_identity.personality || '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。',
                            p: d.data.user_identity.personality || '',
                            gender: d.data.user_identity.gender || d.data.user_identity.sex || '',
                            visualTraits: d.data.user_identity.visualTraits || d.data.user_identity.visual_traits || null
                        });
                    } else {
                        // 默认用户信息
                        window.userInfo = ensureMemberVisuals({
                            a: '👨‍💼',
                            n: '你',
                            r: '参与者',
                            b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                        });
                    }
                
                renderMems();
                
                // 显示成员信息部分
                document.getElementById('memberSection').style.display = 'block';
                document.getElementById('mg').style.display = 'flex';
                document.getElementById('actionButtons').style.display = 'flex';
                
                // 改变按钮文字为"重新生成背景信息"
                alert('✅ 场景设定已重新生成！');
            }
        } else {
            b.textContent = '🔄 重新生成场景设定';
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        console.error('生成场景时出错:', e);
        const b = document.getElementById('sceneGenBtn');
        b.textContent = '🔄 重新生成场景设定';
        alert('❌ 重新生成场景失败：' + e.message);
    } finally {
        const b = document.getElementById('sceneGenBtn');
        b.disabled = false;
        b.textContent = '🔄 重新生成场景设定';
    }
}
async function start(){
chars=mems.map(m=>ensureMemberVisuals({...m}));
show('p3');
$('cl').innerHTML=chars.map(c=>buildHeadCard(c)).join('');
renderConversationState('idle');
runNonverbalLoop();
updScr(50,50);

// 处理面试问题显示
const interviewQuestionBox = $('interviewQuestionBox');
const interviewQuestionContent = $('interviewQuestionContent');
const sceneDescription = $('sceneDescriptionEdit')?.value || $('sceneDescriptionText')?.innerText || '';

if(scene === '群面竞争场'){
    interviewQuestionContent.textContent = sceneDescription;
    interviewQuestionContent.classList.remove('collapsed');
    interviewQuestionBox.style.display = 'block';
    document.querySelector('.interview-question-toggle').classList.remove('collapsed');
    document.getElementById('toggleArrow').textContent = '▼';
} else {
    interviewQuestionBox.style.display = 'none';
}

try{const r=await fetch('/api/session/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario_id:selectedScenarioId,scene_name:scene,characters:chars,scene_description:sceneDescription,user_info:(window.userInfo||null)})});
const d=await r.json();if(!d.success){alert(d.error);return}
sid=d.data.session_id;
if(d.data.is_unified_agent){
    pendingUtterances=d.data.utterances||[];
    shouldAwaitUser=d.data.should_await_user!==false;
    if(pendingUtterances.length>0){
        displayUtterances();
    }else if(shouldAwaitUser){
        shouldAwaitUser=true;
        $('interruptBtn').style.display='none';
    }
}else{
    if(d.data.opening)addBot(d.data.opening,null,detectEmotion(d.data.opening))
}
}catch(e){alert(e)}
}

function toggleInterviewQuestionBox(){
    const content = $('interviewQuestionContent');
    const toggle = document.querySelector('.interview-question-toggle');
    const arrow = document.getElementById('toggleArrow');
    
    if(content.classList.contains('collapsed')){
        content.classList.remove('collapsed');
        toggle.classList.remove('collapsed');
        arrow.textContent = '▼';
    } else {
        content.classList.add('collapsed');
        toggle.classList.add('collapsed');
        arrow.textContent = '▲';
    }
}
async function send(){
const t=$('ci2').value.trim();if(!t||!sid)return;$('ci2').value='';stopNpcVoice();renderConversationState('user_speaking');addUser(t);
const multimodal={emotion:emotionData,voice_features:lastVoiceFeatures||null,voice_text:lastVoiceText||''};
console.log('[Send] 消息:', t);console.log('[Send] 情感数据:', multimodal);
try{const r=await fetch('/api/chat/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:t,multimodal:multimodal})});
const d=await r.json();console.log('[Chat] 响应:', JSON.stringify(d, null, 2));if(d.success){
    if(d.data.utterances){
        pendingUtterances=d.data.utterances||[];
        shouldAwaitUser=d.data.should_await_user!==false;
        if(pendingUtterances.length>0){
            displayUtterances();
        }else if(shouldAwaitUser){
            shouldAwaitUser=true;
            $('interruptBtn').style.display='none';
        }
        if(d.data.judgment){$('cb').style.display='flex';let judge=d.data.judgment;if(d.data.npc_feedback_quality&&d.data.npc_feedback_quality.label){judge+=`（质量：${d.data.npc_feedback_quality.label}）`}$('ct2').textContent=judge}
        if(d.data.new_dominance)updScr(d.data.new_dominance.user,d.data.new_dominance.ai);
        if(d.data.scores)updateMetrics(d.data.scores);
        if(d.data.game_over)setTimeout(end,2000);
    }else{
        if(d.data.ai_text)addBot(d.data.ai_text,d.data.speaker,detectEmotion(d.data.ai_text));
        if(d.data.judgment){$('cb').style.display='flex';let judge=d.data.judgment;if(d.data.npc_feedback_quality&&d.data.npc_feedback_quality.label){judge+=`（质量：${d.data.npc_feedback_quality.label}）`}$('ct2').textContent=judge}
        updScr(d.data.new_dominance.user,d.data.new_dominance.ai);
        updateMetrics(d.data.scores);
        if(d.data.game_over)setTimeout(end,2000)
    }
    if(d.data.multimodal_analysis&&d.data.multimodal_analysis.emotion_state){
        const em=d.data.multimodal_analysis.emotion_state.primary_emotion||'neutral';
        lastVoiceEmotion=em;
        setVoiceEmotion(em);
    }
}}catch(e){console.log('[Chat] 错误:', e)}
lastVoiceFeatures=null;
lastVoiceText='';

}
function addUser(t){hist.push({role:'user',content:t});const c=$('mc2');c.innerHTML+=`<div class="msg u"><div class="mco">${t}</div></div>`;c.scrollTop=c.scrollHeight}

function addBot(t,sp,emo){return addBotStreaming(t,sp,emo)}

function addBotStreaming(t,sp,emo){
    hist.push({role:'assistant',content:t});
    const c=$('mc2');
    const msgId='msg-'+Date.now();
    c.innerHTML+=`<div class="msg b" id="${msgId}">${sp?`<div class="ms">${sp}</div>`:''}${emo?`<span class="msg-emo">${emo}</span>`:''}<div class="mco"></div></div>`;
    c.scrollTop=c.scrollHeight;
    
    const speaker=sp||chars[0]?.n||'';
    if(speaker){
        lastSpeaker=speaker;
        renderConversationState('npc_speaking',speaker);
        const card=document.querySelector(`.ci[data-n="${speaker}"] .ca`);
        if(card){card.style.transform='scale(1.12)';setTimeout(()=>card.style.transform='scale(1)',220)}
    }else{
        renderConversationState('idle');
    }
    
    speakNPC(t, detectEmotionLabel(t));
    const mco=document.querySelector(`#${msgId} .mco`);
    let idx=0;
    return new Promise((resolve)=>{
        const typeChar=()=>{
            if(idx<t.length){
                mco.textContent+=t.charAt(idx);
                idx++;
                c.scrollTop=c.scrollHeight;
                setTimeout(typeChar,30);
            }else{
                if(speaker){
                    setTimeout(()=>renderConversationState('after_npc',speaker),700);
                }
                resolve();
            }
        };
        typeChar();
    });
}

async function displayUtterances(){
    if(pendingUtterances.length===0){
        if(shouldAwaitUser){
            isNPCSpeaking=false;
            $('interruptBtn').style.display='none';
        }else{
            await continueNPC();
        }
        return;
    }
    
    isNPCSpeaking=true;
    $('interruptBtn').style.display='inline-flex';
    
    const utterance=pendingUtterances.shift();
    console.log('[displayUtterances] utterance:', utterance);
    console.log('[displayUtterances] chars:', chars);
    const npc=chars.find(c=>c.n===utterance.npc_id||c.a===utterance.npc_id);
    const speakerName=npc?npc.n:utterance.npc_id;
    console.log('[displayUtterances] speakerName:', speakerName);
    
    await addBotStreaming(utterance.text,speakerName,detectEmotion(utterance.text));
    
    const delay=utterance.delay_ms||3000;
    setTimeout(displayUtterances,delay);
}

async function interrupt(){
    if(!isNPCSpeaking)return;
    
    stopNpcVoice();
    pendingUtterances=[];
    isNPCSpeaking=false;
    $('interruptBtn').style.display='none';
    
    try{
        const r=await fetch('/api/chat/interrupt',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:sid})
        });
        const d=await r.json();
        if(d.success){
            pendingUtterances=d.data.utterances||[];
            shouldAwaitUser=d.data.should_await_user!==false;
        }
    }catch(e){
        console.log('[Interrupt] 错误:',e);
    }
}

async function continueNPC(){
    try{
        const r=await fetch('/api/chat/continue',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:sid})
        });
        const d=await r.json();
        if(d.success){
            pendingUtterances=d.data.utterances||[];
            shouldAwaitUser=d.data.should_await_user!==false;
            if(pendingUtterances.length>0){
                displayUtterances();
            }
        }
    }catch(e){
        console.log('[Continue] 错误:',e);
    }
}
function updScr(u,a){$('us').textContent=Math.round(u);$('as').textContent=Math.round(a)}
async function rescue(){if(!sid)return;try{const r=await fetch('/api/chat/rescue',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})});const d=await r.json();if(d.success)$('ci2').value=d.data.suggestion}catch(e){}}

function buildReportHtml(data){
    const safeData=data||{};
    const npcList=Array.isArray(safeData.npc_os_list)?safeData.npc_os_list:[];
    const scores=(safeData.scores&&typeof safeData.scores==='object')?safeData.scores:{};
    const medalNames={
        '🥇':'社交达人',
        '🥈':'社交能手',
        '🥉':'社交新手',
        '📘':'饭桌木头人'
    };
    const medalName=medalNames[safeData.medal]||'社交新手';
    return `
<div style="display:flex;gap:40px;max-height:80vh;overflow:hidden;width:90%;margin:0 auto;">
    <div style="flex:0 0 380px;display:flex;flex-direction:column;align-items:center;padding:20px;border-right:1px dashed #eee;">
        <h1 style="margin:0;font-size:26px;color:#1a1a1a;">局后复盘</h1>
        <p style="color:#666;font-size:13px;margin-top:4px;">在 "${safeData.scene_name||'未命名场景'}" 中的表现</p>
        <div style="background:#e74c3c;color:white;padding:10px 20px;border-radius:12px;font-weight:800;font-size:18px;transform:rotate(-3deg);box-shadow:4px 8px 15px rgba(231,76,60,0.3);margin:20px 0;">
            ${medalName}
        </div>
        <canvas id="radarChart" width="300" height="300" style="margin:10px 0;"></canvas>
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;width:100%;margin-top:20px;">
            <div style="background:#f8f9fa;padding:10px 5px;border-radius:10px;text-align:center;border:1px solid #eee;">
                <span style="display:block;font-size:11px;color:#666;margin-bottom:4px;">圆滑度</span>
                <b style="font-size:15px;color:#4a5dca;">${scores.oily||0}</b>
            </div>
            <div style="background:#f8f9fa;padding:10px 5px;border-radius:10px;text-align:center;border:1px solid #eee;">
                <span style="display:block;font-size:11px;color:#666;margin-bottom:4px;">亲和力</span>
                <b style="font-size:15px;color:#4a5dca;">${scores.friendliness||0}</b>
            </div>
            <div style="background:#f8f9fa;padding:10px 5px;border-radius:10px;text-align:center;border:1px solid #eee;">
                <span style="display:block;font-size:11px;color:#666;margin-bottom:4px;">逻辑性</span>
                <b style="font-size:15px;color:#4a5dca;">${scores.logic||0}</b>
            </div>
            <div style="background:#f8f9fa;padding:10px 5px;border-radius:10px;text-align:center;border:1px solid #eee;">
                <span style="display:block;font-size:11px;color:#666;margin-bottom:4px;">幽默感</span>
                <b style="font-size:15px;color:#4a5dca;">${scores.humor||0}</b>
            </div>
            <div style="background:#f8f9fa;padding:10px 5px;border-radius:10px;text-align:center;border:1px solid #eee;">
                <span style="display:block;font-size:11px;color:#666;margin-bottom:4px;">懂规矩</span>
                <b style="font-size:15px;color:#4a5dca;">${scores.respect||0}</b>
            </div>
        </div>
    </div>
    <div style="flex:1;overflow-y:auto;padding-right:10px;">
        <div style="background:white;padding:15px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05);margin-bottom:15px;">
            <h3 style="margin:0 0 10px 0;font-size:14px;color:#4a5dca;display:flex;align-items:center;">
                <span style="margin-right:8px;">💬</span> 综合点评
            </h3>
            <p style="margin:0;font-size:13px;line-height:1.6;color:#333;">${safeData.summary||'暂无综合点评'}</p>
        </div>
        <div style="background:white;padding:15px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05);margin-bottom:15px;">
            <h3 style="margin:0 0 10px 0;font-size:14px;color:#4a5dca;display:flex;align-items:center;">
                <span style="margin-right:8px;">🎭</span> NPC 内心 OS
            </h3>
            <div style="display:flex;flex-direction:column;gap:10px;">
                ${npcList.map(npc=>`
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="font-size:24px;">${renderInlineAvatar(npc.avatar,npc.name)}</div>
                    <div>
                        <b style="font-size:13px;color:#333;">${npc.name||'NPC'}</b>
                        <p style="margin:3px 0 0 0;font-size:12px;color:#666;">${npc.content||npc.os||npc.thought||'暂无内心独白'}</p>
                    </div>
                </div>
                `).join('')||'<div style="font-size:12px;color:#94a3b8;">暂无 NPC 侧反馈</div>'}
            </div>
        </div>
        <div style="background:white;padding:15px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h3 style="margin:0 0 10px 0;font-size:14px;color:#4a5dca;display:flex;align-items:center;">
                <span style="margin-right:8px;">🚀</span> 下一轮提升点
            </h3>
            <p style="margin:0;font-size:13px;line-height:1.6;color:#333;">${safeData.suggestion||'继续保持稳定表达，逐步提升场景适配度。'}</p>
        </div>
    </div>
</div>`;
}

function getHistoryKey(){
    const key=authToken||currentUser?.uid||currentUser?.email||'guest';
    return `talkarena_history_${key}`;
}
function loadReports(){
    try{
        const raw=localStorage.getItem(getHistoryKey());
        return raw?JSON.parse(raw):[];
    }catch(e){
        return [];
    }
}
function saveReportToHistory(data){
    if(!data)return;
    const reports=loadReports();
    const record={
        id:Date.now().toString(36),
        timestamp:Date.now(),
        scene_name:data.scene_name||'未命名场景',
        summary:data.summary||'综合点评',
        data:data
    };
    reports.unshift(record);
    localStorage.setItem(getHistoryKey(),JSON.stringify(reports.slice(0,200)));
}
function renderProfile(sceneKey){
    const container=$('profileContent');
    if(!container)return;
    if(!currentUser){
        container.innerHTML=`<div class="report-empty">请先登录后查看个人中心。<div style="margin-top:10px;"><button class="view-btn" onclick="openAuthModal()">去登录</button></div></div>`;
        return;
    }
    const reports=loadReports();
    if(reports.length===0){
        container.innerHTML=`<div class="report-empty">暂无历史对话报告，完成一次练习后会自动保存到这里。</div>`;
        return;
    }
    const groups={};
    reports.forEach(r=>{
        const key=r.scene_name||'未命名场景';
        if(!groups[key])groups[key]=[];
        groups[key].push(r);
    });
    const scenes=Object.keys(groups);
    const active=sceneKey||profileSceneActive||scenes[0];
    profileSceneActive=active;
    const total=reports.length;
    const distHtml=scenes.map(s=>`<div class="dist-row"><span>${s}</span><span>${groups[s].length}</span></div>`).join('');
    const tabsHtml=scenes.map(s=>`<button class="profile-tab ${s===active?'active':''}" data-scene="${s}" onclick="renderProfile(this.dataset.scene)">${s}</button>`).join('');
    const cardsHtml=(groups[active]||[]).map(r=>{
        const date=new Date(r.timestamp);
        const dateLabel=`${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
        return `
        <div class="report-card">
            <h5>${r.scene_name}</h5>
            <p>${r.summary}</p>
            <div style="font-size:12px;color:#94a3b8;">${dateLabel}</div>
            <button class="view-btn" onclick="openHistoryReport('${r.id}')">查看详细报告</button>
        </div>
        `;
    }).join('');
    container.innerHTML=`
        <div class="profile-stats">
            <div class="profile-stat">
                <h4>总练习次数</h4>
                <div class="stat-val">${total}</div>
            </div>
            <div class="profile-stat">
                <h4>各场景练习次数</h4>
                ${distHtml}
            </div>
            <div class="profile-stat">
                <h4>当前场景</h4>
                <div class="stat-val">${active}</div>
            </div>
        </div>
        <div class="profile-tabs">${tabsHtml}</div>
        <div class="report-grid">${cardsHtml}</div>
    `;
}
function openProfile(){
    lastPageBeforeProfile=document.querySelector('.page.active')?.id||'p1';
    show('p5');
    renderProfile();
}
function goBackFromProfile(){
    show(lastPageBeforeProfile||'p1');
}
function openHistoryReport(id){
    const reports=loadReports();
    const record=reports.find(r=>r.id===id);
    if(!record)return;
    $('rc').innerHTML=buildReportHtml(record.data);
    drawRadarChart(record.data.scores);
    show('p4');
}
async function end(){
    if(!sid)return;
    stopNpcVoice();
    try{
        const r=await fetch('/api/session/end',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:sid})
        });
        if(!r.ok){
            throw new Error(`结束会话接口异常: HTTP ${r.status}`);
        }
        const d=await r.json();
        if(!d.success){
            throw new Error(d.error||'结束会话失败');
        }
        const data=(d.data&&typeof d.data==='object')?d.data:{};
        saveReportToHistory(data);
        $('rc').innerHTML=buildReportHtml(data);
        show('p4');
        drawRadarChart(data.scores||{});
    }catch(e){
        console.error('结束会话失败:',e);
        const fallback={
            scene_name:'本轮会话',
            summary:'总结生成失败，请稍后重试。',
            suggestion:'可返回继续对话，或重新开始新会话。',
            scores:{}
        };
        $('rc').innerHTML=buildReportHtml(fallback);
        show('p4');
    }
}

function drawRadarChart(scores){
    const safeScores=(scores&&typeof scores==='object')?scores:{};
    const canvas=document.getElementById('radarChart');
    if(!canvas)return;
    const ctx=canvas.getContext('2d');
    const centerX=canvas.width/2;
    const centerY=canvas.height/2;
    const radius=Math.min(centerX,centerY)-40;
    
    // 五个维度
    const labels=['Oily','Friendly','Logical','Humor','Respect'];
    const values=[
        safeScores.oily||50,
        safeScores.friendliness||50,
        safeScores.logic||50,
        safeScores.humor||50,
        safeScores.respect||50
    ];
    
    // 清空画布
    ctx.clearRect(0,0,canvas.width,canvas.height);
    
    // 绘制背景网格（5 个同心圆）
    ctx.strokeStyle='#e0e0e0';
    ctx.lineWidth=1;
    for(let i=1;i<=5;i++){
        const r=radius*i/5;
        ctx.beginPath();
        ctx.arc(centerX,centerY,r,0,Math.PI*2);
        ctx.stroke();
    }
    
    // 绘制轴线
    ctx.strokeStyle='#d0d0d0';
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const x=centerX+Math.cos(angle)*radius;
        const y=centerY+Math.sin(angle)*radius;
        ctx.beginPath();
        ctx.moveTo(centerX,centerY);
        ctx.lineTo(x,y);
        ctx.stroke();
    }
    
    // 绘制数据多边形
    ctx.strokeStyle='#4a5dca';
    ctx.lineWidth=2;
    ctx.fillStyle='rgba(74,93,202,0.2)';
    ctx.beginPath();
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const value=values[i]/100;
        const x=centerX+Math.cos(angle)*radius*value;
        const y=centerY+Math.sin(angle)*radius*value;
        if(i===0){
            ctx.moveTo(x,y);
        }else{
            ctx.lineTo(x,y);
        }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    
    // 绘制数据点
    ctx.fillStyle='#4a5dca';
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const value=values[i]/100;
        const x=centerX+Math.cos(angle)*radius*value;
        const y=centerY+Math.sin(angle)*radius*value;
        ctx.beginPath();
        ctx.arc(x,y,4,0,Math.PI*2);
        ctx.fill();
    }
    
    // 绘制标签
    ctx.fillStyle='#666';
    ctx.font='11px Arial';
    ctx.textAlign='center';
    ctx.textBaseline='middle';
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const labelX=centerX+Math.cos(angle)*(radius+20);
        const labelY=centerY+Math.sin(angle)*(radius+20);
        ctx.fillText(labels[i],labelX,labelY);
    }
    
    // 绘制刻度值
    ctx.fillStyle='#999';
    ctx.font='10px Arial';
    for(let i=1;i<=5;i++){
        const x=centerX+10;
        const y=centerY-radius*i/5+3;
        ctx.fillText(i*20,x,y);
    }
}
function toggleCameraPanel(){const panel=$('monitorPanel');if(isFirstCameraClick){alert('欢迎使用摄像头功能！\n\n请选择您的摄像头设备，然后点击"开启摄像头"按钮。');isFirstCameraClick=false;}panel.classList.toggle('visible');}
function toggleMicPanel(){if(isFirstMicClick){alert('欢迎使用麦克风功能！\n\n请选择您的麦克风设备，然后点击"开启麦克风"按钮。');isFirstMicClick=false;}toggleM2();}
async function toggleC(){const b=$('cmb'),vid=$('camVideo'),ph=$('camPlaceholder'),camId=$('camSelect').value;if(isC){if(cam)cam.getTracks().forEach(t=>t.stop());if(emotionInterval)clearInterval(emotionInterval);isC=0;b.textContent='📷 开启摄像头';b.classList.remove('on');vid.pause();vid.srcObject=null;vid.style.display='none';ph.style.display='flex';ph.textContent='摄像头未开启';$('ei').textContent='❓';$('et').textContent='未检测';emotionData={confidence:50,calm:50,nervous:20,focus:50};updateEmotionDisplay()}else{try{const constraints={video:{width:320,height:240,facingMode:'user'}};if(camId)constraints.deviceId={exact:camId};cam=await navigator.mediaDevices.getUserMedia(constraints);isC=1;b.textContent='✅ 已开启';b.classList.add('on');vid.srcObject=cam;vid.style.display='block';ph.style.display='none';vid.play().then(()=>{emotionInterval=setInterval(()=>{if(!isC)return;const eList=[{i:'😊',t:'开心',c:80,n:10,cal:60,f:70},{i:'😎',t:'自信',c:90,n:5,cal:50,f:80},{i:'😐',t:'平静',c:40,n:10,cal:90,f:50},{i:'😰',t:'紧张',c:30,n:90,cal:20,f:40},{i:'🤔',t:'思考',c:60,n:30,cal:70,f:95},{i:'🙂',t:'放松',c:70,n:5,cal:80,f:60},{i:'😤',t:'坚定',c:85,n:15,cal:40,f:75}];const e=eList[Math.floor(Math.random()*eList.length)];$('ei').textContent=e.i;$('et').textContent=e.t;emotionData={confidence:e.c,nervous:e.n,calm:e.cal,focus:e.f};updateEmotionDisplay();console.log('[Emotion] 实时分析:', emotionData)},1500)}).catch(e=>{console.log('播放失败:',e)})}catch(e){alert('无法开启摄像头: '+e.message)}}}
function updateEmotionDisplay(){$('val-confidence').textContent=emotionData.confidence;$('val-calm').textContent=emotionData.calm;$('val-nervous').textContent=emotionData.nervous;$('val-focus').textContent=emotionData.focus;$('bar-confidence').style.width=emotionData.confidence+'%';$('bar-calm').style.width=emotionData.calm+'%';$('bar-nervous').style.width=emotionData.nervous+'%';$('bar-focus').style.width=emotionData.focus+'%'}
let micButton=null;
let micStream=null;
let audioContext=null;
let recorderNode=null;
let recordingBuffer=[];
let lastVoiceFeatures=null;
let lastVoiceText='';
let lastVoiceEmotion='neutral';
let recordingStart=0;
let recordSampleRate=44100;

function setMicStatus(text){const el=$('micStatus');if(el)el.textContent=text}
function setVoiceEmotion(text){const el=$('voiceEmotion');if(el)el.textContent=`语音情感: ${text||'--'}`}

function downsampleBuffer(buffer, inRate, outRate){
    if(outRate===inRate)return buffer;
    const sampleRateRatio=inRate/outRate;
    const newLength=Math.round(buffer.length/sampleRateRatio);
    const result=new Float32Array(newLength);
    let offsetResult=0;
    let offsetBuffer=0;
    while(offsetResult<result.length){
        const nextOffsetBuffer=Math.round((offsetResult+1)*sampleRateRatio);
        let accum=0, count=0;
        for(let i=offsetBuffer;i<nextOffsetBuffer&&i<buffer.length;i++){accum+=buffer[i];count++}
        result[offsetResult]=accum/count;
        offsetResult++;
        offsetBuffer=nextOffsetBuffer;
    }
    return result;
}
function encodeWav(samples, sampleRate){
    const buffer=new ArrayBuffer(44+samples.length*2);
    const view=new DataView(buffer);
    const writeString=(offset,str)=>{for(let i=0;i<str.length;i++)view.setUint8(offset+i,str.charCodeAt(i))};
    writeString(0,'RIFF');
    view.setUint32(4,36+samples.length*2,true);
    writeString(8,'WAVE');
    writeString(12,'fmt ');
    view.setUint32(16,16,true);
    view.setUint16(20,1,true);
    view.setUint16(22,1,true);
    view.setUint32(24,sampleRate,true);
    view.setUint32(28,sampleRate*2,true);
    view.setUint16(32,2,true);
    view.setUint16(34,16,true);
    writeString(36,'data');
    view.setUint32(40,samples.length*2,true);
    let offset=44;
    for(let i=0;i<samples.length;i++){
        const s=Math.max(-1,Math.min(1,samples[i]));
        view.setInt16(offset, s<0?s*0x8000:s*0x7FFF, true);
        offset+=2;
    }
    return new Blob([view],{type:'audio/wav'});
}
function startLocalRecording(stream){
    recordingBuffer=[];
    recordingStart=Date.now();
    audioContext=new (window.AudioContext||window.webkitAudioContext)();
    recordSampleRate=audioContext.sampleRate;
    const source=audioContext.createMediaStreamSource(stream);
    recorderNode=audioContext.createScriptProcessor(4096,1,1);
    recorderNode.onaudioprocess=(e)=>{
        const input=e.inputBuffer.getChannelData(0);
        recordingBuffer.push(new Float32Array(input));
    };
    source.connect(recorderNode);
    recorderNode.connect(audioContext.destination);
}
async function stopLocalRecording(){
    if(!audioContext)return null;
    recorderNode.disconnect();
    recorderNode=null;
    await audioContext.close();
    audioContext=null;
    const length=recordingBuffer.reduce((acc,cur)=>acc+cur.length,0);
    const merged=new Float32Array(length);
    let offset=0;
    recordingBuffer.forEach(buf=>{merged.set(buf,offset);offset+=buf.length});
    const downsampled=downsampleBuffer(merged,recordSampleRate,16000);
    return encodeWav(downsampled,16000);
}

async function toggleM2(){
    if(!micButton)micButton=document.getElementById('micInputBtn');
    const b=$('mmb');
    const micId=$('micSelect')?$('micSelect').value:'';
    if(isM){
        isM=0;
        if(micStream)micStream.getTracks().forEach(t=>t.stop());
        if(b)b.textContent='🎤 开始录音';
        if(b)b.classList.remove('on');
        if(micButton)micButton.classList.remove('active');
        setMicStatus('正在识别...');
        const wavBlob=await stopLocalRecording();
        if(wavBlob)await submitLocalSTT(wavBlob);
        return;
    }
    try{
        const constraints={audio:true};
        if(micId)constraints.deviceId={exact:micId};
        micStream=await navigator.mediaDevices.getUserMedia(constraints);
        isM=1;
        if(b)b.textContent='⏹️ 停止录音';
        if(b)b.classList.add('on');
        if(micButton)micButton.classList.add('active');
        setMicStatus('录音中...');
        startLocalRecording(micStream);
    }catch(e){
        alert('无法开启麦克风: '+e.message);
    }
}

async function submitLocalSTT(wavBlob){
    try{
        const fd=new FormData();
        fd.append('file', wavBlob, 'speech.wav');
        const r=await fetch('/api/stt',{method:'POST',body:fd});
        const d=await r.json();
        if(d.success){
            const text=d.data.text||'';
            const input=$('ci2');
            if(input)input.value=text;
            lastVoiceText=text;
            lastVoiceFeatures=d.data.voice_features||null;
            const emo=d.data.emotion_state?.primary_emotion||'neutral';
            lastVoiceEmotion=emo;
            setVoiceEmotion(emo);
            setMicStatus('识别完成');
        }else{
            setMicStatus('识别失败');
        }
    }catch(e){
        setMicStatus('识别失败');
    }
}
function updateMetrics(scores){console.log('[Metrics] 收到分数:', scores);if(scores){const total=Math.round((scores.emotional_intelligence+scores.response_quality+scores.pressure_handling+scores.cultural_fit)/4);$('val-score').textContent=total;$('bar-score').style.width=total+'%'}else{console.log('[Metrics] 分数为空')}}
function toggleM(){toggleM2()}
async function loadDevices(){try{const devs=await navigator.mediaDevices.enumerateDevices();const cams=devs.filter(d=>d.kind==='videoinput');const mics=devs.filter(d=>d.kind==='audioinput');$('camSelect').innerHTML='<option value="">📷 选择摄像头</option>'+cams.map((d,i)=>`<option value="${d.deviceId}">${d.label||'摄像头'+(i+1)}</option>`).join('');$('micSelect').innerHTML='<option value="">🎤 选择麦克风</option>'+mics.map((d,i)=>`<option value="${d.deviceId}">${d.label||'麦克风'+(i+1)}</option>`).join('')}catch(e){}}
window.onload=()=>{
    updateNpcVoiceButton();
    // 初始化场景选择，确保压力敏感区正确显示
    renderScenes();
    genMems();
    // 找到并选中默认场景，确保压力敏感区和酒局等级正确显示
    setTimeout(() => {
        const pressureSectionWrapper = $('pressureSectionWrapper');
        const banquetLevelWrapper = $('banquetLevelWrapper');
        
        if(scene.includes('家庭')){
            pressureSectionWrapper.style.display = 'block';
        }
        
        if(scene === '商务饭局谈判'){
            banquetLevelWrapper.style.display = 'block';
            applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
        }
    }, 100);
    loadDevices();
};
</script>
<!-- 认证模块 -->
<div id="authModal" class="auth-modal" style="display:none;">
    <div class="auth-modal-content">
        <div class="auth-header">
            <h2 id="authTitle">登录</h2>
            <span class="auth-close" onclick="closeAuthModal()">&times;</span>
        </div>
        <div class="auth-tabs">
            <button class="auth-tab active" onclick="switchAuthTab('login')">登录</button>
            <button class="auth-tab" onclick="switchAuthTab('register')">注册</button>
        </div>
        <div class="auth-body">
            <form id="loginForm" class="auth-form">
                <div class="form-group">
                    <label>邮箱</label>
                    <input type="email" id="loginEmail" required placeholder="请输入邮箱">
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" id="loginPassword" required placeholder="请输入密码">
                </div>
                <button type="submit" class="auth-btn">登录</button>
            </form>
            <form id="registerForm" class="auth-form" style="display:none;">
                <div class="form-group">
                    <label>邮箱</label>
                    <input type="email" id="registerEmail" required placeholder="请输入邮箱">
                </div>
                <div class="form-group">
                    <label>用户名</label>
                    <input type="text" id="registerUsername" placeholder="请输入用户名（可选）">
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" id="registerPassword" required placeholder="请设置密码（至少 6 位）" minlength="6">
                </div>
                <button type="submit" class="auth-btn">注册</button>
            </form>
            <div id="authMessage" class="auth-message"></div>
        </div>
    </div>
</div>
<div id="userDisplay" class="user-display" style="display:none;">
    <div class="user-info">
        <span class="user-avatar">👤</span>
        <span class="user-name" id="userName">用户</span>
    </div>
    <button class="profile-btn" onclick="openProfile()">个人中心</button>
    <button class="logout-btn" onclick="logout()">退出</button>
</div>
<style>
.auth-modal{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10000}
.auth-modal-content{background:#fff;border-radius:16px;width:90%;max-width:420px;box-shadow:0 20px 60px rgba(0,0,0,0.3);overflow:hidden}
.auth-header{padding:24px;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center}
.auth-header h2{margin:0;color:#C8102E;font-size:24px}
.auth-close{font-size:28px;color:#999;cursor:pointer;line-height:1}
.auth-close:hover{color:#333}
.auth-tabs{display:flex;border-bottom:1px solid #e5e7eb}
.auth-tab{flex:1;padding:16px;background:none;border:none;border-bottom:3px solid transparent;cursor:pointer;font-size:16px;font-weight:600;color:#666;transition:all .3s}
.auth-tab.active{color:#C8102E;border-bottom-color:#C8102E}
.auth-tab:hover{color:#C8102E}
.auth-body{padding:24px}
.auth-form{display:flex;flex-direction:column;gap:20px}
.form-group{display:flex;flex-direction:column;gap:8px}
.form-group label{font-weight:600;color:#333;font-size:14px}
.form-group input{padding:12px 16px;border:2px solid #e5e7eb;border-radius:8px;font-size:15px;transition:border-color .3s}
.form-group input:focus{outline:none;border-color:#C8102E}
.auth-btn{background:#C8102E;color:#fff;border:none;padding:14px;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;transition:all .3s}
.auth-btn:hover{background:#a00d25;transform:translateY(-2px)}
.auth-message{margin-top:16px;padding:12px;border-radius:8px;font-size:14px;display:none}
.auth-message.success{background:#d4edda;color:#155724;border:1px solid #c3e6cb}
.auth-message.error{background:#f8d7da;color:#721c24;border:1px solid #f5c6cb}
.user-display{position:absolute;top:20px;right:20px;background:#fff;padding:10px 20px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);display:flex;align-items:center;gap:15px}
.user-info{display:flex;align-items:center;gap:10px}
.user-avatar{font-size:24px}
.user-name{font-weight:600;color:#333}
.profile-btn{background:#fff;border:2px solid #c8dcff;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;color:#2563eb;transition:all .3s}
.profile-btn:hover{background:#eff6ff;border-color:#93c5fd;color:#1d4ed8}
.logout-btn{background:#f8f9fa;border:2px solid #e5e7eb;padding:8px 16px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;color:#666;transition:all .3s}
.logout-btn:hover{background:#e5e7eb;color:#333}
.login-btn{position:absolute;top:20px;right:20px;padding:10px 20px;background:#C8102E;color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer;z-index:9999}
.login-btn:hover{background:#a00d25}
</style>
<script>
let currentUser=null,authToken=null;
window.addEventListener('DOMContentLoaded',()=>{checkAuthStatus()});
async function checkAuthStatus(){const token=localStorage.getItem('authToken');if(!token)return;try{const response=await fetch('/api/auth/me',{headers:{'Authorization':`Bearer ${token}`}});const data=await response.json();if(data.success){currentUser=data.user;authToken=token;showUserDisplay();document.getElementById('loginBtn').style.display='none'}else{localStorage.removeItem('authToken')}}catch(error){console.error('检查认证状态失败:',error)}}
function openAuthModal(){document.getElementById('authModal').style.display='flex'}
function closeAuthModal(){document.getElementById('authModal').style.display='none';clearAuthMessage()}
function switchAuthTab(tab){try{const loginForm=document.getElementById('loginForm'),registerForm=document.getElementById('registerForm'),tabs=document.querySelectorAll('.auth-tab');if(tab==='login'){loginForm.style.display='flex';registerForm.style.display='none';if(tabs[0])tabs[0].classList.add('active');if(tabs[1])tabs[1].classList.remove('active');document.getElementById('authTitle').textContent='登录'}else{loginForm.style.display='none';registerForm.style.display='flex';if(tabs[0])tabs[0].classList.remove('active');if(tabs[1])tabs[1].classList.add('active');document.getElementById('authTitle').textContent='注册'}clearAuthMessage()}catch(e){console.error('切换标签失败:',e)}}
function showAuthMessage(message,type){const msgEl=document.getElementById('authMessage');msgEl.textContent=message;msgEl.className='auth-message';msgEl.classList.add(type);msgEl.style.display='block';console.log('显示消息:',message,type)}
function clearAuthMessage(){const msgEl=document.getElementById('authMessage');msgEl.style.display='none';msgEl.textContent='';msgEl.className='auth-message'}
document.getElementById('loginForm').addEventListener('submit',async(e)=>{e.preventDefault();const email=document.getElementById('loginEmail').value,password=document.getElementById('loginPassword').value;try{const response=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password})});const data=await response.json();if(data.success){authToken=data.uid;localStorage.setItem('authToken',authToken);currentUser=data.user;showAuthMessage('登录成功！','success');showUserDisplay();document.getElementById('loginBtn').style.display='none';setTimeout(()=>{closeAuthModal()},1000)}else{showAuthMessage(data.message,'error')}}catch(error){showAuthMessage('登录失败，请稍后重试','error');console.error('登录错误:',error)}});
document.getElementById('registerForm').addEventListener('submit',async(e)=>{e.preventDefault();e.stopPropagation();console.log('开始注册...');const email=document.getElementById('registerEmail').value,password=document.getElementById('registerPassword').value,username=document.getElementById('registerUsername').value||null;console.log('注册信息:',{email,username});try{console.log('发送请求...');const response=await fetch('/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password,username})});console.log('响应状态:',response.status);const data=await response.json();console.log('响应数据:',data);if(data.success){authToken=data.uid;localStorage.setItem('authToken',authToken);currentUser=data.user;showAuthMessage('注册成功！','success');showUserDisplay();document.getElementById('loginBtn').style.display='none';setTimeout(()=>{closeAuthModal()},1000)}else{console.log('注册失败:',data.message);showAuthMessage(data.message,'error')}}catch(error){console.error('注册错误:',error);showAuthMessage('注册失败，请稍后重试','error')}});
function showUserDisplay(){if(!currentUser)return;const userDisplay=document.getElementById('userDisplay'),userName=document.getElementById('userName');userName.textContent=currentUser.username||currentUser.email;userDisplay.style.display='flex'}
function logout(){currentUser=null;authToken=null;localStorage.removeItem('authToken');document.getElementById('userDisplay').style.display='none';document.getElementById('loginBtn').style.display='block';alert('已退出登录')}
window.addEventListener('click',(e)=>{const modal=document.getElementById('authModal');if(e.target===modal){closeAuthModal()}});

const tooltip = document.createElement('div');
tooltip.id = 'customTooltip';
document.body.appendChild(tooltip);

let currentTooltipElement = null;

document.addEventListener('mouseover', (e) => {
    const target = e.target.closest('.mc-tooltip');
    if (target) {
        currentTooltipElement = target;
        const tooltipText = target.getAttribute('data-tooltip');
        if (tooltipText) {
            tooltip.textContent = tooltipText;
            tooltip.classList.add('visible');
        }
    }
});

document.addEventListener('mousemove', (e) => {
    if (tooltip.classList.contains('visible')) {
        const x = e.clientX + 15;
        const y = e.clientY + 15;
        
        const tooltipRect = tooltip.getBoundingClientRect();
        const adjustedX = Math.min(x, window.innerWidth - tooltipRect.width - 20);
        const adjustedY = Math.min(y, window.innerHeight - tooltipRect.height - 20);
        
        tooltip.style.left = adjustedX + 'px';
        tooltip.style.top = adjustedY + 'px';
    }
});

document.addEventListener('mouseout', (e) => {
    const target = e.target.closest('.mc-tooltip');
    if (target === currentTooltipElement) {
        tooltip.classList.remove('visible');
        currentTooltipElement = null;
    }
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
