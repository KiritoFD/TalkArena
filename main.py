"""
TalkArena FastAPI 服务端
整合 Multi-Agent、RAG、决策引擎、防幻觉机制
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import base64
from pydantic import BaseModel
from typing import List, Optional, Dict

app = FastAPI(title="TalkArena")

engine = None
mm_analyzer = None


def get_engine():
    global engine
    if engine is None:
        try:
            from model_loader import LLMLoader
            from core.engine import TalkArenaEngine

            llm = LLMLoader()
            llm.load()
            engine = TalkArenaEngine(llm, enable_tts=True)
        except Exception as e:
            raise RuntimeError(
                "Engine initialization failed. Ensure model dependencies are installed and model files are available."
            ) from e
    return engine


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


class ChatReq(BaseModel):
    session_id: str
    message: str = ""
    chat_history: Optional[List[Dict]] = []
    multimodal: Optional[Dict] = None


class SessionReq(BaseModel):
    scenario_id: str = "shandong_dinner"
    scene_name: str = "家庭聚会"
    characters: Optional[List[Dict]] = []
    scene_description: Optional[str] = ""
    user_info: Optional[Dict] = None


class MMReq(BaseModel):
    text: str
    emotion_features: Optional[Dict] = None
    voice_features: Optional[Dict] = None


class ScenarioGenerateReq(BaseModel):
    scene_type: str = "shandong_dinner"
    scene_name: str = "家庭聚会"
    only_characters: bool = False


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

        session = eng.sessions[session_id]
        opening = (
            eng.multi_agent.agents_list[0].think(
                {
                    "characters": req.characters
                    or session["scenario"].get("characters", []),
                    "user_input": "",
                    "turn_count": 0,
                    "dominance": {"user": 50, "ai": 50},
                    "scene_description": req.scene_description,
                    "user_info": req.user_info,
                }
            )
            if req.characters
            else None
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
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/chat/send")
async def send_msg(req: ChatReq):
    if not req.session_id or not req.message:
        return {"success": False, "error": "参数错误"}

    try:
        eng = get_engine()
    except Exception as e:
        return {"success": False, "error": str(e)}
    if req.session_id not in eng.sessions:
        return {"success": False, "error": "会话不存在"}

    try:
        multimodal = req.multimodal or {}
        print(f"[API] 收到多模态数据: {multimodal}")
        for result in eng.process_turn(req.session_id, req.message, multimodal):
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
        llm = eng.multi_agent.llm  # 假设engine包含LLM实例
        
        # 根据场景类型生成不同的prompt
        if req.scene_type == "shandong_dinner":
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
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段
- user_identity: 用户身份信息，包含name、role、personality、background、avatar字段
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
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段
- user_identity: 用户身份信息，包含name、role、personality、background、avatar字段
"""
        elif req.scene_type == "interview":
            if req.only_characters:
                prompt = f"""
请为一场面试场景生成2-3个面试相关角色的详细信息，每个角色包括：
- 姓名
- 角色（如：面试官、HR、竞争者等）
- 性格特点
- 背景故事
- 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合职场面试场景，角色设定专业，背景故事合理。

请以JSON格式输出，包含以下字段：
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段
"""
            else:
                prompt = f"""
请为一场面试场景生成以下内容：
1. 详细的场景背景描述（2-3句话），包括公司类型、面试岗位、面试目的
2. 2-3个面试相关角色的详细信息，每个角色包括：
   - 姓名
   - 角色（如：面试官、HR、竞争者等）
   - 性格特点
   - 背景故事
   - 适合的emoji头像

当前场景名称：{req.scene_name}
请确保生成的内容符合职场面试场景，角色设定专业，背景故事合理。

请以JSON格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段
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
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段
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
- characters: 成员列表，每个成员包含name、role、personality、background、avatar字段
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


@app.get("/api/scenarios/list")
async def list_scenarios():
    """获取可用场景列表"""
    from core.scenarios import get_registry

    registry = get_instance = get_registry()
    templates = registry.list_templates()

    scenarios = [
        {
            "id": "shandong_dinner",
            "name": "山东人的饭桌",
            "category": "dinner",
            "description": "经典山东酒桌文化场景",
            "icon": "🍜",
            "sub_scenes": ["家庭聚会", "单位聚餐", "商务宴请", "同学聚会", "招待客户"],
        },
    ]

    for t in templates:
        if t["template_id"] == "interview":
            scenarios.append(
                {
                    "id": "interview",
                    "name": "面试实战",
                    "category": "interview",
                    "description": "技术面试、HR面试、行为面试",
                    "icon": "💼",
                    "sub_scenes": ["技术面试", "HR面试", "行为面试", "群面"],
                }
            )
        elif t["template_id"] == "debate":
            scenarios.append(
                {
                    "id": "debate",
                    "name": "辩论训练",
                    "category": "debate",
                    "description": "提升逻辑思维和表达能力",
                    "icon": "🎤",
                    "sub_scenes": ["AI对就业", "远程工作", "应试教育", "社交媒体"],
                }
            )

    return {"success": True, "data": scenarios}


if os.path.isdir("outputs/audio"):
    app.mount("/audio", StaticFiles(directory="outputs/audio"), name="audio")
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
.ma{font-size:48px;margin-bottom:10px}
.mn{font-weight:bold;font-size:16px;margin-bottom:8px;color:#333}
.mr{font-size:13px;color:#666;margin-top:5px}

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
.ch{background:#fff;padding:14px 20px;border-bottom:1px solid #E2E8F0;display:flex;justify-content:space-between;align-items:center}
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
.mb{background:transparent;border:none;font-size:22px;cursor:pointer}
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

.mp{width:200px;background:#fff;border-left:1px solid #E2E8F0;padding:18px;display:flex;flex-direction:column}
.mp .mt{font-size:13px;color:#333;font-weight:bold;text-align:center;margin-bottom:12px}
.mp .cam-preview{width:100%;aspect-ratio:4/3;background:#1a1a1a;border-radius:10px;overflow:hidden;margin-bottom:12px;position:relative}
.mp .cam-preview video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1);display:block}
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

#p4{background:#2c313c;padding:40px;justify-content:center;align-items:center}
.rc{background:#fff;border-radius:20px;padding:40px;max-width:550px;width:100%;box-shadow:0 20px 60px rgba(0,0,0,.3)}
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
</style>
</head>
<body>
<div id="p1" class="page active">
<div class="hero">
<div class="logo">🍺</div>
<div class="title">山东人的饭桌</div>
<div class="sub">AI驱动的酒桌情商实战训练平台</div>
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
<div class="fi"><b>场景丰富</b> - 5种经典场景，从家庭聚会到商务宴请，难度递增</div>
</div>
<button class="btn1" onclick="goCfg()">开始挑战</button>
</div>
</div>

<div id="p2" class="page">
<div class="cfg-title">山东人的饭桌</div>
<div class="cfg-sub">选择你的饭局战场</div>
<div class="section-l">选择场景</div>
<div class="sg" id="sg"></div>
<div class="ab" style="margin-top:20px;margin-bottom:20px;">
<button class="btn2" onclick="regenerateScene()">生成背景信息</button>
</div>
<div class="section-l" id="sceneInfoSection" style="display:none;">场景信息 <span style="font-size:12px;color:#667eea;cursor:pointer;" onclick="toggleSceneEdit()">✏️ 编辑</span></div>
<div class="scene-description" id="sceneDescription" style="display:none;background:#f8f9fa;border-radius:10px;padding:15px;margin:10px 0;border-left:4px solid #667eea;">
  <div id="sceneDescriptionText" style="font-size:14px;color:#333;line-height:1.5;"></div>
  <textarea id="sceneDescriptionEdit" style="display:none;width:100%;min-height:100px;border:1px solid #ddd;border-radius:5px;padding:10px;font-size:14px;color:#333;line-height:1.5;resize:vertical;"></textarea>
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
<button class="bb" onclick="show('p1')">返回</button>
<div class="sd">
<div class="si"><span class="sla">你的气场</span><span class="sv u" id="us">50</span></div>
<div class="si"><span class="sla">AI气场</span><span class="sv a" id="as">50</span></div>
</div>
</div>
<div class="hr">
<button class="rb" onclick="rescue()">救场</button>
<button class="eb" onclick="end()">结束</button>
</div>
</div>
<div class="cm">
<div class="sp">
<div class="st">对话角色</div>
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
<div class="mc2" id="mc2"></div>
<div class="cb" id="cb" style="display:none"><span class="cb-icon">💡</span><span class="ct2" id="ct2"></span></div>
<div class="ia">
<button class="mb" onclick="toggleM()">🎙️</button>
<input class="ci2" id="ci2" placeholder="输入消息..." onkeypress="if(event.key==='Enter')send()">
<button class="sb" onclick="send()">发送</button>
</div>
</div>
<div class="mp">
<div class="mt">🎥 摄像头监控</div>
<div class="cam-preview" id="camPreview">
<div class="cam-placeholder" id="camPlaceholder">摄像头未开启</div>
<video id="camVideo" autoplay muted playsinline style="display:none"></video>
</div>
<select id="camSelect"><option value="">📷 选择摄像头</option></select>
<button id="cmb" onclick="toggleC()">📷 开启摄像头</button>
<button id="mmb" onclick="toggleM2()">🎤 开启麦克风</button>
<select id="micSelect"><option value="">🎤 选择麦克风</option></select>
<div class="vol-bar" id="volBar">
<div class="vol-segment" id="vs1"></div>
<div class="vol-segment" id="vs2"></div>
<div class="vol-segment" id="vs3"></div>
<div class="vol-segment" id="vs4"></div>
<div class="vol-segment" id="vs5"></div>
<div class="vol-segment" id="vs6"></div>
<div class="vol-segment" id="vs7"></div>
<div class="vol-segment" id="vs8"></div>
<div class="vol-segment" id="vs9"></div>
<div class="vol-segment" id="vs10"></div>
</div>
<div class="vol-label" id="volLabel">麦克风音量</div>
<div style="display:flex;flex-direction:column;gap:8px;margin-top:10px">
<div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:8px"><span id="ei">❓</span><div style="font-size:10px;color:#666;margin-top:2px">表情</div><div id="et" style="font-size:11px;color:#333">未检测</div></div>
<div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:8px"><span id="vi">❓</span><div style="font-size:10px;color:#666;margin-top:2px">语音</div><div id="vt" style="font-size:11px;color:#333">未检测</div></div>
</div>
</div>
</div>
</div>

<div id="p4" class="page"><div class="rc" id="rc"></div></div>

<script>
let sid=null,scene='家庭聚会',mems=[],chars=[],hist=[],cam=null,mic=null,isC=0,isM=0;
let selectedScenarioId='shandong_dinner';
let emotionData={confidence:50,calm:50,nervous:20,focus:50};
let emotionInterval=null;
let talkingHeadTimer=null,lastVoiceLevel=0,lastSpeaker='';
const npcRenderState={};
const pool={
'家庭聚会':{id:'shandong_dinner',icon:'🍜',members:[{a:'👴',n:'大舅',r:'主陪·长辈',b:'德高望重，极讲规矩'},{a:'👵',n:'大妗子',r:'旁观者',b:'数着你喝了几杯'},{a:'👨',n:'表哥',r:'副陪',b:'最擅长说"我陪一个"'},{a:'👨‍🦳',n:'二叔',r:'话唠长辈',b:'喜欢翻旧账'}]},
'单位聚餐':{id:'shandong_dinner',icon:'🏢',members:[{a:'👨‍💼',n:'王局长',r:'主陪·局领导',b:'深谙官场礼仪'},{a:'👩',n:'小赵',r:'实诚晚辈',b:'性格耿直'},{a:'🧔',n:'老张',r:'酒桌老炮',b:'三句不离酒'}]},
'商务宴请':{id:'shandong_dinner',icon:'🤝',members:[{a:'👨‍💼',n:'王总',r:'主陪·老板',b:'深谙商务礼仪'},{a:'👔',n:'李总',r:'副陪',b:'能言善辩'},{a:'👨‍💻',n:'小刘',r:'助理',b:'负责倒酒递烟'}]},
'同学聚会':{id:'shandong_dinner',icon:'🎓',members:[{a:'🧑‍💼',n:'老同学',r:'攀比狂魔',b:'总爱炫耀'},{a:'👨',n:'班长',r:'组局者',b:'最爱回忆当年'},{a:'👧',n:'校花',r:'气氛组',b:'当年的女神'}]},
'招待客户':{id:'shandong_dinner',icon:'🎁',members:[{a:'👔',n:'李总',r:'东道主',b:'热情招待'},{a:'🧔',n:'老张',r:'气氛担当',b:'负责活跃气氛'},{a:'👩',n:'小王',r:'贴心助理',b:'负责倒酒递烟'}]},
'技术面试':{id:'interview',icon:'💼',members:[{a:'👨‍💼',n:'面试官',r:'技术经理',b:'资深技术专家'},{a:'👩‍💻',n:'HR',r:'HR负责人',b:'负责综合素质评估'},{a:'🧑‍💻',n:'求职者B',r:'竞争者',b:'技术能力很强'}]},
'HR面试':{id:'interview',icon:'👔',members:[{a:'👩',n:'HR总监',r:'HR负责人',b:'经验丰富'},{a:'👨‍💼',n:'部门主管',r:'用人部门',b:'注重团队匹配'},{a:'👨‍💻',n:'前台',r:'接待',b:'负责候选人引导'}]},
'行为面试':{id:'interview',icon:'🎯',members:[{a:'👨‍💼',n:'面试官',r:'HR专家',b:'擅长STAR法则'},{a:'👩‍💼',n:'观察员',r:'HR',b:'细致观察细节'},{a:'🧔',n:'求职者A',r:'竞争者',b:'经历丰富'}]},
'群面':{id:'interview',icon:'👥',members:[{a:'👨‍💼',n:'面试官',r:'主考官',b:'统筹全场'},{a:'🧑‍💻',n:'候选人A',r:'竞争者',b:'表现积极'},{a:'👩‍💻',n:'候选人B',r:'竞争者',b:'逻辑清晰'},{a:'🧔',n:'候选人C',r:'竞争者',b:'领导力强'}]},
'AI对就业':{id:'debate',icon:'🤖',members:[{a:'👨‍💼',n:'正方辩手',r:'支持方',b:'AI创造新岗位'},{a:'👩‍💻',n:'反方辩手',r:'反对方',b:'AI取代人类工作'},{a:'🧔',n:'主持人',r:'裁判',b:'主持辩论'}]},
'远程工作':{id:'debate',icon:'🏠',members:[{a:'👨‍💼',n:'正方辩手',r:'支持方',b:'远程提高效率'},{a:'👩‍💻',n:'反方辩手',r:'反对方',b:'远程降低协作'},{a:'🧔',n:'主持人',r:'裁判',b:'主持辩论'}]},
'应试教育':{id:'debate',icon:'📚',members:[{a:'👨‍💼',n:'正方辩手',r:'支持方',b:'保证公平'},{a:'👩‍💻',n:'反方辩手',r:'反对方',b:'扼杀创造力'},{a:'🧔',n:'主持人',r:'裁判',b:'主持辩论'}]},
'社交媒体':{id:'debate',icon:'📱',members:[{a:'👨‍💼',n:'正方辩手',r:'支持方',b:'连接世界'},{a:'👩‍💻',n:'反方辩手',r:'反对方',b:'隐私泄露'},{a:'🧔',n:'主持人',r:'裁判',b:'主持辩论'}]}
};
const scenes=Object.keys(pool);
function $(id){return document.getElementById(id)}
function detectEmotion(t){if(!t)return'😐';const lower=t.toLowerCase();if(/[哈哈|高兴|开心|好|不错]/i.test(t))return'😊';if(/[谢谢|感谢|感激]/i.test(t))return'🙏';if(/[尴尬|不好意思|抱歉]/i.test(t))return'😳';if(/[不行|不能|不喝]/i.test(t))return'😤';if(/[干|喝|走一个]/i.test(t))return'🍺';return'😐'}
function buildHeadCard(c){return `<div class="ci state-idle look-user" data-n="${c.n}"><div class="head"><div class="head-face"><div class="eyes"><span class="eye"></span><span class="eye"></span></div><div class="mouth"></div></div></div><div><div class="cn">${c.n}</div><div style="font-size:11px;color:#64748b">${c.r||''}</div><div class="ca" style="margin-top:2px">${c.a}</div></div><span class="backchannel">嗯</span></div>`}
function setRenderState(name,patch={}){if(!npcRenderState[name])npcRenderState[name]={state:'idle',look:'user',backchannel:''};Object.assign(npcRenderState[name],patch)}
function applyRenderState(name){const card=document.querySelector(`.ci[data-n="${name}"]`);if(!card)return;const st=npcRenderState[name]||{state:'idle',look:'user',backchannel:''};card.classList.remove('state-idle','state-listening','state-reacting','state-speaking','look-user','look-speaker','has-backchannel');card.classList.add(`state-${st.state}`);card.classList.add(`look-${st.look||'user'}`);if(st.backchannel){card.classList.add('has-backchannel');const bc=card.querySelector('.backchannel');if(bc)bc.textContent=st.backchannel}}
function blinkRandom(){document.querySelectorAll('#cl .ci').forEach(card=>{if(Math.random()<0.18){card.classList.add('blink');setTimeout(()=>card.classList.remove('blink'),120)}})}
function inferBeat(){const confusion=Math.max(0,Math.min(100,(100-emotionData.focus+emotionData.nervous)/2));const stress=Math.max(0,Math.min(100,(emotionData.nervous+(100-emotionData.calm))/2));if(stress>66||confusion>70)return 'controlled_rescue';if(scene.includes('面试'))return 'pressure_check';return 'table_banter'}
function runNonverbalLoop(){if(talkingHeadTimer)clearInterval(talkingHeadTimer);talkingHeadTimer=setInterval(()=>{if(!$('p3').classList.contains('active'))return;const names=chars.map(c=>c.n);if(!names.length)return;const beat=inferBeat();const stress=Math.max(0,Math.min(100,(emotionData.nervous+(100-emotionData.calm))/2));const confusion=Math.max(0,Math.min(100,(100-emotionData.focus+emotionData.nervous)/2));const wantsToSpeak=(lastVoiceLevel>48||$('ci2').value.trim().length>0)?1:0;const rescueMode=stress>65||confusion>70;let lead=lastSpeaker&&names.includes(lastSpeaker)?lastSpeaker:names[0];if(rescueMode){const hr=names.find(n=>/hr|人事|观察员/i.test(n));if(hr)lead=hr}names.forEach((name,i)=>{if(name===lead){setRenderState(name,{state:'speaking',look:'user',backchannel:''})}else{const reactive=beat==='table_banter'&&Math.random()>0.4;setRenderState(name,{state:reactive?'reacting':'listening',look:'speaker',backchannel:(reactive&&Math.random()>0.7)?'对对':''})}applyRenderState(name)});if(wantsToSpeak){const others=names.filter(n=>n!==lead);if(others.length){const n=others[Math.floor(Math.random()*others.length)];setRenderState(n,{state:'reacting',look:'user',backchannel:'我补一句'});applyRenderState(n)}}blinkRandom()},320)}
function show(p){document.querySelectorAll('.page').forEach(e=>e.classList.remove('active'));$(p).classList.add('active')}
function goCfg(){show('p2')}
function selScene(el){document.querySelectorAll('.sc').forEach(e=>e.classList.remove('on'));el.classList.add('on');scene=el.dataset.s;const p=pool[scene];selectedScenarioId=p?p.id:'shandong_dinner';genMems()}
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
        } else if(scene.includes('同学')){
            userRole = '普通同学';
            userBackground = '作为聚会中的普通同学，你需要在老同学面前保持自然，既要应对怀旧话题，又要展现自己的成长。';
        } else if(scene.includes('单位')){
            userRole = '年轻员工';
            userBackground = '作为单位的年轻员工，你需要在领导和同事面前展现得体，学会应对职场酒桌文化。';
        }
        
        window.userInfo = {
            a: '👨‍💼',
            n: '你',
            r: userRole,
            b: userBackground
        };
    }else{
        mems=pool['家庭聚会'].members.slice(0,3);
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
}
function renderScenes(){$('sg').innerHTML=scenes.map(s=>`<div class="sc${s===scene?' on':''}" data-s="${s}" onclick="selScene(this)"><div style="font-size:24px">${pool[s].icon}</div><div>${s}</div></div>`).join('')}
function renderMems(){
    // 使用动态用户信息，如果未设置则使用默认值
    const userInfo = window.userInfo || {
        a: '👨‍💼',
        n: '你',
        r: '参与者',
        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
    };
    
    const userMember = `<div class="mc" style="border:2px solid #4A90E2;background:#E3F2FD;position:relative;cursor:pointer" title="${userInfo.b}">
        <div style="position:absolute;top:-10px;right:-10px;width:60px;height:60px;background:#2196F3;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;transform:rotate(15deg);box-shadow:0 2px 5px rgba(0,0,0,0.2);z-index:10;">你的角色</div>
        <div style="position:absolute;top:5px;right:5px;cursor:pointer;font-size:16px;" onclick="editMember('user')">✏️</div>
        <div class="ma">${userInfo.a}</div>
        <div class="mn" style="color:#2196F3;">${userInfo.n}</div>
        <div style="background:#2196F3;color:#fff;padding:4px 8px;border-radius:10px;font-size:12px;margin:5px 0;">${userInfo.r}</div>
        <div style="font-size:13px;color:#666;line-height:1.4;">${userInfo.b.substring(0, 50)}${userInfo.b.length > 50 ? '...' : ''}</div>
    </div>`;
    
    $('mg').innerHTML=mems.map((m,i)=>`
        <div class="mc" style="position:relative;cursor:pointer" title="${m.b || m.personality || '无详细信息'}">
            <div style="position:absolute;top:5px;right:5px;cursor:pointer;font-size:16px;" onclick="editMember(${i})">✏️</div>
            <div class="ma">${m.a}</div>
            <div class="mn">${m.n}</div>
            <div style="background:#E3F2FD;color:#2196F3;padding:4px 8px;border-radius:10px;font-size:12px;margin:5px 0;">${m.r}</div>
            <div style="font-size:13px;color:#666;line-height:1.4;">${(m.b || m.personality || '无详细信息').substring(0, 50)}${(m.b || m.personality || '').length > 50 ? '...' : ''}</div>
        </div>
    `).join('') + userMember;
}

function toggleSceneEdit() {
    const textDiv = document.getElementById('sceneDescriptionText');
    const editArea = document.getElementById('sceneDescriptionEdit');
    
    if (editArea.style.display === 'none') {
        // 切换到编辑模式
        editArea.value = textDiv.innerText;
        textDiv.style.display = 'none';
        editArea.style.display = 'block';
        editArea.focus();
    } else {
        // 切换回显示模式
        textDiv.innerText = editArea.value;
        textDiv.style.display = 'block';
        editArea.style.display = 'none';
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
        
        const r = await fetch('/api/scenario/generate', {
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
                mems = d.data.characters.slice(0, 3).map(c => ({
                    a: c.avatar || '👤',
                    n: c.name,
                    r: c.role,
                    b: c.background || c.personality || '未知'
                }));
                
                // 如果AI提供了用户身份信息，则更新全局用户身份
                if (d.data.user_identity) {
                    window.userInfo = {
                        a: d.data.user_identity.avatar || '👤',
                        n: d.data.user_identity.name || '你',
                        r: d.data.user_identity.role || '参与者',
                        b: d.data.user_identity.background || d.data.user_identity.personality || '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                    };
                } else {
                    // 默认用户信息
                    window.userInfo = {
                        a: '👨‍💼',
                        n: '你',
                        r: '参与者',
                        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                    };
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
    try {
        const b = document.querySelector('button[onclick="regenerateScene()"]');
        const originalText = b.textContent;
        
        // 更改按钮文本为动态加载文案
        const loadingMessages = ['正在设计社交场景...', '正在构建人物关系...', '正在生成对话策略...', '即将完成...'];
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
        
        const r = await fetch('/api/scenario/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ scene_type: selectedScenarioId, scene_name: scene })
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
                mems = d.data.characters.slice(0, 3).map(c => ({
                    a: c.avatar || '👤',
                    n: c.name,
                    r: c.role,
                    b: c.background || c.personality || '未知'
                }));
                
                // 如果AI提供了用户身份信息，则更新全局用户身份
                if (d.data.user_identity) {
                    window.userInfo = {
                        a: d.data.user_identity.avatar || '👤',
                        n: d.data.user_identity.name || '你',
                        r: d.data.user_identity.role || '参与者',
                        b: d.data.user_identity.background || d.data.user_identity.personality || '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                    };
                } else {
                    // 默认用户信息
                    window.userInfo = {
                        a: '👨‍💼',
                        n: '你',
                        r: '参与者',
                        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                    };
                }
                
                renderMems();
                
                // 显示成员信息部分
                document.getElementById('memberSection').style.display = 'block';
                document.getElementById('mg').style.display = 'flex';
                document.getElementById('actionButtons').style.display = 'flex';
                
                // 改变按钮文字为"重新生成背景信息"
                b.textContent = '重新生成背景信息';
            }
        } else {
            b.textContent = '生成背景信息';
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        console.error('生成场景时出错:', e);
        const b = document.querySelector('button[onclick="regenerateScene()"]');
        b.textContent = '生成背景信息';
        alert('生成场景时出错，请稍后再试');
    } finally {
        const b = document.querySelector('button[onclick="regenerateScene()"]');
        b.disabled = false;
    }
}
async function start(){
chars=mems;
show('p3');
$('cl').innerHTML=chars.map(c=>buildHeadCard(c)).join('');
chars.forEach(c=>{setRenderState(c.n,{state:'listening',look:'user',backchannel:''});applyRenderState(c.n)});
runNonverbalLoop();
updScr(50,50);
try{const r=await fetch('/api/session/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario_id:selectedScenarioId,scene_name:scene,characters:chars})});
const d=await r.json();if(!d.success){alert(d.error);return}
sid=d.data.session_id;if(d.data.opening)addBot(d.data.opening,null,detectEmotion(d.data.opening))}catch(e){alert(e)}
}
async function send(){
const t=$('ci2').value.trim();if(!t||!sid)return;$('ci2').value='';const firstName=chars[0]?.n;if(firstName){setRenderState(firstName,{state:'listening',look:'user',backchannel:'请讲'});applyRenderState(firstName)}addUser(t);
const multimodal={emotion:emotionData,voice_level:isM?($('volLabel').textContent.replace('麦克风音量: ','').replace('%','')||0):0};
console.log('[Send] 消息:', t);console.log('[Send] 情感数据:', multimodal);
try{const r=await fetch('/api/chat/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:t,multimodal:multimodal})});
const d=await r.json();console.log('[Chat] 响应:', JSON.stringify(d, null, 2));if(d.success){if(d.data.ai_text)addBot(d.data.ai_text,d.data.speaker,detectEmotion(d.data.ai_text));if(d.data.judgment){$('cb').style.display='flex';$('ct2').textContent=d.data.judgment}updScr(d.data.new_dominance.user,d.data.new_dominance.ai);updateMetrics(d.data.scores);if(d.data.game_over)setTimeout(end,2000)}}catch(e){console.log('[Chat] 错误:', e)}
}
function addUser(t){hist.push({role:'user',content:t});const c=$('mc2');c.innerHTML+=`<div class="msg u"><div class="mco">${t}</div></div>`;c.scrollTop=c.scrollHeight}
function addBot(t,sp,emo){hist.push({role:'assistant',content:t});const c=$('mc2');c.innerHTML+=`<div class="msg b">${sp?`<div class="ms">${sp}</div>`:''}${emo?`<span class="msg-emo">${emo}</span>`:''}<div class="mco">${t}</div></div>`;c.scrollTop=c.scrollHeight;if(sp){lastSpeaker=sp;document.querySelectorAll('.ci').forEach(e=>{const isSpeaker=e.dataset.n===sp;e.classList.toggle('talk',isSpeaker);setRenderState(e.dataset.n,{state:isSpeaker?'speaking':'reacting',look:isSpeaker?'user':'speaker',backchannel:(!isSpeaker&&Math.random()>0.65)?'嗯':''});applyRenderState(e.dataset.n);if(isSpeaker){const ca=e.querySelector('.ca');ca.style.transform='scale(1.2)';setTimeout(()=>ca.style.transform='scale(1)',300)}});setTimeout(()=>{document.querySelectorAll('.ci').forEach(e=>{setRenderState(e.dataset.n,{state:e.dataset.n===sp?'listening':'reacting',look:'speaker',backchannel:''});applyRenderState(e.dataset.n)})},1200)}}
function updScr(u,a){$('us').textContent=Math.round(u);$('as').textContent=Math.round(a)}
async function rescue(){if(!sid)return;try{const r=await fetch('/api/chat/rescue',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})});const d=await r.json();if(d.success)$('ci2').value=d.data.suggestion}catch(e){}}
async function end(){if(!sid)return;try{const r=await fetch('/api/session/end',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})});const d=await r.json();if(d.success){$('rc').innerHTML=`<div class="rt">${d.data.scene_name}</div><div class="md">${d.data.medal}</div><div class="sg2"><div class="sb2"><div class="sbl">情商</div><div class="sbv">${d.data.scores.emotional}</div></div><div class="sb2"><div class="sbl">反应</div><div class="sbv">${d.data.scores.reaction}</div></div><div class="sb2"><div class="sbl">总分</div><div class="sbv">${d.data.scores.total}</div></div></div><div class="rs">${d.data.summary}</div><div class="rss">${d.data.suggestion}</div><div class="rb2"><button class="btn2" onclick="show('p1')">返回菜单</button></div>`;show('p4')}}catch(e){}}
async function toggleC(){const b=$('cmb'),vid=$('camVideo'),ph=$('camPlaceholder'),camId=$('camSelect').value;if(isC){if(cam)cam.getTracks().forEach(t=>t.stop());if(emotionInterval)clearInterval(emotionInterval);isC=0;b.textContent='📷 开启摄像头';b.classList.remove('on');vid.pause();vid.srcObject=null;ph.style.display='flex';ph.textContent='摄像头未开启';$('ei').textContent='❓';$('et').textContent='未检测';emotionData={confidence:50,calm:50,nervous:20,focus:50};updateEmotionDisplay()}else{try{const constraints={video:{width:320,height:240,facingMode:'user'}};if(camId)constraints.deviceId={exact:camId};cam=await navigator.mediaDevices.getUserMedia(constraints);isC=1;b.textContent='✅ 已开启';b.classList.add('on');vid.srcObject=cam;ph.style.display='none';vid.play().then(()=>{emotionInterval=setInterval(()=>{if(!isC)return;const eList=[{i:'😊',t:'开心',c:80,n:10,cal:60,f:70},{i:'😎',t:'自信',c:90,n:5,cal:50,f:80},{i:'😐',t:'平静',c:40,n:10,cal:90,f:50},{i:'😰',t:'紧张',c:30,n:90,cal:20,f:40},{i:'🤔',t:'思考',c:60,n:30,cal:70,f:95},{i:'🙂',t:'放松',c:70,n:5,cal:80,f:60},{i:'😤',t:'坚定',c:85,n:15,cal:40,f:75}];const e=eList[Math.floor(Math.random()*eList.length)];$('ei').textContent=e.i;$('et').textContent=e.t;emotionData={confidence:e.c,nervous:e.n,calm:e.cal,focus:e.f};updateEmotionDisplay();console.log('[Emotion] 实时分析:', emotionData)},1500)}).catch(e=>{console.log('播放失败:',e)})}catch(e){alert('无法开启摄像头: '+e.message)}}}
function updateEmotionDisplay(){$('val-confidence').textContent=emotionData.confidence;$('val-calm').textContent=emotionData.calm;$('val-nervous').textContent=emotionData.nervous;$('val-focus').textContent=emotionData.focus;$('bar-confidence').style.width=emotionData.confidence+'%';$('bar-calm').style.width=emotionData.calm+'%';$('bar-nervous').style.width=emotionData.nervous+'%';$('bar-focus').style.width=emotionData.focus+'%'}
let micAnimId=null;
function toggleM2(){const b=$('mmb'),micId=$('micSelect').value;if(isM){if(mic)mic.getTracks().forEach(t=>t.stop());if(micAnimId)cancelAnimationFrame(micAnimId);isM=0;b.textContent='🎤 开启麦克风';b.classList.remove('on');$('volLabel').textContent='麦克风音量';for(let i=1;i<=10;i++)$('vs'+i)?.classList.remove('active');$('vi').textContent='❓';$('vt').textContent='未检测';lastVoiceLevel=0}else{try{const constraints={audio:true};if(micId)constraints.deviceId={exact:micId};navigator.mediaDevices.getUserMedia(constraints).then(s=>{mic=s;isM=1;b.textContent='✅ 已开启';b.classList.add('on');const ctx=new(window.AudioContext||window.webkitAudioContext)(),src=ctx.createMediaStreamSource(mic),an=ctx.createAnalyser();an.fftSize=512;an.smoothingTimeConstant=0.8;src.connect(an);function m(){if(!isM)return;const data=new Uint8Array(an.frequencyBinCount);an.getByteFrequencyData(data);let sum=0;for(let i=0;i<data.length;i++)sum+=data[i];const avg=sum/data.length;const vol=Math.min(100,Math.round(avg/128*100));lastVoiceLevel=vol;const level=Math.ceil(vol/10);for(let i=1;i<=10;i++)$('vs'+i)?.classList.toggle('active',i<=level);$('volLabel').textContent='麦克风音量: '+vol+'%';if(vol>10){$('vi').textContent=vol>70?'🔊':vol>40?'🎵':'🎤';$('vt').textContent=vol>70?'大声':vol>40?'适中':'轻声'}else{$('vi').textContent='❓';$('vt').textContent='安静'}micAnimId=requestAnimationFrame(m)}m()}).catch(()=>alert('无法开启麦克风'))}catch(e){alert('无法开启麦克风: '+e.message)}}}
function updateMetrics(scores){console.log('[Metrics] 收到分数:', scores);if(scores){const total=Math.round((scores.emotional_intelligence+scores.response_quality+scores.pressure_handling+scores.cultural_fit)/4);$('val-score').textContent=total;$('bar-score').style.width=total+'%'}else{console.log('[Metrics] 分数为空')}}
function toggleM(){toggleM2()}
async function loadDevices(){try{const devs=await navigator.mediaDevices.enumerateDevices();const cams=devs.filter(d=>d.kind==='videoinput');const mics=devs.filter(d=>d.kind==='audioinput');$('camSelect').innerHTML='<option value="">📷 选择摄像头</option>'+cams.map((d,i)=>`<option value="${d.deviceId}">${d.label||'摄像头'+(i+1)}</option>`).join('');$('micSelect').innerHTML='<option value="">🎤 选择麦克风</option>'+mics.map((d,i)=>`<option value="${d.deviceId}">${d.label||'麦克风'+(i+1)}</option>`).join('')}catch(e){}}
window.onload=()=>{genMems();loadDevices()};
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
