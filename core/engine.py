"""
Core runtime engine for TalkArena.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional
import logging
import uuid

logger = logging.getLogger("TalkArena")


@dataclass
class ProcessResult:
    stage: str
    data: Optional[Dict] = None
    error: Optional[str] = None


class TalkArenaEngine:
    def __init__(self, llm=None, enable_tts: bool = False):
        from core.agents.multi_agent import MultiAgentOrchestrator
        from core.rag.knowledge_base import RAGEngine
        from core.decision.engine import DecisionEngine
        from core.validators.output_validator import OutputValidator

        self.llm = llm
        self.tts = None
        self.enable_tts = enable_tts

        self.multi_agent = MultiAgentOrchestrator(llm)
        self.rag_engine = RAGEngine()
        self.decision_engine = DecisionEngine()
        self.validators: Dict[str, OutputValidator] = {}

        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.scenarios: Dict[str, Dict[str, Any]] = self._load_scenarios()

    def _load_scenarios(self) -> Dict[str, Dict[str, Any]]:
        return {
            "shandong_dinner": {
                "name": "家庭饭桌试炼",
                "description": "家庭饭桌高压社交，强调礼貌与边界感。",
                "characters": [
                    {"name": "大舅", "avatar": "host", "bio": "senior elder"},
                    {"name": "大妗子", "avatar": "observer", "bio": "detail observer"},
                    {"name": "表哥", "avatar": "reactor", "bio": "pace pusher"},
                ],
            },
            "business_dinner": {
                "name": "商务饭局谈判",
                "description": "合作前夜饭局，关注信任、边界和执行。",
                "characters": [
                    {"name": "王总", "avatar": "sponsor", "bio": "result oriented"},
                    {"name": "李总", "avatar": "bd", "bio": "deal pacing"},
                    {"name": "周顾问", "avatar": "risk", "bio": "risk oriented"},
                ],
            },
            "interview": {
                "name": "高压结构化面试",
                "description": "终面高压追问，强调结构化表达和证据。",
                "characters": [
                    {"name": "主面试官", "avatar": "lead", "bio": "decision quality"},
                    {"name": "HR", "avatar": "hr", "bio": "fit assessment"},
                    {"name": "技术负责人", "avatar": "tech", "bio": "technical depth"},
                ],
            },
            "debate": {
                "name": "立场攻防辩论",
                "description": "围绕公共议题进行定义、证据与反驳攻防。",
                "characters": [
                    {"name": "正方辩手", "avatar": "pro", "bio": "benefit argument"},
                    {"name": "反方辩手", "avatar": "con", "bio": "risk argument"},
                    {"name": "点评席", "avatar": "judge", "bio": "logic scrutiny"},
                ],
            },
        }

    def start_session(
        self,
        scenario_id: str,
        characters: Optional[List[Dict]] = None,
        scene_name: str = "",
        scene_description: str = "",
        user_info: Optional[Dict] = None,
    ) -> str:
        from core.validators.output_validator import OutputValidator

        sid = str(uuid.uuid4())[:8]
        scenario = dict(self.scenarios.get(scenario_id, self.scenarios["shandong_dinner"]))

        if characters:
            scenario["characters"] = characters
        if scene_description:
            scenario["description"] = scene_description
        if user_info:
            scenario["user_info"] = user_info

        self.sessions[sid] = {
            "scenario_id": scenario_id,
            "scenario": scenario,
            "scene_name": scene_name or scenario.get("name", "TalkArena"),
            "turn_count": 0,
            "dominance": {"user": 50, "ai": 50},
            "history": [],
            "scores_history": [],
        }
        self.validators[sid] = OutputValidator(scenario.get("characters", []))
        return sid

    def process_turn(
        self, session_id: str, user_input: str, multimodal: Optional[Dict] = None
    ) -> Generator[ProcessResult, None, None]:
        if session_id not in self.sessions:
            yield ProcessResult("error", error="session_not_found")
            return

        session = self.sessions[session_id]
        multimodal = multimodal or {}

        yield ProcessResult("stage_analysis", {"message": "analyzing"})
        analysis = self.decision_engine.analyze_input(
            user_input,
            {"dominance": session["dominance"], "turn_count": session["turn_count"]},
        )

        yield ProcessResult("stage_rag", {"message": "retrieving"})
        rag_context = self.rag_engine.enhance_context(
            user_input, {"intent": analysis["intent"], "topics": analysis["topics"]}
        )

        yield ProcessResult("stage_planning", {"message": "planning"})
        decisions = self.decision_engine.make_decision(
            {
                "dominance": session["dominance"],
                "turn_count": session["turn_count"],
                "intent": analysis["intent"],
                "multimodal": {"available": bool(multimodal)},
            }
        )

        yield ProcessResult("stage_generation", {"message": "generating"})
        context = {
            "user_input": user_input,
            "scenario_id": session.get("scenario_id", "shandong_dinner"),
            "characters": session["scenario"].get("characters", []),
            "turn_count": session["turn_count"],
            "dominance": session["dominance"],
            "multimodal": multimodal,
            "rag_knowledge": rag_context.get("rag_knowledge", ""),
            "strategies": analysis["strategies"],
            "scene_description": session["scenario"].get("description", ""),
            "user_info": session["scenario"].get("user_info"),
        }
        result = self.multi_agent.process_turn(context)

        validator = self.validators.get(session_id)
        if validator:
            result = validator.validate_and_correct(result)

        session["turn_count"] += 1
        session["dominance"] = result.get("new_dominance", session["dominance"])
        session["history"].append(
            {
                "user": user_input,
                "ai": result.get("ai_response", ""),
                "speaker": result.get("speaker"),
                "scores": result.get("scores"),
                "multimodal": multimodal or {},
            }
        )
        if "scores" in result:
            session["scores_history"].append(result["scores"])

        yield ProcessResult(
            "complete",
            {
                "ai_text": result.get("ai_response", ""),
                "speaker": result.get("speaker"),
                "judgment": result.get("judgment", ""),
                "scores": result.get("scores", {}),
                "new_dominance": session["dominance"],
                "game_over": result.get("game_over", False),
                "analysis": analysis,
                "decisions": decisions,
                "rag_used": bool(rag_context.get("rag_knowledge")),
            },
        )

    def get_rescue_suggestion(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return "会话不存在"
        session = self.sessions[session_id]
        last = session["history"][-1] if session["history"] else {}
        context = {
            "user_input": last.get("user", ""),
            "ai_response": last.get("ai", ""),
            "scenario_id": session.get("scenario_id", "shandong_dinner"),
            "scene_description": session["scenario"].get("description", ""),
            "user_info": session["scenario"].get("user_info"),
            "characters": session["scenario"].get("characters", []),
            "multimodal": last.get("multimodal", {}),
            "dominance": session["dominance"],
            "turn_count": session["turn_count"],
        }
        return self.multi_agent.get_rescue_suggestion(context)

    def end_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {"error": "session_not_found"}

        session = self.sessions[session_id]
        avg_scores: Dict[str, float] = {}
        if session["scores_history"]:
            keys = session["scores_history"][0].keys()
            for key in keys:
                vals = [s.get(key, 50) for s in session["scores_history"]]
                avg_scores[key] = sum(vals) / len(vals)

        total = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 50.0
        medal = self._determine_medal(total)
        summary = self._generate_summary(session)
        suggestion = self._generate_suggestion(avg_scores)

        return {
            "scene_name": session["scene_name"],
            "turn_count": session["turn_count"],
            "medal": medal,
            "scores": {
                "emotional": round(avg_scores.get("emotional_intelligence", 50)),
                "reaction": round(avg_scores.get("response_quality", 50)),
                "total": round(total),
            },
            "summary": summary,
            "suggestion": suggestion,
            "npc_os_list": self._generate_npc_thoughts(session),
        }

    def _determine_medal(self, score: float) -> str:
        if score >= 85:
            return "🥇"
        if score >= 70:
            return "🥈"
        if score >= 55:
            return "🥉"
        return "📘"

    def _generate_summary(self, session: Dict[str, Any]) -> str:
        turns = session["turn_count"]
        user_dom = session["dominance"].get("user", 50)
        if user_dom >= 70:
            return f"{turns}轮对话中你保持了主动节奏，表达稳定。"
        if user_dom >= 50:
            return f"{turns}轮对话中你整体稳住了局面。"
        return f"{turns}轮对话中你在压力下有波动，建议继续训练。"

    def _generate_suggestion(self, scores: Dict[str, float]) -> str:
        if not scores:
            return "下一轮建议：结论先行，给出1条可验证证据。"
        if scores.get("response_quality", 50) < 60:
            return "提升结构化表达：先结论，再证据，再复盘。"
        if scores.get("pressure_handling", 50) < 60:
            return "高压追问下放慢语速，优先回答问题主干。"
        return "保持当前节奏，增加量化细节让回答更有说服力。"

    def _generate_npc_thoughts(self, session: Dict[str, Any]) -> List[Dict[str, str]]:
        characters = session["scenario"].get("characters", [])
        thoughts = []
        user_dom = session["dominance"].get("user", 50)
        for char in characters:
            if user_dom >= 70:
                thought = "这个回答很稳，节奏掌控不错。"
            elif user_dom >= 50:
                thought = "有来有回，继续看后续发挥。"
            else:
                thought = "压力上来后出现犹豫，还能再提升。"
            thoughts.append({"name": char.get("name", "NPC"), "avatar": char.get("avatar", "npc"), "thought": thought})
        return thoughts
