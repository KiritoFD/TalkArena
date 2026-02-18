"""
核心处理器
整合 Multi-Agent、RAG、决策引擎、防幻觉机制
"""

from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("TalkArena")


@dataclass
class ProcessResult:
    stage: str
    data: Dict = None
    error: str = None


class TalkArenaEngine:
    """TalkArena 核心引擎 - 整合所有高级技术"""

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

        self.sessions: Dict[str, Dict] = {}
        self.scenarios: Dict[str, Dict] = self._load_scenarios()

        logger.info("TalkArenaEngine 初始化完成")
        logger.info("- Multi-Agent: 已启用")
        logger.info("- RAG知识库: 已启用")
        logger.info("- 决策引擎: 已启用")
        logger.info("- 防幻觉机制: 已启用")

    def _load_scenarios(self) -> Dict:
        """加载场景配置"""
        return {
            "shandong_dinner": {
                "name": "山东人的饭桌",
                "characters": [
                    {"name": "大舅", "avatar": "👴", "bio": "主陪，德高望重的长辈"},
                    {"name": "大妗子", "avatar": "👵", "bio": "旁观者，明劝实激"},
                    {"name": "表哥", "avatar": "👨", "bio": "副陪，起哄能手"},
                ],
            }
        }

    def start_session(
        self,
        scenario_id: str,
        characters: Optional[List[Dict]] = None,
        scene_name: str = "",
    ) -> str:
        """开始会话"""
        from core.validators.output_validator import OutputValidator

        import uuid

        session_id = str(uuid.uuid4())[:8]

        scenario = self.scenarios.get(scenario_id, {}).copy()
        if characters:
            scenario["characters"] = characters

        self.sessions[session_id] = {
            "scenario_id": scenario_id,
            "scenario": scenario,
            "scene_name": scene_name or scenario.get("name", "TalkArena"),
            "turn_count": 0,
            "dominance": {"user": 50, "ai": 50},
            "history": [],
            "scores_history": [],
        }

        self.validators[session_id] = OutputValidator(scenario.get("characters", []))

        logger.info(f"会话创建: {session_id}")
        return session_id

    def process_turn(
        self, session_id: str, user_input: str, multimodal: Dict = None
    ) -> Generator[ProcessResult, None, None]:
        """处理一轮对话 - 多Agent协同"""
        if session_id not in self.sessions:
            yield ProcessResult("error", error="会话不存在")
            return

        session = self.sessions[session_id]

        yield ProcessResult("stage_analysis", data={"message": "分析用户输入..."})

        analysis = self.decision_engine.analyze_input(
            user_input,
            {"dominance": session["dominance"], "turn_count": session["turn_count"]},
        )

        yield ProcessResult("stage_rag", data={"message": "检索知识库..."})

        rag_context = self.rag_engine.enhance_context(
            user_input, {"intent": analysis["intent"], "topics": analysis["topics"]}
        )

        yield ProcessResult("stage_planning", data={"message": "规划响应策略..."})

        decisions = self.decision_engine.make_decision(
            {
                "dominance": session["dominance"],
                "turn_count": session["turn_count"],
                "intent": analysis["intent"],
                "multimodal": {"available": multimodal is not None},
            }
        )

        yield ProcessResult("stage_generation", data={"message": "生成AI响应..."})

        context = {
            "user_input": user_input,
            "characters": session["scenario"].get("characters", []),
            "turn_count": session["turn_count"],
            "dominance": session["dominance"],
            "multimodal": multimodal,
            "rag_knowledge": rag_context.get("rag_knowledge", ""),
            "strategies": analysis["strategies"],
        }

        result = self.multi_agent.process_turn(context)

        yield ProcessResult("stage_validation", data={"message": "验证输出..."})

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
            }
        )
        if "scores" in result:
            session["scores_history"].append(result["scores"])

        yield ProcessResult(
            "complete",
            data={
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
        """获取救场建议"""
        if session_id not in self.sessions:
            return "会话不存在"

        session = self.sessions[session_id]

        context = {
            "user_input": session["history"][-1].get("user", "")
            if session["history"]
            else "",
            "ai_response": session["history"][-1].get("ai", "")
            if session["history"]
            else "",
            "dominance": session["dominance"],
            "turn_count": session["turn_count"],
        }

        return self.multi_agent.get_rescue_suggestion(context)

    def end_session(self, session_id: str) -> Dict:
        """结束会话并生成报告"""
        if session_id not in self.sessions:
            return {"error": "会话不存在"}

        session = self.sessions[session_id]

        avg_scores = {}
        if session["scores_history"]:
            for key in session["scores_history"][0].keys():
                values = [s.get(key, 50) for s in session["scores_history"]]
                avg_scores[key] = sum(values) / len(values)

        total_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 50

        medal = self._determine_medal(total_score)

        summary = self._generate_summary(session, avg_scores)
        suggestion = self._generate_suggestion(avg_scores)

        return {
            "scene_name": session["scene_name"],
            "turn_count": session["turn_count"],
            "medal": medal,
            "scores": {
                "emotional": round(avg_scores.get("emotional_intelligence", 50)),
                "reaction": round(avg_scores.get("response_quality", 50)),
                "total": round(total_score),
            },
            "summary": summary,
            "suggestion": suggestion,
            "npc_os_list": self._generate_npc_thoughts(session),
        }

    def _determine_medal(self, score: float) -> str:
        """确定勋章"""
        if score >= 85:
            return "🏆 酒桌王者"
        elif score >= 70:
            return "🥇 情商高手"
        elif score >= 55:
            return "🥈 应变达人"
        elif score >= 40:
            return "🥉 初出茅庐"
        else:
            return "💔 需要修炼"

    def _generate_summary(self, session: Dict, scores: Dict) -> str:
        """生成总结"""
        turn_count = session["turn_count"]
        final_dominance = session["dominance"]["user"]

        if final_dominance >= 70:
            return f"经过{turn_count}轮较量，你以{final_dominance}分的气场压制全场，展现了出色的酒桌应变能力！"
        elif final_dominance >= 50:
            return f"经过{turn_count}轮较量，你稳住了局面，气场值{final_dominance}分，表现中规中矩。"
        else:
            return f"经过{turn_count}轮较量，你稍显被动，气场值{final_dominance}分，还需要多加练习。"

    def _generate_suggestion(self, scores: Dict) -> str:
        """生成建议"""
        suggestions = []

        eq = scores.get("emotional_intelligence", 50)
        if eq < 50:
            suggestions.append("多使用敬语和感谢词，提升情商表现")

        resp = scores.get("response_quality", 50)
        if resp < 50:
            suggestions.append("回答可以更有条理，适当使用转折词")

        if not suggestions:
            suggestions.append("整体表现不错，继续保持！")

        return suggestions[0]

    def _generate_npc_thoughts(self, session: Dict) -> List[Dict]:
        """生成NPC内心OS"""
        characters = session["scenario"].get("characters", [])
        thoughts = []

        for char in characters:
            name = char.get("name", "NPC")

            if session["dominance"]["user"] >= 70:
                thought = "这年轻人有两下子，不得不服！"
            elif session["dominance"]["user"] >= 50:
                thought = "还行，能应付得来。"
            else:
                thought = "还是太嫩了点，得多练练。"

            thoughts.append(
                {"name": name, "avatar": char.get("avatar", "👤"), "thought": thought}
            )

        return thoughts
