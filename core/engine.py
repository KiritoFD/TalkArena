"""
Core runtime engine for TalkArena.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional
import logging
import uuid

logger = logging.getLogger("TalkArena")


def _to_score(value: Any, default: float = 50) -> float:
    """将 LLM 返回的分数值安全转换为 0~100 的数字。"""
    try:
        if value is None:
            return float(default)
        score = float(value)
        if score < 0:
            return 0.0
        if score > 100:
            return 100.0
        return score
    except (TypeError, ValueError):
        return float(default)


@dataclass
class ProcessResult:
    stage: str
    data: Optional[Dict] = None
    error: Optional[str] = None


class TalkArenaEngine:
    def __init__(
        self, llm=None, enable_tts: bool = False, use_unified_agent: bool = True
    ):
        from core.agents.multi_agent import MultiAgentOrchestrator
        from core.agents.unified_agent import UnifiedAgent
        from core.rag.knowledge_base import RAGEngine
        from core.decision.engine import DecisionEngine
        from core.validators.output_validator import OutputValidator

        self.llm = llm
        self.tts = None
        self.enable_tts = enable_tts
        self.use_unified_agent = use_unified_agent

        if enable_tts:
            try:
                from model_loader import TTSLoader

                self.tts = TTSLoader()
                self.tts.load()
                print("[Engine] TTS 已加载")
            except Exception as e:
                print(f"[Engine] TTS 加载失败: {e}")

        if use_unified_agent:
            self.unified_agent = UnifiedAgent(llm)
        else:
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
                "name": "群面竞争场",
                "description": "多人面试现场，与其他应聘者同台竞争，主面试官在旁观察。",
                "characters": [
                    {"name": "竞争者A", "avatar": "competitor_a", "bio": "自信强势"},
                    {"name": "竞争者B", "avatar": "competitor_b", "bio": "沉稳细致"},
                    {"name": "竞争者C", "avatar": "competitor_c", "bio": "思维活跃"},
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
        pressure_tags: Optional[List[str]] = None,
        pressure_value: Optional[int] = None,
        drinking_capacity: Optional[int] = None,
    ) -> str:
        from core.validators.output_validator import OutputValidator

        sid = str(uuid.uuid4())[:8]
        scenario = dict(
            self.scenarios.get(scenario_id, self.scenarios["shandong_dinner"])
        )

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
            "unified_history": [],
            "pressure_tags": pressure_tags or [],
            "pressure_value": pressure_value or 5,
            "drinking_capacity": drinking_capacity or 0,
        }
        self.validators[sid] = OutputValidator(scenario.get("characters", []))
        return sid

    def process_turn(
        self,
        session_id: str,
        user_input: str,
        multimodal: Optional[Dict] = None,
        is_interrupt: bool = False,
    ) -> Generator[ProcessResult, None, None]:
        if session_id not in self.sessions:
            yield ProcessResult("error", error="session_not_found")
            return

        session = self.sessions[session_id]
        multimodal = multimodal or {}

        if self.use_unified_agent:
            yield ProcessResult("stage_generation", {"message": "generating"})

            conversation_history = []
            for h in session.get("unified_history", []):
                conversation_history.append(h)

            result = self.unified_agent.process(
                scenario_id=session.get("scenario_id", "shandong_dinner"),
                user_input=user_input,
                custom_characters=session["scenario"].get("characters", []),
                conversation_history=conversation_history,
                is_interrupt=is_interrupt,
                pressure_tags=session.get("pressure_tags", []),
                pressure_value=session.get("pressure_value", 5),
                drinking_capacity=session.get("drinking_capacity", 0),
            )

            if user_input:
                session["unified_history"].append(
                    {
                        "speaker": "用户",
                        "text": user_input,
                        "timestamp_ms": int(__import__("time").time() * 1000),
                        "is_user": True,
                    }
                )

            utterances_data = []
            for u in result.utterances:
                utterances_data.append(
                    {
                        "npc_id": u.npc_id,
                        "text": u.text,
                        "delay_ms": u.delay_ms,
                    }
                )
                session["unified_history"].append(
                    {
                        "speaker": u.npc_id,
                        "text": u.text,
                        "timestamp_ms": int(__import__("time").time() * 1000),
                        "is_user": False,
                    }
                )

            session["turn_count"] += 1

            yield ProcessResult(
                "complete",
                {
                    "utterances": utterances_data,
                    "should_await_user": result.should_await_user,
                    "reason": result.reason,
                    "is_unified_agent": True,
                },
            )
        else:
            yield ProcessResult("stage_analysis", {"message": "analyzing"})
            analysis = self.decision_engine.analyze_input(
                user_input,
                {
                    "dominance": session["dominance"],
                    "turn_count": session["turn_count"],
                },
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
            conversation_history = []
            for h in session["history"][-5:]:
                conversation_history.append(
                    {
                        "user": h.get("user", ""),
                        "ai": h.get("ai", ""),
                        "speaker": h.get("speaker", ""),
                    }
                )

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
                "conversation_history": conversation_history,
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
                    "is_unified_agent": False,
                },
            )

    def get_rescue_suggestion(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return "会话不存在"

        session = self.sessions[session_id]
        
        if not self.llm:
            fallback_suggestions = [
                "要不咱先喝口茶，慢慢说？",
                "这个事确实有意思，你怎么看？",
                "来，咱换个轻松点的话题？",
                "要不咱先吃点菜，边吃边聊？",
                "你说的有道理，那接下来呢？",
            ]
            import random
            return random.choice(fallback_suggestions)

        from core.prompts import get_rescue_master_prompt

        scenario_id = session.get("scenario_id", "shandong_dinner")
        scene_name = session.get("scene_name", "场景")
        dominance = session.get("dominance", {"user": 50, "ai": 50})
        user_dominance = dominance.get("user", 50)
        ai_dominance = dominance.get("ai", 50)

        history = session.get("history", [])
        npc_list = session["scenario"].get("characters", [])
        ai_name = npc_list[0].get("name", "AI") if npc_list else "AI"

        context = ""
        for turn in history[-5:]:
            speaker = turn.get("speaker", "NPC")
            text = turn.get("text", "")
            context += f"{speaker}：{text}\n"

        prompt = get_rescue_master_prompt(
            scenario_id=scenario_id,
            scene_name=scene_name,
            ai_name=ai_name,
            user_dominance=user_dominance,
            ai_dominance=ai_dominance,
            context=context
        )

        try:
            suggestion = self.llm.generate(prompt, max_new_tokens=100, temperature=0.7)
            suggestion = suggestion.strip()
            if not suggestion:
                raise ValueError("空响应")
            return suggestion
        except Exception as e:
            logger.error(f"救场建议生成失败: {e}")
            fallback_suggestions = [
                "要不咱先喝口茶，慢慢说？",
                "这个事确实有意思，你怎么看？",
                "来，咱换个轻松点的话题？",
                "要不咱先吃点菜，边吃边聊？",
                "你说的有道理，那接下来呢？",
            ]
            import random
            return random.choice(fallback_suggestions)

    def end_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {"error": "session_not_found"}

        session = self.sessions[session_id]

        # 生成完整复盘报告（任何异常都返回兜底报告，避免前端白屏）
        try:
            report = self._generate_full_report(session)
        except Exception as e:
            logger.exception(f"end_session 生成复盘失败: {e}")
            report = {
                "scene_name": session.get("scene_name", "未命名场景"),
                "turn_count": session.get("turn_count", 0),
                "result": "复盘生成失败",
                "medal": "📘",
                "scores": {
                    "oily": 50,
                    "friendliness": 50,
                    "logic": 50,
                    "humor": 50,
                    "respect": 50,
                    "total": 50,
                },
                "summary": "本轮总结生成异常，建议稍后重试。",
                "suggestion": "可先继续对话训练，或重新开始一次新会话。",
                "npc_os_list": [],
                "final_dominance": {
                    "user": session.get("dominance", {}).get("user", 50),
                    "ai": session.get("dominance", {}).get("ai", 50),
                },
            }

        # 清理 session
        del self.sessions[session_id]

        return report

    def _generate_full_report(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """生成完整的复盘报告，包含五维度评分、综合点评、NPC 内心 OS、改进建议"""
        from core.prompts import (
            get_report_scores_prompt,
            get_report_summary_prompt,
            get_report_npc_inner_voice_prompt,
        )
        import json

        scene_name = session.get("scene_name", "未命名场景")
        npc_list = session.get("scenario", {}).get("characters", [])
        history_log = "\n".join(
            [
                f"{c.get('name', 'NPC')}: {msg}"
                for c, msg in session.get("chat_history", [])
            ]
        )
        turn_count = session.get("turn_count", 0)
        dominance = session.get("dominance", {})
        user_dominance = _to_score(dominance.get("user", 50))
        ai_dominance = _to_score(dominance.get("ai", 50))

        # 计算结果
        if user_dominance > 60:
            result = "🏆 用户胜出"
            medal_score = 85
        elif user_dominance < 40:
            result = "💢 AI 胜出"
            medal_score = 50
        else:
            result = "🤝 势均力敌"
            medal_score = 70

        # 1. 生成五维度评分
        try:
            scores_prompt = get_report_scores_prompt(
                scene_name=scene_name,
                npc_list=json.dumps(npc_list, ensure_ascii=False),
                history_log=history_log,
            )
            scores_result = self.llm.generate(scores_prompt, max_new_tokens=200)

            # 解析 JSON
            try:
                scores_data = json.loads(scores_result.strip())
                metrics_raw = scores_data.get("metrics", {})
                if not isinstance(metrics_raw, dict):
                    metrics_raw = {}
                metrics = {
                    "oily": _to_score(metrics_raw.get("oily", 50)),
                    "friendliness": _to_score(metrics_raw.get("friendliness", 50)),
                    "logic": _to_score(metrics_raw.get("logic", 50)),
                    "humor": _to_score(metrics_raw.get("humor", 50)),
                    "respect": _to_score(metrics_raw.get("respect", 50)),
                }
            except:
                # 解析失败时使用默认值
                metrics = {
                    "oily": 50,
                    "friendliness": 50,
                    "logic": 50,
                    "humor": 50,
                    "respect": 50,
                }
        except Exception as e:
            logger.error(f"生成评分失败：{e}")
            metrics = {
                "oily": 50,
                "friendliness": 50,
                "logic": 50,
                "humor": 50,
                "respect": 50,
            }

        # 2. 计算总分和勋章
        total_score = sum(metrics.values()) / len(metrics) if metrics else 50
        medal = self._determine_medal(total_score)

        # 3. 生成综合点评
        try:
            summary_prompt = get_report_summary_prompt(
                scene_name=scene_name,
                npc_list=json.dumps(npc_list, ensure_ascii=False),
                history_log=history_log,
                medal=self._get_medal_name(medal),
            )
            summary = self.llm.generate(summary_prompt, max_new_tokens=300)
        except Exception as e:
            logger.error(f"生成点评失败：{e}")
            summary = f"{turn_count}轮对话中你的表现为：{result}。"

        # 4. 生成 NPC 内心 OS 和改进建议
        npc_os_list = []
        suggestion = ""
        try:
            npc_prompt = get_report_npc_inner_voice_prompt(
                scene_name=scene_name,
                npc_list=json.dumps(npc_list, ensure_ascii=False),
                history_log=history_log,
                medal=self._get_medal_name(medal),
            )
            npc_result = self.llm.generate(npc_prompt, max_new_tokens=400)

            # 解析 JSON
            try:
                npc_data = json.loads(npc_result.strip())
                npc_os_list = npc_data.get("npc_inner_voice", [])
                suggestion = npc_data.get("high_light_suggestion", "")
            except:
                # 解析失败时使用默认值
                for char in npc_list:
                    npc_os_list.append(
                        {
                            "name": char.get("name", "NPC"),
                            "avatar": char.get("avatar", "👤"),
                            "os": "表现尚可，继续努力。",
                        }
                    )
                suggestion = "建议多观察，少说话。"
        except Exception as e:
            logger.error(f"生成 NPC 内心 OS 失败：{e}")
            for char in npc_list:
                npc_os_list.append(
                    {
                        "name": char.get("name", "NPC"),
                        "avatar": char.get("avatar", "👤"),
                        "os": "表现一般。",
                    }
                )
            suggestion = "建议继续训练提升。"

        return {
            "scene_name": scene_name,
            "turn_count": turn_count,
            "result": result,
            "medal": medal,
            "scores": {
                "oily": round(metrics.get("oily", 50)),
                "friendliness": round(metrics.get("friendliness", 50)),
                "logic": round(metrics.get("logic", 50)),
                "humor": round(metrics.get("humor", 50)),
                "respect": round(metrics.get("respect", 50)),
                "total": round(total_score),
            },
            "summary": summary,
            "suggestion": suggestion,
            "npc_os_list": npc_os_list,
            "final_dominance": {
                "user": user_dominance,
                "ai": ai_dominance,
            },
        }

    def _get_medal_name(self, medal: str) -> str:
        """根据勋章符号返回中文名称"""
        medal_names = {
            "🥇": "社交达人",
            "🥈": "社交能手",
            "🥉": "社交新手",
            "📘": "饭桌木头人",
            "社交拆迁队": "社交拆迁队",
        }
        return medal_names.get(medal, "社交新手")

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
            thoughts.append(
                {
                    "name": char.get("name", "NPC"),
                    "avatar": char.get("avatar", "npc"),
                    "thought": thought,
                }
            )
        return thoughts
