from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
import json
import logging
import random
import re

logger = logging.getLogger("MultiAgent")


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    DIALOGUE = "dialogue"
    EVALUATOR = "evaluator"
    RESCUER = "rescuer"
    MEMORY = "memory"


@dataclass
class AgentMessage:
    role: AgentRole
    content: str
    metadata: Dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class AgentState:
    context: Dict = field(default_factory=dict)
    history: List[AgentMessage] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)


class BaseAgent:
    def __init__(self, role: AgentRole, llm=None):
        self.role = role
        self.llm = llm
        self.state = AgentState()

    def think(self, context: Dict) -> AgentMessage:
        raise NotImplementedError

    def update_state(self, message: AgentMessage):
        self.state.history.append(message)


class EmpatheticDialogueAgent(BaseAgent):
    """Compatibility wrapper; delegates to DialogueAgent."""

    def __init__(self, llm=None):
        super().__init__(AgentRole.DIALOGUE, llm)
        self._delegate = DialogueAgent(llm)

    def think(self, context: Dict) -> AgentMessage:
        return self._delegate.think(context)


class DialogueAgent(BaseAgent):
    SCENES = {
        "shandong_dinner": {
            "name": "山东家庭饭桌",
            "atmosphere": "热闹、熟人压力强、讲面子和分寸，话里常带人情账。",
            "rhetoric": "多用对偶和重复，短句推进，口气接地气。",
            "word_limit": 56,
            "characters": {
                "大舅": {
                    "personality": "长辈主桌，重礼数，先关心后施压。",
                    "style": "一句抬人，一句压人；有亲近感但不失权威。",
                },
                "大姑": {
                    "personality": "会圆场，擅观察，擅软性引导。",
                    "style": "语气柔和，先顺着说，再轻推一步。",
                },
                "表哥": {
                    "personality": "活跃气氛，推进节奏。",
                    "style": "直接利落，可打趣但不刻薄。",
                },
            },
            "openings": {
                "大舅": "来，先坐稳先吃菜，酒慢慢来，话一条一条说。",
                "大姑": "先别紧张，先吃口热菜，咱慢慢聊、细细聊。",
                "表哥": "不急着拼酒，先碰个轻的，边吃边把事说透。",
            },
            "fallback": "行，咱慢慢聊，你先说你最在意的一点。",
        },
        "business_dinner": {
            "name": "商务饭局谈判",
            "atmosphere": "表面轻松，底层在试探资源、边界和执行力。",
            "rhetoric": "先礼后实，先关系后条款；并列句强调可执行性。",
            "word_limit": 68,
            "characters": {
                "王总": {
                    "personality": "结果导向，重时效和确定性。",
                    "style": "先定目标、再压节点、最后问风险兜底。",
                },
                "李总": {
                    "personality": "善沟通，掌控节奏。",
                    "style": "客气但锋利，追问时间点和责任人。",
                },
                "周顾问": {
                    "personality": "偏风控，重合规和边界。",
                    "style": "先列约束，再给可落地路径。",
                },
            },
            "openings": {
                "王总": "先把目标对齐、再把节奏对齐，今晚我们只谈落地。",
                "李总": "咱们先把共识摆在桌上，再把分工写在纸上。",
                "周顾问": "先说红线，再说方案；边界清楚，合作才稳。",
            },
            "fallback": "这个方向可以，先把时间表和责任边界说清楚。",
        },
        "interview": {
            "name": "高压结构化面试",
            "atmosphere": "节奏快、追问深，重证据链和复盘能力。",
            "rhetoric": "结论前置，三段并列，少形容多事实。",
            "word_limit": 78,
            "characters": {
                "主面试官": {
                    "personality": "判断严格，关注抗压和清晰度。",
                    "style": "追问关键细节，不接受空泛答案。",
                },
                "HR": {
                    "personality": "看匹配度与稳定沟通。",
                    "style": "温和提问，但直击价值观与协作方式。",
                },
                "技术负责人": {
                    "personality": "看技术深度与工程落地。",
                    "style": "要场景、要指标、要取舍。",
                },
            },
            "openings": {
                "主面试官": "我们直接进入正题：先给结论，再给证据，再给复盘。",
                "HR": "先放轻松，我们看三点：匹配度、稳定性、协作感。",
                "技术负责人": "请用一个真实项目展开，重点讲权衡与结果。",
            },
            "fallback": "请你用 STAR 结构，给一个可量化结果的例子。",
        },
        "debate": {
            "name": "立场攻防辩论",
            "atmosphere": "观点对撞，强调定义、证据、反驳链完整。",
            "rhetoric": "先定义后论证，反问与排比结合，但不跑题。",
            "word_limit": 88,
            "characters": {
                "正方辩手": {
                    "personality": "立场鲜明，强调收益与可行性。",
                    "style": "先立论、后证据、再反击。",
                },
                "反方辩手": {
                    "personality": "擅拆前提和逻辑漏洞。",
                    "style": "抓定义、举反例、迫使收敛命题。",
                },
                "点评席": {
                    "personality": "中立，重逻辑一致性。",
                    "style": "指出跳步、偷换概念、证据断点。",
                },
            },
            "openings": {
                "正方辩手": "我先明确命题边界，再给证据链，最后回应反驳点。",
                "反方辩手": "先别急着下结论，先看前提是否成立、证据是否闭环。",
                "点评席": "本轮只看三件事：定义是否清楚、证据是否有效、推理是否连贯。",
            },
            "fallback": "你的结论有方向，但证据链还不够闭环。",
        },
    }

    def __init__(self, llm=None):
        super().__init__(AgentRole.DIALOGUE, llm)

    def _resolve_scene(self, scenario_id: str) -> Dict:
        return self.SCENES.get(scenario_id, self.SCENES["shandong_dinner"])

    def _resolve_speaker(self, context: Dict) -> str:
        chars = context.get("characters") or []
        turn_count = int(context.get("turn_count", 0))
        if not chars:
            return "NPC"
        speaker = chars[turn_count % len(chars)]
        return speaker.get("name") or speaker.get("n") or "NPC"

    def _emotion_hint(self, multimodal: Dict) -> str:
        emo = multimodal.get("emotion", {}) if isinstance(multimodal, dict) else {}
        nervous = int(emo.get("nervous", 20))
        confidence = int(emo.get("confidence", 50))
        focus = int(emo.get("focus", 50))
        calm = int(emo.get("calm", 50))

        hints = []
        if nervous >= 70:
            hints.append("用户紧张，先稳情绪，再推进核心问题")
        if confidence <= 30:
            hints.append("用户底气弱，给台阶并给可执行下一步")
        if focus <= 35:
            hints.append("用户分心，收束到一个关键点")
        if confidence >= 75 and focus >= 70:
            hints.append("用户状态好，可提升问题深度")
        if calm >= 70 and nervous <= 30:
            hints.append("氛围平稳，可进入细节追问")
        return "；".join(hints) if hints else "按当前节奏自然推进"

    def _opening_line(self, scene: Dict, speaker_name: str) -> str:
        openings = scene.get("openings", {})
        return openings.get(speaker_name) or next(iter(openings.values()), scene.get("fallback", "我们开始吧。"))

    def _build_prompt(self, context: Dict, speaker_name: str, scene: Dict) -> str:
        user_input = context.get("user_input", "")
        scene_desc = context.get("scene_description", "")
        user_info = context.get("user_info") or {}
        char_info = scene.get("characters", {}).get(speaker_name, {})
        multimodal = context.get("multimodal", {})

        word_limit = int(scene.get("word_limit", 60))
        personality = char_info.get("personality", f"你是{speaker_name}")
        style = char_info.get("style", "自然对话")

        user_identity = ""
        if user_info:
            user_identity = (
                f"用户身份: {user_info.get('n', '用户')} / {user_info.get('r', '参与者')} / {user_info.get('b', '')}"
            )

        return (
            f"你在场景《{scene.get('name', '对话')}》扮演“{speaker_name}”。\n"
            f"场景氛围: {scene.get('atmosphere', '')}\n"
            f"修辞要求: {scene.get('rhetoric', '')}\n"
            f"角色设定: {personality}\n"
            f"说话风格: {style}\n"
            f"补充背景: {scene_desc}\n"
            f"{user_identity}\n"
            f"多模态提示: {self._emotion_hint(multimodal)}\n"
            f"用户刚说: {user_input}\n\n"
            "输出规则:\n"
            "1) 只输出NPC的一句话。\n"
            "2) 不要复述用户原话，不要出现引号包裹的用户台词。\n"
            "3) 不要输出角色名、旁白、系统提示。\n"
            "4) 句子要贴场景、可执行、有情绪分寸。\n"
            f"5) 长度不超过{word_limit}字。"
        )

    def _sanitize_model_reply(self, raw_text: str, user_input: str, speaker_name: str, max_chars: int) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""

        text = re.sub(rf"^\s*{re.escape(speaker_name)}\s*[：:]\s*", "", text)
        text = re.sub(r"^\s*(NPC|AI|assistant|角色)\s*[：:]\s*", "", text, flags=re.I)
        text = text.strip(" \t\r\n\"'“”‘’")

        lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
        if lines:
            text = lines[0]

        user_norm = re.sub(r"\s+", "", user_input or "")
        if user_norm:
            text = re.sub(
                r"[“\"']([^”\"']+)[”\"']",
                lambda m: "" if re.sub(r"\s+", "", m.group(1)) == user_norm else m.group(0),
                text,
            )
            if re.sub(r"\s+", "", text) == user_norm:
                text = ""

        for sep in ("。", "！", "？", "!", "?"):
            if sep in text:
                text = text.split(sep, 1)[0].strip() + sep
                break

        if len(text) > max_chars:
            text = text[:max_chars].rstrip("，,;；:： ") + "。"

        bad_markers = (
            "如用户说",
            "如果用户没有问题",
            "答案应为",
            "输出规则",
            "你在场景",
            "角色设定",
            "用户刚说",
            "请只输出",
        )
        if any(m in text for m in bad_markers):
            return ""

        return text.strip()

    def think(self, context: Dict) -> AgentMessage:
        scene = self._resolve_scene(context.get("scenario_id", "shandong_dinner"))
        speaker_name = self._resolve_speaker(context)
        user_input = (context.get("user_input", "") or "").strip()

        if not context.get("characters"):
            return AgentMessage(self.role, scene["fallback"], {"speaker": speaker_name}, 0.6)

        # Opening line per scene/speaker when session starts.
        if not user_input:
            return AgentMessage(
                self.role,
                self._opening_line(scene, speaker_name),
                {"speaker": speaker_name, "scenario": context.get("scenario_id", "shandong_dinner")},
                confidence=0.92,
            )

        if self.llm:
            try:
                prompt = self._build_prompt(context, speaker_name, scene)
                response = self.llm.generate(prompt, max_new_tokens=120, temperature=0.7)
                content = self._sanitize_model_reply(
                    response,
                    user_input=user_input,
                    speaker_name=speaker_name,
                    max_chars=int(scene.get("word_limit", 60)),
                )
                if not content:
                    content = scene["fallback"]
                return AgentMessage(
                    self.role,
                    content,
                    {"speaker": speaker_name, "scenario": context.get("scenario_id", "shandong_dinner")},
                    confidence=0.9,
                )
            except Exception as e:
                logger.error("[DialogueAgent] generation failed: %s", e)

        return AgentMessage(
            self.role,
            scene["fallback"],
            {"speaker": speaker_name, "scenario": context.get("scenario_id", "shandong_dinner")},
            confidence=0.6,
        )


class EvaluatorAgent(BaseAgent):
    EVAL_CRITERIA = {
        "emotional_intelligence": 0.35,
        "response_quality": 0.30,
        "pressure_handling": 0.20,
        "cultural_fit": 0.15,
    }

    def __init__(self, llm=None):
        super().__init__(AgentRole.EVALUATOR, llm)

    def think(self, context: Dict) -> AgentMessage:
        user_input = context.get("user_input", "")
        prev = context.get("dominance", {"user": 50, "ai": 50})
        multimodal = context.get("multimodal", {})

        scores = self._evaluate(user_input, multimodal)
        total = sum(scores[k] * w for k, w in self.EVAL_CRITERIA.items())
        if total >= 72:
            delta = 8
            judgment = "这一轮回应很稳，信息密度和分寸都在线。"
        elif total >= 58:
            delta = 3
            judgment = "这轮表现合格，继续提高证据和表达力度。"
        elif total >= 42:
            delta = -2
            judgment = "这轮有点被动，建议更直接地给结论。"
        else:
            delta = -8
            judgment = "这轮失分明显，需要快速重建结构化表达。"

        new_user = max(10, min(90, int(prev.get("user", 50)) + delta))
        new_ai = 100 - new_user

        return AgentMessage(
            self.role,
            judgment,
            {
                "scores": scores,
                "new_dominance": {"user": new_user, "ai": new_ai},
                "delta": delta,
            },
            confidence=0.85,
        )

    def _evaluate(self, user_input: str, multimodal: Dict) -> Dict[str, int]:
        text = (user_input or "").lower()
        emo = multimodal.get("emotion", {}) if isinstance(multimodal, dict) else {}

        eq = 55
        if any(w in text for w in ["谢谢", "理解", "明白", "麻烦"]):
            eq += 15
        if any(w in text for w in ["不行", "随便", "管不了"]):
            eq -= 20

        quality = 50
        if len(user_input) >= 12:
            quality += 10
        if any(w in text for w in ["因为", "所以", "首先", "其次"]):
            quality += 12

        pressure = 55 + int(emo.get("confidence", 50)) // 10 - int(emo.get("nervous", 20)) // 10
        culture = 55
        if any(w in text for w in ["您", "敬", "请"]):
            culture += 10

        def clip(v: int) -> int:
            return max(0, min(100, int(v)))

        return {
            "emotional_intelligence": clip(eq),
            "response_quality": clip(quality),
            "pressure_handling": clip(pressure),
            "cultural_fit": clip(culture),
        }


class RescuerAgent(BaseAgent):
    RESCUE_TEMPLATES = {
        "shandong_dinner": [
            "先接情绪再设边界：您说得对，我先敬茶，酒我少量慢来。",
            "先给台阶：我今天状态一般，先把这杯换成茶，咱照样把话聊透。",
        ],
        "business_dinner": [
            "先确认共同目标，再给时间表：这事我今晚给你一版可执行节点。",
            "先稳关系再谈条件：合作方向一致，细节我明早给你书面版。",
        ],
        "interview": [
            "用 STAR：先结论，再说情境-行动-结果，最后补复盘。",
            "先给指标：我做了X，结果Y提升Z%，复盘里我会改进A。",
        ],
        "debate": [
            "先定义争议点，再给两条证据，最后预判对方反驳。",
            "把命题收窄到可验证范围，不要泛化。",
        ],
    }

    def __init__(self, llm=None):
        super().__init__(AgentRole.RESCUER, llm)

    def think(self, context: Dict) -> AgentMessage:
        scenario_id = context.get("scenario_id", "shandong_dinner")
        user_input = context.get("user_input", "")
        ai_response = context.get("ai_response", "")

        if self.llm:
            try:
                prompt = (
                    f"场景:{scenario_id}\n用户:{user_input}\nNPC:{ai_response}\n"
                    "给一句30字内救场建议，只输出建议，不要解释。"
                )
                txt = (self.llm.generate(prompt, max_new_tokens=80, temperature=0.6) or "").strip()
                txt = re.sub(r"\s+", " ", txt)
                if txt:
                    return AgentMessage(self.role, txt[:60], {"scenario": scenario_id}, 0.9)
            except Exception:
                pass

        pool = self.RESCUE_TEMPLATES.get(scenario_id, self.RESCUE_TEMPLATES["shandong_dinner"])
        return AgentMessage(self.role, random.choice(pool), {"scenario": scenario_id}, 0.75)


class MemoryAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(AgentRole.MEMORY, llm)
        self.long_term_memory: Dict[str, Dict] = {}

    def think(self, context: Dict) -> AgentMessage:
        session_id = context.get("session_id", "default")
        action = context.get("memory_action", "retrieve")

        if action == "store":
            self._store_memory(session_id, context.get("turn_data", {}))
            return AgentMessage(self.role, "stored", confidence=1.0)

        memory = self._retrieve_memory(session_id)
        return AgentMessage(self.role, json.dumps(memory, ensure_ascii=False), {"memory": memory}, 1.0)

    def _store_memory(self, session_id: str, turn_data: Dict):
        if session_id not in self.long_term_memory:
            self.long_term_memory[session_id] = {"turns": [], "scores": []}
        self.long_term_memory[session_id]["turns"].append(turn_data)
        if "scores" in turn_data and turn_data["scores"]:
            self.long_term_memory[session_id]["scores"].append(turn_data["scores"])

    def _retrieve_memory(self, session_id: str) -> Dict:
        return self.long_term_memory.get(session_id, {"turns": [], "scores": []})


class MultiAgentOrchestrator:
    def __init__(self, llm=None):
        self.llm = llm
        self.agents = {
            AgentRole.DIALOGUE: DialogueAgent(llm),
            AgentRole.EVALUATOR: EvaluatorAgent(llm),
            AgentRole.RESCUER: RescuerAgent(llm),
            AgentRole.MEMORY: MemoryAgent(llm),
        }
        self.agents_list = [self.agents[AgentRole.DIALOGUE]]
        self.state = AgentState()

    def process_turn(self, context: Dict) -> Dict:
        memory_msg = self.agents[AgentRole.MEMORY].think({**context, "memory_action": "retrieve"})
        context["memory"] = json.loads(memory_msg.content) if memory_msg.content else {}

        dialogue_msg = self.agents[AgentRole.DIALOGUE].think(context)
        context["ai_response"] = dialogue_msg.content
        evaluator_msg = self.agents[AgentRole.EVALUATOR].think(context)

        turn_data = {
            "user_input": context.get("user_input", ""),
            "ai_response": dialogue_msg.content,
            "speaker": dialogue_msg.metadata.get("speaker"),
            "scores": evaluator_msg.metadata.get("scores"),
        }
        self.agents[AgentRole.MEMORY].think({**context, "memory_action": "store", "turn_data": turn_data})
        self.state.history.extend([memory_msg, dialogue_msg, evaluator_msg])

        return {
            "ai_response": dialogue_msg.content,
            "speaker": dialogue_msg.metadata.get("speaker"),
            "judgment": evaluator_msg.content,
            "scores": evaluator_msg.metadata.get("scores", {}),
            "new_dominance": evaluator_msg.metadata.get("new_dominance", {"user": 50, "ai": 50}),
            "game_over": self._check_game_over(evaluator_msg.metadata.get("new_dominance", {})),
        }

    def get_rescue_suggestion(self, context: Dict) -> str:
        return self.agents[AgentRole.RESCUER].think(context).content

    def _check_game_over(self, dominance: Dict) -> bool:
        user = int((dominance or {}).get("user", 50))
        return user <= 10 or user >= 90
