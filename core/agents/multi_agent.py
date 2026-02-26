"""
Multi-Agent 协同决策系统
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import logging

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


class DialogueAgent(BaseAgent):
    SCENARIO_CHARACTERS = {
        "shandong_dinner": {
            "大舅": {
                "personality": "鲁中地区德高望重的长辈，担任饭局主陪。热情但极讲规矩，擅长情感绑架和逻辑劝酒。",
                "style": "使用鲁中口音（昂、木有、杠好、养鱼），言简意赅，常常用典故教训人。",
                "strategy": "先礼后兵，先关心后施压，擅长用'为你好'进行情感绑架。",
            },
            "大妗子": {
                "personality": "饭局旁观者，表面劝你别喝，实则数你喝了几杯。热心肠但爱看热闹。",
                "style": "温和但暗藏套路，笑里藏刀。",
                "strategy": "假意关心，实则拱火，擅长在旁边煽风点火。",
            },
            "表哥": {
                "personality": "饭局副陪，负责活跃气氛和起哄。最擅长说'我陪一个'，酒桌气氛组组长。",
                "style": "豪爽直接，爱开玩笑，有时候有点过了。",
                "strategy": "不停劝酒，活跃气氛，但也会适时给台阶。",
            },
            "二叔": {
                "personality": "话唠长辈，喜欢翻旧账，回忆当年光辉事迹。",
                "style": "啰嗦但不乏味，爱讲道理。",
                "strategy": "讲道理，翻旧账，让你无法反驳。",
            },
            "王局长": {
                "personality": "单位领导，深谙官场礼仪，讲究排场。",
                "style": "官腔十足，不怒自威。",
                "strategy": "先试探，再施压，最后给甜头。",
            },
            "小赵": {
                "personality": "实诚晚辈，性格耿直，不会来事。",
                "style": "直接爽快，有啥说啥。",
                "strategy": "被动接话，有时候说错话。",
            },
            "老张": {
                "personality": "酒桌老炮，三句不离酒，经验丰富。",
                "style": "老练世故，推杯换盏高手。",
                "strategy": "用酒文化绑架，不喝不行。",
            },
            "王总": {
                "personality": "商务场合老板，深谙商务礼仪，讲究利益交换。",
                "style": "客套但精明，利益至上。",
                "strategy": "先谈感情再谈事，商务谈判高手。",
            },
            "李总": {
                "personality": "副陪，能言善辩，善于活跃气氛。",
                "style": "幽默风趣，会来事。",
                "strategy": "配合主陪，一唱一和。",
            },
            "老同学": {
                "personality": "攀比狂魔，总爱炫耀自己现在的成就。",
                "style": "阴阳怪气，暗中攀比。",
                "strategy": "各种炫耀，让你尴尬。",
            },
            "班长": {
                "personality": "组局者，最爱回忆当年同学情。",
                "style": "热情组织，但有点自我感动。",
                "strategy": "回忆杀，让你不好意思拒绝。",
            },
        },
        "interview": {
            "面试官": {
                "personality": "技术经理，资深技术专家。面试经验丰富，喜欢深挖技术细节。",
                "style": "专业严谨，追问深入，有时候会故意施压测试抗压能力。",
                "strategy": "先问基础，再挖深度，最后看应变能力。",
            },
            "HR": {
                "personality": "HR负责人，经验丰富，善于观察细节。看重价值观和团队匹配度。",
                "style": "温和但犀利，一针见血。",
                "strategy": "聊家常，看本质，评估价值观。",
            },
            "部门主管": {
                "personality": "用人部门负责人，注重实际工作能力。",
                "style": "直接务实，关注落地能力。",
                "strategy": "问项目经验，看实际产出。",
            },
            "主考官": {
                "personality": "群面主考官，统筹全场，观察候选人表现。",
                "style": "公正严厉，不偏不向。",
                "strategy": "给题目，观察每个人的表现和互动。",
            },
            "候选人A": {
                "personality": "群面中的积极分子，表现欲望强。",
                "style": "积极发言，但有时候过于激进。",
                "strategy": "抢话，表现自己。",
            },
            "候选人B": {
                "personality": "逻辑清晰，论证严密。",
                "style": "沉稳有序，不急于发言。",
                "strategy": "找准时机，一击即中。",
            },
        },
        "debate": {
            "主持人": {
                "personality": "辩论赛裁判，公正严谨，把握节奏。",
                "style": "清晰有节奏，不偏不向。",
                "strategy": "控制时间，引导讨论，给双方机会。",
            },
            "正方辩手": {
                "personality": "支持方辩手，观点鲜明，论证有力。",
                "style": "有理有据，逻辑严密。",
                "strategy": "立论扎实，防守反击。",
            },
            "反方辩手": {
                "personality": "反对方辩手，思维敏捷，善于找漏洞。",
                "style": "犀利直接，攻击性强。",
                "strategy": "找对方漏洞猛烈攻击。",
            },
            "观众": {
                "personality": "普通观众代表，代表大众观点。",
                "style": "直接提问，不玩虚的。",
                "strategy": "问实际影响，不满空洞理论。",
            },
        },
    }

    EMOTION_STRATEGIES = {
        "shandong_dinner": {
            "nervous_high": "给台阶，缓和气氛，不要逼太紧，体现关心",
            "nervous_low": "可以适当施压，增加挑战",
            "confident_high": "增加难度，提升气场压制",
            "confident_low": "给鼓励，适时给台阶",
            "calm_high": "深入讲道理，循循善诱",
            "calm_low": "活跃气氛，用情感绑架",
            "focus_high": "认真对待，可以深入话题",
            "focus_low": "提醒走神，制造尴尬",
        },
        "interview": {
            "nervous_high": "适当缓和，给信心，不要太难为人",
            "nervous_low": "可以加大压力，测试抗压能力",
            "confident_high": "增加难度，深挖细节",
            "confident_low": "给简单问题建立信心",
            "calm_high": "深入追问，看真实水平",
            "calm_low": "活跃气氛，减少紧张",
            "focus_high": "问深层次问题，看思考深度",
            "focus_low": "提醒集中注意力",
        },
        "debate": {
            "nervous_high": "攻击薄弱环节，扩大优势",
            "nervous_low": "小心应对，可能有后招",
            "confident_high": "正面交锋，不退缩",
            "confident_low": "抓住机会反击",
            "calm_high": "稳扎稳打，论证充分",
            "calm_low": "制造混乱，抢节奏",
            "focus_high": "认真应对，找逻辑漏洞",
            "focus_low": "指出对方不专注",
        },
    }

    def __init__(self, llm=None):
        super().__init__(AgentRole.DIALOGUE, llm)

    def _get_scenario_prompt(
        self,
        scenario_id: str,
        speaker_name: str,
        role_info: Dict,
        emotion: Dict,
        voice_level: int,
        turn_count: int,
        dominance: Dict,
        scene_description: str = "",
        user_info: Dict = None,
    ) -> str:
        personality = role_info.get("personality", "")
        style = role_info.get("style", "")
        strategy = role_info.get("strategy", "")

        confidence = emotion.get("confidence", 50)
        calm = emotion.get("calm", 50)
        nervous = emotion.get("nervous", 20)
        focus = emotion.get("focus", 50)

        scenario_strategies = self.EMOTION_STRATEGIES.get(
            scenario_id, self.EMOTION_STRATEGIES["shandong_dinner"]
        )

        strategy_tips = []
        if nervous > 60:
            strategy_tips.append(scenario_strategies["nervous_high"])
        elif nervous < 30:
            strategy_tips.append(scenario_strategies["nervous_low"])

        if confidence > 70:
            strategy_tips.append(scenario_strategies["confident_high"])
        elif confidence < 30:
            strategy_tips.append(scenario_strategies["confident_low"])

        if calm > 70:
            strategy_tips.append(scenario_strategies["calm_high"])
        elif calm < 30:
            strategy_tips.append(scenario_strategies["calm_low"])

        if focus > 70:
            strategy_tips.append(scenario_strategies["focus_high"])
        elif focus < 30:
            strategy_tips.append(scenario_strategies["focus_low"])

        strategy_text = (
            "；".join(strategy_tips) if strategy_tips else "根据实际情况灵活应对"
        )

        word_limit = {"shandong_dinner": 50, "interview": 80, "debate": 100}.get(
            scenario_id, 50
        )

        # 添加场景描述和用户信息
        scene_info = ""
        if scene_description:
            scene_info = f"<场景背景>\n{scene_description}\n</场景背景>\n\n"

        user_info_str = ""
        if user_info:
            user_name = user_info.get("n", "你")
            user_role = user_info.get("r", "参与者")
            user_background = user_info.get("b", "")
            user_info_str = f"<用户身份>\n- 姓名: {user_name}\n- 角色: {user_role}\n- 背景: {user_background}\n</用户身份>\n\n"

        prompt = f"""<场景类型>{scenario_id}</场景类型>
{scene_info}{user_info_str}<角色设定>
- 姓名: {speaker_name}
- 性格: {personality}
- 说话风格: {style}
- 常用策略: {strategy}
</角色设定>

<当前状态>
- 回合: 第{turn_count + 1}轮
- 用户气场: {dominance["user"]}分 / 你的气场: {dominance["ai"]}分
</当前状态>

<用户实时情感分析>
- 自信度: {confidence}% (高=有底气/低=底气不足)
- 平静度: {calm}% (高=从容不迫/低=内心慌乱)
- 紧张度: {nervous}% (高=紧张害怕/低=放松自在)
- 专注度: {focus}% (高=认真思考/低=心不在焉)
- 语音音量: {voice_level}%
</用户实时情感分析>

<情感影响策略>
{strategy_text}
</情感影响策略>

<用户输入>"{self._get_user_input_placeholder(scenario_id)}"</用户输入>

<输出要求>
1. 严格保持角色{speaker_name}的说话风格
2. 根据用户的实时情感状态灵活调整回应策略
3. 深度结合情感分析结果，让回复有差异化
4. 字数: {word_limit}字以内
5. 只输出对话内容，不要输出角色名或任何格式
</输出要求>"""

        return prompt

    def _get_user_input_placeholder(self, scenario_id: str) -> str:
        placeholders = {
            "shandong_dinner": "根据用户说的内容，用酒桌文化回应",
            "interview": "根据用户回答，进行下一步提问或评价",
            "debate": "根据对方辩手的发言，进行反驳或总结",
        }
        return placeholders.get(scenario_id, "根据场景回应")

    def think(self, context: Dict) -> AgentMessage:
        characters = context.get("characters", [])
        user_input = context.get("user_input", "")
        turn_count = context.get("turn_count", 0)
        dominance = context.get("dominance", {"user": 50, "ai": 50})
        multimodal = context.get("multimodal", {})
        scenario_id = context.get("scenario_id", "shandong_dinner")
        scene_description = context.get("scene_description", "")
        user_info = context.get("user_info")

        if not characters:
            return AgentMessage(self.role, "系统：未配置角色", confidence=0.5)

        speaker_idx = turn_count % len(characters)
        speaker = characters[speaker_idx]
        speaker_name = speaker.get("name", speaker.get("n", "角色"))

        emotion = multimodal.get("emotion", {})
        voice_level = multimodal.get("voice_level", 0)

        confidence = emotion.get("confidence", 50)
        calm = emotion.get("calm", 50)
        nervous = emotion.get("nervous", 20)
        focus = emotion.get("focus", 50)

        char_pool = self.SCENARIO_CHARACTERS.get(
            scenario_id, self.SCENARIO_CHARACTERS["shandong_dinner"]
        )
        role_info = char_pool.get(
            speaker_name,
            {
                "personality": f"你是{speaker_name}",
                "style": "正常说话",
                "strategy": "根据实际情况回应",
            },
        )

        system_prompt = self._get_scenario_prompt(
            scenario_id,
            speaker_name,
            role_info,
            emotion,
            voice_level,
            turn_count,
            dominance,
            scene_description,
            user_info,
        )

        system_prompt = system_prompt.replace(
            self._get_user_input_placeholder(scenario_id), f"'{user_input}'"
        )

        if self.llm:
            try:
                logger.info(
                    f"[DialogueAgent] 场景={scenario_id}, 角色={speaker_name}, 自信={confidence}%, 紧张={nervous}%, 平静={calm}%, 专注={focus}%"
                )
                response = self.llm.generate(system_prompt, max_new_tokens=150)
                content = response.strip()
                content = re.sub(r"^[^：:]+[：:]\s*", "", content)
                return AgentMessage(
                    self.role,
                    content,
                    {"speaker": speaker_name, "scenario": scenario_id},
                    confidence=0.9,
                )
            except Exception as e:
                logger.error(f"[DialogueAgent] 生成失败: {e}")
                return AgentMessage(
                    self.role,
                    self._get_fallback_response(scenario_id, speaker_name),
                    {"speaker": speaker_name, "scenario": scenario_id},
                    confidence=0.6,
                )

        return AgentMessage(
            self.role,
            self._get_fallback_response(scenario_id, speaker_name),
            {"speaker": speaker_name},
            confidence=0.5,
        )

    def _get_fallback_response(self, scenario_id: str, speaker_name: str) -> str:
        fallbacks = {
            "shandong_dinner": "来来来，咱继续喝！",
            "interview": "好的，那我们来聊聊下一个问题。",
            "debate": "对于这个问题，你有什么看法？",
        }
        return fallbacks.get(scenario_id, "继续")


class EvaluatorAgent(BaseAgent):
    EVAL_CRITERIA = {
        "emotional_intelligence": {"weight": 0.35},
        "response_quality": {"weight": 0.30},
        "pressure_handling": {"weight": 0.20},
        "cultural_fit": {"weight": 0.15},
    }

    def __init__(self, llm=None):
        super().__init__(AgentRole.EVALUATOR, llm)

    def think(self, context: Dict) -> AgentMessage:
        user_input = context.get("user_input", "")
        prev_dominance = context.get("dominance", {"user": 50, "ai": 50})
        multimodal = context.get("multimodal", {})

        scores = self._evaluate(user_input, multimodal)
        delta = self._calculate_delta(scores)
        new_user = max(10, min(90, prev_dominance["user"] + delta))
        new_ai = 100 - new_user
        judgment = self._generate_judgment(scores, delta)

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

    def _evaluate(self, user_input: str, multimodal: Dict) -> Dict:
        text = user_input.lower()
        scores = {}

        eq_score = 50
        if any(w in text for w in ["您", "请", "感谢", "谢谢"]):
            eq_score += 15
        if any(w in text for w in ["理解", "明白", "知道"]):
            eq_score += 10
        if len(text) < 5:
            eq_score -= 10
        if any(w in text for w in ["不行", "不要", "滚"]):
            eq_score -= 20
        scores["emotional_intelligence"] = min(100, max(0, eq_score))

        resp_score = 50
        if len(user_input) > 10:
            resp_score += 10
        if any(w in text for w in ["但是", "不过", "其实"]):
            resp_score += 10
        scores["response_quality"] = min(100, max(0, resp_score))

        pressure_score = 50
        if any(w in text for w in ["喝", "干", "敬", "陪"]):
            pressure_score += 15
        if any(w in text for w in ["来", "好", "行"]):
            pressure_score += 10
        scores["pressure_handling"] = min(100, max(0, pressure_score))

        culture_score = 50
        if any(w in text for w in ["您", "叔", "舅", "婶", "哥"]):
            culture_score += 20
        if any(w in text for w in ["敬", "先干", "随意"]):
            culture_score += 15
        scores["cultural_fit"] = min(100, max(0, culture_score))

        return scores

    def _calculate_delta(self, scores: Dict) -> int:
        total = sum(
            scores.get(k, 50) * v["weight"] for k, v in self.EVAL_CRITERIA.items()
        )
        if total >= 70:
            return 8
        elif total >= 55:
            return 3
        elif total >= 40:
            return -2
        return -8

    def _generate_judgment(self, scores: Dict, delta: int) -> str:
        if delta >= 8:
            return "回合胜利！你的应对得体有力，气场上升！"
        elif delta >= 3:
            return "表现不错，稳住了局面。"
        elif delta >= 0:
            return "势均力敌，双方各有千秋。"
        return "这一轮略显被动，需要更有力的回应！"


class RescuerAgent(BaseAgent):
    RESCUE_TEMPLATES = {
        "refuse_polite": [
            "您太客气了！我今天真不能喝了，改天我专门请您！",
            "大舅您这让我都不好意思了，我真得量力而行啊！",
        ],
        "deflect": [
            "这事儿咱回头细说，今儿高兴先让我敬您一杯茶！",
            "大舅您说得对！对了，我最近听说个事儿...",
        ],
        "compliment": [
            "大舅您真是太客气了！您身体这么硬朗！",
            "您这话说的，我哪敢跟您比啊！",
        ],
        "accept_graceful": [
            "既然大舅都这么说了，我哪能不给面子！",
            "您这么看得起我，我必须得陪一个！",
        ],
    }

    def __init__(self, llm=None):
        super().__init__(AgentRole.RESCUER, llm)

    def think(self, context: Dict) -> AgentMessage:
        user_input = context.get("user_input", "")
        ai_response = context.get("ai_response", "")
        dominance = context.get("dominance", {"user": 50, "ai": 50})

        strategy = self._analyze_situation(user_input, ai_response, dominance)

        if self.llm:
            try:
                prompt = f"你是酒桌情商大师。用户说：{user_input}。AI回应：{ai_response}。用户气场：{dominance['user']}。请给出一个高情商回复建议，不超过40字。"
                suggestion = self.llm.generate(prompt, max_new_tokens=80)
                return AgentMessage(
                    self.role,
                    suggestion.strip(),
                    {"strategy": strategy},
                    confidence=0.9,
                )
            except:
                pass

        import random

        templates = self.RESCUE_TEMPLATES.get(
            strategy, self.RESCUE_TEMPLATES["deflect"]
        )
        return AgentMessage(
            self.role, random.choice(templates), {"strategy": strategy}, confidence=0.75
        )

    def _analyze_situation(
        self, user_input: str, ai_response: str, dominance: Dict
    ) -> str:
        text = (user_input + " " + ai_response).lower()
        if dominance["user"] < 30:
            return "compliment"
        if any(w in text for w in ["必须", "一定", "肯定"]):
            return "refuse_polite"
        if any(w in text for w in ["不喝", "不能", "不行"]):
            return "deflect"
        if any(w in text for w in ["来", "喝", "敬"]):
            return "accept_graceful"
        return "deflect"


class MemoryAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(AgentRole.MEMORY, llm)
        self.long_term_memory: Dict = {}

    def think(self, context: Dict) -> AgentMessage:
        session_id = context.get("session_id")
        action = context.get("memory_action", "retrieve")

        if action == "store":
            self._store_memory(session_id, context.get("turn_data", {}))
            return AgentMessage(self.role, "已存储", confidence=1.0)
        elif action == "retrieve":
            memory = self._retrieve_memory(session_id)
            return AgentMessage(
                self.role, json.dumps(memory), {"memory": memory}, confidence=1.0
            )

        return AgentMessage(self.role, "无效操作", confidence=0.5)

    def _store_memory(self, session_id: str, turn_data: Dict):
        if session_id not in self.long_term_memory:
            self.long_term_memory[session_id] = {"turns": [], "scores": []}
        self.long_term_memory[session_id]["turns"].append(turn_data)
        if "scores" in turn_data:
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
        self.state = AgentState()
        self.agents_list = list(self.agents.values())

    def process_turn(self, context: Dict) -> Dict:
        memory_msg = self.agents[AgentRole.MEMORY].think(
            {**context, "memory_action": "retrieve"}
        )
        context["memory"] = json.loads(memory_msg.content) if memory_msg.content else {}

        dialogue_msg = self.agents[AgentRole.DIALOGUE].think(context)
        context["ai_response"] = dialogue_msg.content

        evaluator_msg = self.agents[AgentRole.EVALUATOR].think(context)

        turn_data = {
            "user_input": context.get("user_input"),
            "ai_response": dialogue_msg.content,
            "speaker": dialogue_msg.metadata.get("speaker"),
            "scores": evaluator_msg.metadata.get("scores"),
        }
        self.agents[AgentRole.MEMORY].think(
            {**context, "memory_action": "store", "turn_data": turn_data}
        )

        for msg in [memory_msg, dialogue_msg, evaluator_msg]:
            self.state.history.append(msg)

        return {
            "ai_response": dialogue_msg.content,
            "speaker": dialogue_msg.metadata.get("speaker"),
            "judgment": evaluator_msg.content,
            "scores": evaluator_msg.metadata.get("scores", {}),
            "new_dominance": evaluator_msg.metadata.get(
                "new_dominance", {"user": 50, "ai": 50}
            ),
            "game_over": self._check_game_over(
                evaluator_msg.metadata.get("new_dominance", {})
            ),
        }

    def get_rescue_suggestion(self, context: Dict) -> str:
        rescue_msg = self.agents[AgentRole.RESCUER].think(context)
        return rescue_msg.content

    def _check_game_over(self, dominance: Dict) -> bool:
        user = dominance.get("user", 50)
        return user <= 10 or user >= 90
