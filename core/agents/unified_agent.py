from __future__ import annotations

import json
import logging
import random
import time
from typing import Dict, List, Optional

from .unified_agent_contracts import (
    NPCCharacter,
    NPCUtterance,
    ConversationTurn,
    UnifiedAgentResponse,
    SceneType,
)

logger = logging.getLogger("UnifiedAgent")


class UnifiedAgent:
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
                    "avatar": "🍷",
                },
                "大妗子": {
                    "personality": "会圆场，擅观察，擅软性引导。",
                    "style": "语气柔和，先顺着说，再轻推一步。",
                    "avatar": "☕",
                },
                "表哥": {
                    "personality": "活跃气氛，推进节奏。",
                    "style": "直接利落，可打趣但不刻薄。",
                    "avatar": "🍺",
                },
            },
            "openings": {
                "大舅": "来，先坐稳先吃菜，酒慢慢来，话一条一条说。",
                "大妗子": "先别紧张，先吃口热菜，咱慢慢聊、细细聊。",
                "表哥": "不急着拼酒，先碰个轻的，边吃边把事说透。",
            },
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
                    "avatar": "💼",
                },
                "李总": {
                    "personality": "善沟通，掌控节奏。",
                    "style": "客气但锋利，追问时间点和责任人。",
                    "avatar": "📊",
                },
                "周顾问": {
                    "personality": "偏风控，重合规和边界。",
                    "style": "先列约束，再给可落地路径。",
                    "avatar": "📋",
                },
            },
        },
        "interview": {
            "name": "群面竞争场",
            "atmosphere": "竞争激烈、同台竞技，主面试官在旁观察。",
            "rhetoric": "结论前置，突出优势，展现团队合作意识。",
            "word_limit": 78,
            "characters": {
                "竞争者A": {
                    "personality": "自信强势，善于表现自己。",
                    "style": "积极发言，善于展示优势。",
                    "avatar": "👩‍💼",
                },
                "竞争者B": {
                    "personality": "沉稳细致，回答有条理。",
                    "style": "稳扎稳打，回答有条理。",
                    "avatar": "🧑‍💼",
                },
                "竞争者C": {
                    "personality": "思维活跃，常有创新观点。",
                    "style": "思维活跃，常有创新观点。",
                    "avatar": "👨‍💼",
                },
            },
        },
    }

    def __init__(self, llm=None):
        self.llm = llm
        self.last_spoken: Dict[str, float] = {}

    def _resolve_scene(self, scenario_id: str) -> Dict:
        return self.SCENES.get(scenario_id, self.SCENES["shandong_dinner"])

    def _get_characters(self, scenario_id: str, custom_chars: Optional[List[Dict]] = None) -> List[NPCCharacter]:
        scene = self._resolve_scene(scenario_id)
        
        if custom_chars and len(custom_chars) > 0:
            chars = []
            for char in custom_chars:
                name = char.get("name", char.get("n", "NPC"))
                bio = char.get("bio", char.get("role", char.get("personality", "普通角色")))
                avatar = char.get("avatar", char.get("a", "👤"))
                
                chars.append(NPCCharacter(
                    npc_id=name,
                    name=name,
                    personality=bio,
                    speaking_style="自然对话",
                    avatar=avatar,
                ))
            return chars
        
        return [
            NPCCharacter(
                npc_id=name,
                name=name,
                personality=char_info["personality"],
                speaking_style=char_info["style"],
                avatar=char_info.get("avatar", "👤"),
            )
            for name, char_info in scene["characters"].items()
        ]

    def _build_history_str(self, history: List[ConversationTurn]) -> str:
        if not history:
            return ""
        
        lines = []
        for turn in history[-8:]:
            if turn.is_user:
                lines.append(f"用户：{turn.text}")
            else:
                lines.append(f"{turn.speaker}：{turn.text}")
        
        return "\n".join(lines)

    def _build_prompt(
        self,
        scenario_id: str,
        user_input: Optional[str],
        characters: List[NPCCharacter],
        history: List[ConversationTurn],
        is_interrupt: bool = False,
    ) -> str:
        scene = self._resolve_scene(scenario_id)
        history_str = self._build_history_str(history)
        
        chars_desc = "\n".join([
            f"- {c.name}：性格{c.personality}，说话风格{c.speaking_style}"
            for c in characters
        ])
        
        char_names = ", ".join([c.name for c in characters])
        
        interrupt_hint = "用户刚刚打断了对话，请注意衔接自然。" if is_interrupt else ""
        
        user_input_hint = f"用户刚说：{user_input}" if user_input else "对话刚开始，请开场。"
        
        return f"""你是一个酒局/对话场景的总导演，负责操控所有NPC进行自然对话。

场景：{scene.get('name', '对话')}
场景氛围：{scene.get('atmosphere', '')}
修辞要求：{scene.get('rhetoric', '')}

在场的NPC：
{chars_desc}

之前的对话：
{history_str if history_str else '（暂无对话）'}

{interrupt_hint}
{user_input_hint}

你的任务：
1. 生成「一轮」完整的NPC对话
2. 「一轮」指的是：从上一次用户发言后，到下一次等待用户发言前的所有对话
3. 这一轮对话结束后，必须把话头抛给用户
4. 为每个NPC生成符合其性格的对话内容

输出格式（JSON）：
{{
    "utterances": [
        {{
            "npc_id": "NPC名字",
            "text": "对话内容（不超过{scene.get('word_limit', 60)}字）",
            "delay_ms": 1200
        }}
    ],
    "should_await_user": true,
    "reason": "简要说明决策原因"
}}

规则说明：
- should_await_user 必须设置为 true，表示这一轮对话结束后等待用户发言
- 每个NPC的对话要符合其性格设定
- NPC之间可以有来有回地对话
- 一轮对话最多包含3-4个NPC的发言
- 最后一个NPC的发言要自然地把话头抛给用户
- delay_ms建议在800-2000毫秒之间"""

    def _parse_llm_response(self, raw_text: str) -> UnifiedAgentResponse:
        try:
            text = raw_text.strip()
            
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                
                utterances = [
                    NPCUtterance(
                        npc_id=u.get("npc_id", ""),
                        text=u.get("text", ""),
                        delay_ms=u.get("delay_ms", 1500),
                    )
                    for u in data.get("utterances", [])
                ]
                
                return UnifiedAgentResponse(
                    utterances=utterances,
                    should_await_user=True,
                    reason=data.get("reason", ""),
                )
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
        
        return UnifiedAgentResponse(
            utterances=[],
            should_await_user=True,
            reason="fallback",
        )

    def _generate_fallback(
        self,
        scenario_id: str,
        characters: List[NPCCharacter],
        user_input: Optional[str],
        history: List[ConversationTurn],
    ) -> UnifiedAgentResponse:
        scene = self._resolve_scene(scenario_id)
        
        if not user_input and not history:
            openings = scene.get("openings", {})
            scene_char_names = list(scene.get("characters", {}).keys())
            
            if openings and len(characters) > 0:
                use_preset_openings = True
                for c in characters:
                    if c.name not in scene_char_names:
                        use_preset_openings = False
                        break
                
                if use_preset_openings:
                    utterances = []
                    for i, (char_name, opening_text) in enumerate(openings.items()):
                        utterances.append(NPCUtterance(
                            npc_id=char_name,
                            text=opening_text,
                            delay_ms=3000 if i > 0 else 0,
                        ))
                    return UnifiedAgentResponse(
                        utterances=utterances,
                        should_await_user=True,
                        reason="开场",
                    )
        
        utterances = []
        num_utterances = min(2 + random.randint(0, 1), len(characters))
        used_chars = []
        
        for i in range(num_utterances):
            available_chars = [c for c in characters if c.name not in used_chars]
            if not available_chars:
                available_chars = characters
            
            speaker = random.choice(available_chars)
            used_chars.append(speaker.name)
            self.last_spoken[speaker.name] = time.time()
            
            fallback_texts = [
                "来，咱继续聊。",
                "你觉得呢？",
                "这个事有意思。",
                "接着说接着说。",
                "有道理！",
                "那咱接着往下说。",
                "来来来，继续继续。",
                "这个话题挺有意思。",
            ]
            
            text = random.choice(fallback_texts)
            
            utterances.append(NPCUtterance(
                npc_id=speaker.name,
                text=text,
                delay_ms=3000 if i > 0 else 0,
            ))
        
        return UnifiedAgentResponse(
            utterances=utterances,
            should_await_user=True,
            reason="fallback",
        )

    def process(
        self,
        scenario_id: str,
        user_input: Optional[str] = None,
        custom_characters: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        is_interrupt: bool = False,
    ) -> UnifiedAgentResponse:
        characters = self._get_characters(scenario_id, custom_characters)
        
        history = []
        if conversation_history:
            for h in conversation_history:
                history.append(ConversationTurn(
                    speaker=h.get("speaker", "用户" if h.get("is_user") else "NPC"),
                    text=h.get("text", ""),
                    timestamp_ms=h.get("timestamp_ms", int(time.time() * 1000)),
                    is_user=h.get("is_user", False),
                ))
        
        if self.llm:
            try:
                prompt = self._build_prompt(
                    scenario_id=scenario_id,
                    user_input=user_input,
                    characters=characters,
                    history=history,
                    is_interrupt=is_interrupt,
                )
                
                response = self.llm.generate(prompt, max_new_tokens=800, temperature=0.7)
                result = self._parse_llm_response(response)
                
                if result.utterances:
                    for u in result.utterances:
                        self.last_spoken[u.npc_id] = time.time()
                    return result
            except Exception as e:
                logger.error(f"UnifiedAgent LLM调用失败: {e}")
        
        return self._generate_fallback(scenario_id, characters, user_input, history)
