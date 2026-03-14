from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from .unified_agent_contracts import (
    NPCCharacter,
    NPCUtterance,
    ConversationTurn,
    UnifiedAgentResponse,
    SceneType,
)
from ..prompts.registry import get_unified_agent_dialogue_prompt

logger = logging.getLogger("UnifiedAgent")


class UnifiedAgent:
    SCENE_FALLBACK_UTTERANCES = {
        "shandong_dinner": [
            "咱先不急，先把最要紧的一点摆在桌上。",
            "先吃口菜缓一下，你把真实顾虑直说。",
            "礼数先到位，后面我们一条条把事说透。",
        ],
        "business_dinner": [
            "这个方向可以，我先把目标和时间点对齐。",
            "先把边界说清，再谈资源和执行节奏。",
            "我建议先锁定结果口径，再拆分落地步骤。",
        ],
        "interview": [
            "我先给结论，再补一个可量化案例。",
            "这个问题我分三点答，先说最关键的一点。",
            "我先说明行动和结果，再补复盘反思。",
        ],
        "debate": [
            "我先收敛定义，再给证据链。",
            "先确认前提成立，再讨论结论强度。",
            "我们先围绕可验证事实，不做情绪化扩展。",
        ],
    }

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
        for turn in history[-4:]:
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
        pressure_tags: Optional[List[str]] = None,
        pressure_value: Optional[int] = None,
        drinking_capacity: Optional[int] = None,
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
        
        pressure_hint = ""
        if pressure_tags and len(pressure_tags) > 0:
            pressure_hint = f"用户压力敏感区：{', '.join(pressure_tags)}。\n压力强度：{pressure_value or 5}/10（10为最高）。\n请NPC在对话中主动试探这些敏感话题，制造相应强度的社交压力。"
        
        drinking_hint = ""
        if drinking_capacity and drinking_capacity > 0:
            drinking_desc = {
                1: "很差，几乎不喝，一杯就倒",
                2: "一般，只能喝一点，容易醉",
                3: "不错，酒量较好，能喝不少",
            }
            drinking_hint = f"用户酒量：{drinking_desc.get(drinking_capacity, '未知')}（{drinking_capacity}/3星）。\n请NPC在对话中根据用户酒量来劝酒或安排酒局节奏。"
        
        return get_unified_agent_dialogue_prompt(
            scene_name=scene.get('name', '对话'),
            atmosphere=scene.get('atmosphere', ''),
            rhetoric=scene.get('rhetoric', ''),
            pressure_hint=pressure_hint,
            drinking_hint=drinking_hint,
            chars_desc=chars_desc,
            history_str=history_str if history_str else '（暂无对话）',
            interrupt_hint=interrupt_hint,
            user_input_hint=user_input_hint,
            word_limit=scene.get('word_limit', 60)
        )

    def _parse_llm_response(self, raw_text: str) -> UnifiedAgentResponse:
        text = (raw_text or "").strip()
        if not text:
            raise ValueError("LLM response is empty")

        candidates = self._extract_json_objects(text)
        if not candidates:
            candidates = [text]

        parsed_errs: List[str] = []
        data = None
        for cand in candidates:
            try:
                cand_s = cand.strip()
                if not cand_s:
                    continue
                data = json.loads(cand_s)
                break
            except Exception as e:
                parsed_errs.append(str(e))
                continue
        if data is None:
            raise ValueError(
                "LLM response JSON parse failed: " + " | ".join(parsed_errs[:3])
            )
        if not isinstance(data, dict):
            raise ValueError("LLM response JSON is not an object")

        if not isinstance(data.get("utterances"), list):
            raise ValueError("LLM response missing utterances array")
        utterances = [
            NPCUtterance(
                npc_id=u.get("npc_id", ""),
                text=u.get("text", ""),
                delay_ms=u.get("delay_ms", 700),
            )
            for u in data.get("utterances", [])
        ]
        return UnifiedAgentResponse(
            utterances=utterances,
            should_await_user=True,
            reason=data.get("reason", ""),
        )

    def _extract_json_objects(self, text: str) -> List[str]:
        s = str(text or "")
        objs: List[str] = []
        depth = 0
        in_str = False
        escaped = False
        start = -1
        for i, ch in enumerate(s):
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        objs.append(s[start : i + 1])
                        start = -1
        return objs

    def _build_json_regen_prompt(self, original_prompt: str) -> str:
        return (
            "严格按要求重新生成，不要解释，不要代码块，不要前后缀。\n"
            "只输出一行JSON。\n"
            "JSON schema:\n"
            '{"utterances":[{"npc_id":"string","text":"string","delay_ms":1200}],"should_await_user":true,"reason":"string"}\n'
            "注意：所有字符串必须是合法JSON字符串，内部双引号必须转义。\n\n"
            f"{original_prompt}"
        )

    def _build_json_repair_prompt(self, original_prompt: str, bad_output: str, parse_error: str) -> str:
        return (
            "You must fix an invalid JSON output.\n"
            "Return ONLY valid JSON, no markdown, no explanation.\n"
            "Keep the same schema exactly:\n"
            "{\n"
            '  "utterances": [{"npc_id": "string", "text": "string", "delay_ms": 1200}],\n'
            '  "should_await_user": true,\n'
            '  "reason": "string"\n'
            "}\n\n"
            "Constraints:\n"
            "- utterances must be a non-empty array.\n"
            "- Each text must be <= 80 Chinese characters.\n"
            "- Do not use unescaped double quotes inside JSON strings.\n"
            "- Ensure strict JSON syntax (double quotes, commas, no trailing commas).\n\n"
            f"Original generation prompt:\n{original_prompt}\n\n"
            f"Invalid output:\n{bad_output}\n\n"
            f"Parser error:\n{parse_error}\n"
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
                            delay_ms=700 if i > 0 else 0,
                        ))
                    return UnifiedAgentResponse(
                        utterances=utterances,
                        should_await_user=True,
                        reason="开场",
                    )
        
        utterances = []
        num_utterances = min(2 + random.randint(0, 1), len(characters))
        used_chars = []
        used_texts = set()
        scene_fallbacks = self.SCENE_FALLBACK_UTTERANCES.get(
            scenario_id, self.SCENE_FALLBACK_UTTERANCES["shandong_dinner"]
        )
        
        for i in range(num_utterances):
            available_chars = [c for c in characters if c.name not in used_chars]
            if not available_chars:
                available_chars = characters
            
            speaker = random.choice(available_chars)
            used_chars.append(speaker.name)
            self.last_spoken[speaker.name] = time.time()

            # Avoid same fallback line being spoken by multiple roles in one round.
            candidates = [t for t in scene_fallbacks if t not in used_texts]
            if not candidates:
                candidates = scene_fallbacks[:]
            text = random.choice(candidates)
            used_texts.add(text)
            
            utterances.append(NPCUtterance(
                npc_id=speaker.name,
                text=text,
                delay_ms=700 if i > 0 else 0,
            ))
        
        return UnifiedAgentResponse(
            utterances=utterances,
            should_await_user=True,
            reason="fallback",
        )

    def _dedupe_round_utterances(self, scenario_id: str, utterances: List[NPCUtterance]) -> List[NPCUtterance]:
        if not utterances:
            return utterances
        scene_fallbacks = self.SCENE_FALLBACK_UTTERANCES.get(
            scenario_id, self.SCENE_FALLBACK_UTTERANCES["shandong_dinner"]
        )
        used = set()
        out: List[NPCUtterance] = []
        for u in utterances:
            txt = (u.text or "").strip()
            if txt and txt in used:
                candidates = [t for t in scene_fallbacks if t not in used]
                if candidates:
                    txt = random.choice(candidates)
            if txt:
                used.add(txt)
            out.append(
                NPCUtterance(
                    npc_id=u.npc_id,
                    text=txt or u.text,
                    delay_ms=u.delay_ms,
                )
            )
        return out

    def _assert_no_duplicate_utterances(self, utterances: List[NPCUtterance]) -> None:
        seen = set()
        for u in utterances or []:
            txt = (u.text or "").strip()
            if not txt:
                continue
            if txt in seen:
                raise RuntimeError("UnifiedAgent generated duplicate utterances in one round; fallback disabled.")
            seen.add(txt)

    def process(
        self,
        scenario_id: str,
        user_input: Optional[str] = None,
        custom_characters: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        is_interrupt: bool = False,
        pressure_tags: Optional[List[str]] = None,
        pressure_value: Optional[int] = None,
        drinking_capacity: Optional[int] = None,
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
        
        if not self.llm:
            raise RuntimeError("UnifiedAgent requires an available LLM; fallback disabled.")

        prompt = self._build_prompt(
            scenario_id=scenario_id,
            user_input=user_input,
            characters=characters,
            history=history,
            is_interrupt=is_interrupt,
            pressure_tags=pressure_tags or [],
            pressure_value=pressure_value or 5,
            drinking_capacity=drinking_capacity or 0,
        )
        response = self.llm.generate(prompt, max_new_tokens=220, temperature=0.45)
        try:
            result = self._parse_llm_response(response)
        except Exception as parse_err:
            regen_prompt = self._build_json_regen_prompt(prompt)
            regenerated = self.llm.generate(regen_prompt, max_new_tokens=220, temperature=0.2)
            try:
                result = self._parse_llm_response(regenerated)
            except Exception as regen_err:
                repair_prompt = self._build_json_repair_prompt(prompt, regenerated or response, str(regen_err))
                repaired = self.llm.generate(repair_prompt, max_new_tokens=260, temperature=0.1)
                result = self._parse_llm_response(repaired)
        if not result.utterances:
            raise RuntimeError("UnifiedAgent returned empty utterances; fallback disabled.")
        self._assert_no_duplicate_utterances(result.utterances)
        for u in result.utterances:
            self.last_spoken[u.npc_id] = time.time()
        return result
