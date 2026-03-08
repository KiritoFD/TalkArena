"""
Prompt 注册表
集中管理所有 LLM 调用的 prompt 模板
"""

from typing import Dict, Any, Optional


# ============================================================================
# 场景生成 Prompts
# ============================================================================

SCENARIO_GENERATION_PROMPTS = {
    "shandong_dinner": {
        "full": """
请为一场山东饭桌场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括时间、地点、目的和氛围
2. 3 个饭桌成员的详细信息，每个成员包括：
   - 姓名
   - 角色（如：长辈、晚辈、同事等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像
3. 用户身份信息，用户身份应符合年轻人群体，例如：晚辈、年轻人、刚工作的新人等

当前场景名称：{scene_name}
请确保生成的内容符合山东酒桌文化特点，角色设定合理，背景故事生动。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
- user_identity: 用户身份信息，包含 name、role、personality、background、avatar 字段
""",
        "characters_only": """
请为一场山东饭桌场景生成 3 个饭桌成员的详细信息，每个成员包括：
- 姓名
- 角色（如：长辈、晚辈、同事等）
- 性格特点
- 背景故事
- 适合的 emoji 头像

当前场景名称：{scene_name}
请确保生成的内容符合山东酒桌文化特点，角色设定合理，背景故事生动。

请以 JSON 格式输出，包含以下字段：
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
"""
    },
    "business_dinner": {
        "full": """
请为一场商务饭局场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括公司类型、合作目的、参与人员和氛围
2. 3 个饭局成员的详细信息，每个成员包括：
   - 姓名
   - 角色（如：老板、客户、合作伙伴等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像
3. 用户身份信息，用户身份应符合职场人士，例如：下属、新人、项目负责人等

当前场景名称：{scene_name}
请确保生成的内容符合商务饭局场景，角色设定专业，背景故事合理。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
- user_identity: 用户身份信息，包含 name、role、personality、background、avatar 字段
""",
        "characters_only": """
请为一场商务饭局场景生成 3 个饭局成员的详细信息，每个成员包括：
- 姓名
- 角色（如：老板、客户、合作伙伴等）
- 性格特点
- 背景故事
- 适合的 emoji 头像

当前场景名称：{scene_name}
请确保生成的内容符合商务饭局场景，角色设定专业，背景故事合理。

请以 JSON 格式输出，包含以下字段：
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
"""
    },
    "interview": {
        "full": """
请为一场面试场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括公司类型、面试岗位、面试目的
2. 2-3 个面试相关角色的详细信息，每个角色包括：
   - 姓名
   - 角色（如：面试官、HR、竞争者等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像

当前场景名称：{scene_name}
请确保生成的内容符合职场面试场景，角色设定专业，背景故事合理。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
""",
        "characters_only": """
请为一场面试场景生成 2-3 个面试相关角色的详细信息，每个角色包括：
- 姓名
- 角色（如：面试官、HR、竞争者等）
- 性格特点
- 背景故事
- 适合的 emoji 头像

当前场景名称：{scene_name}
请确保生成的内容符合职场面试场景，角色设定专业，背景故事合理。

请以 JSON 格式输出，包含以下字段：
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
"""
    },
    "debate": {
        "full": """
请为一场辩论场景生成以下内容：
1. 详细的场景背景描述（2-3 句话），包括辩论主题、辩论形式、参与人员
2. 3 个辩论相关角色的详细信息，每个角色包括：
   - 姓名
   - 角色（如：正方辩手、反方辩手、主持人等）
   - 性格特点
   - 背景故事
   - 适合的 emoji 头像

当前场景名称：{scene_name}
请确保生成的内容符合辩论场景特点，角色设定鲜明，背景故事合理。

请以 JSON 格式输出，包含以下字段：
- description: 场景描述
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
""",
        "characters_only": """
请为一场辩论场景生成 3 个辩论相关角色的详细信息，每个角色包括：
- 姓名
- 角色（如：正方辩手、反方辩手、主持人等）
- 性格特点
- 背景故事
- 适合的 emoji 头像

当前场景名称：{scene_name}
请确保生成的内容符合辩论场景特点，角色设定鲜明，背景故事合理。

请以 JSON 格式输出，包含以下字段：
- characters: 成员列表，每个成员包含 name、role、personality、background、avatar 字段
"""
    }
}


# ============================================================================
# 对话生成 Prompts
# ============================================================================

DIALOGUE_GENERATION_PROMPT = """
你在场景《{scene_name}》扮演"{speaker_name}"。
场景氛围：{atmosphere}
修辞要求：{rhetoric}
角色设定：{personality}
说话风格：{style}
补充背景：{scene_description}
{user_identity}
多模态提示：{emotion_hint}
用户刚说：{user_input}

输出规则:
1) 只输出 NPC 的一句话。
2) 不要复述用户原话，不要出现引号包裹的用户台词。
3) 不要输出角色名、旁白、系统提示。
4) 控制在{word_limit}字以内。
5) 符合你的身份和性格。
6) 如果用户提到敏感话题，可以巧妙转移。
7) 保持对话流畅自然。
"""


# ============================================================================
# 救场大师 Prompts
# ============================================================================

RESCUE_MASTER_PROMPT = """你是一位顶尖的沟通专家。用户在以下场景中需要帮助，请你以用户的身份（晚辈/下属）生成一段高情商回复供其参考。

【场景】{scene_name}
【对手】{ai_name}
【当前气场】用户 {user_dominance} vs AI {ai_dominance}

【对话历史】
{context}

【任务】
你要以用户（晚辈/下属）的第一人称身份生成一条得体的回复，用户可以直接复制发送。
要求：
1. 必须以第一人称说话（"我..."），不能用第三人称（禁止"你应该...""可以说..."）
2. 简短有力，直击要害，不超过 50 字
3. 符合晚辈/下属身份，谦逊但不失气场
4. 能化解困境或扶回局势

请直接输出台词，不要有任何解释。"""


# ============================================================================
# 裁判 Prompts
# ============================================================================

DOMINANCE_JUDGE_PROMPT = """你是专业的辩论/谈判裁判。分析这轮交锋，判断气场转移。

【场景】{scene_name}
【当前气场】用户 {user_dominance} vs AI {ai_dominance}（总和 100）

【用户发言】
"{user_text}"

【{ai_name}回应】
"{ai_text}"

【评判维度】
1. 论点强度：论据充分性、逻辑严密性
2. 气势表现：语气自信度、压迫感
3. 反击有效性：是否有效回应对方攻击
4. 心理战术：是否动摇对方信心

【输出格式】（严格按此格式，只输出两行）
气场转移：[整数，-25 到 +25，正数表示用户占优，负数表示 AI 占优]
点评：[一句话点评]"""


# ============================================================================
# 复盘报告 Prompts
# ============================================================================

REPORT_SCORES_PROMPT = """# Role
你是"山东人饭局情商大挑战"的打分裁判，负责给玩家在饭局对话中的表现从五个维度打分。

# Input
- 场景描述：{scene_name}
- NPC 设定列表：{npc_list}
- 历史对话：
{history_log}

# Task
分析对话，给出玩家在五个维度的客观得分，满分 10，输出从 0-100 的数值。5 个指标如下：
1. "oily": 圆滑度：避重就轻、推诱话题的能力
2. "friendliness": 亲和力：共情与情绪价值提供
3. "logic": 逻辑性：论据支撑与表达条理
4. "humor": 幽默感：破冰与自嘲能力
5. "respect": 懂规矩：礼仪遵守与分寸感

# Output Format (JSON Only)
{{
  "metrics": {{
    "oily": int,
    "friendliness": int,
    "logic": int,
    "humor": int,
    "respect": int
  }}
}}

# Constraints
只输出 JSON 格式，不得输出任何额外解释文字"""


REPORT_SUMMARY_PROMPT = """# Role
你是一位在山东饭局混迹三十年、眼光毒辣的人情世故宗师。你的任务是根据玩家在"山东人饭局情商大挑战"中的对话表现，给出一份既专业又扎心的总结陈词。

# Input
- 场景描述：{scene_name}
- NPC 设定列表：{npc_list}
- 历史对话：
{history_log}
- 玩家称号：{medal}

# Task 
分析对话历史，撰写一段 100 字以内的玩家表现综合点评。

# Writing Constraints
- 犀利度：不要客气，要像一位严厉的长辈或刻薄的职场前辈。如果表现差，请使用"社交自杀"、"拆迁队"、"冷场王"等词汇。
- 专业深度：点评必须基于真实的社交潜规则。
- 称号挂钩：点评必须匹配生成的玩家称号。
- 结构化：第一句：定性评价；中间语句：逻辑分析；结尾句：总结。

# Constraints
直接输出总结陈词内容，不得输出任何额外解释文字"""


REPORT_NPC_INNER_VOICE_PROMPT = """# Role
你是一位在山东饭局混迹三十年、毒舌且看透世事的"人情世故大宗师"。

# Input Data
- 场景描述：{scene_name}
- NPC 设定列表：{npc_list}
- 历史对话：
{history_log}
- 玩家称号：{medal}

# Tasks
1. 生成 NPC 内心 OS：为 NPC 列表中的每人生成一段 20 字以内的心理活动。要求口语化，符合人设。
2. 生成改进建议：针对玩家最不合时宜的一句话，给出高情商台词改写及避坑逻辑。

# Output Format (Strict JSON)
{{
  "npc_inner_voice": [
    {{"name": "...", "os": "..."}},
    {{"name": "...", "os": "..."}}
  ],
  "high_light_suggestion": "..."
}}

# Constraints
只输出 JSON 格式，不得输出任何额外解释文字"""


# ============================================================================
# 对决总结 Prompts
# ============================================================================

DUEL_SUMMARY_PROMPT = """你是一位专业的沟通教练。分析以下对决并给出详细点评和改进建议。

【场景】{scene_name}
【对手】{ai_name}
【最终气场】用户 {user_dominance} vs AI {ai_dominance}
【回合数】{turn_count}

【对话记录】
{dialogue}

请 output（严格按以下 format）：

## 🎯 对决结果
[{result}，最终气场比分]

## 📊 表现分析
- 优势：[列举 2-3 个亮点]
- 不足：[列举 2-3 个问题]

## 🔑 关键回合复盘
[指出 1-2 个关键转折点，分析为什么赢/输]

## 💡 改进建议
[给出 3 条具体可操作的建议]"""


# ============================================================================
# 工具函数
# ============================================================================

def format_prompt(template: str, **kwargs) -> str:
    """格式化 prompt 模板"""
    return template.format(**kwargs)


def get_scenario_generation_prompt(scene_type: str, only_characters: bool = False) -> Optional[str]:
    """获取场景生成 prompt"""
    if scene_type not in SCENARIO_GENERATION_PROMPTS:
        return None
    
    prompts = SCENARIO_GENERATION_PROMPTS[scene_type]
    return prompts["characters_only"] if only_characters else prompts["full"]


def get_dialogue_generation_prompt(
    scene_name: str,
    speaker_name: str,
    atmosphere: str,
    rhetoric: str,
    personality: str,
    style: str,
    scene_description: str,
    user_identity: str,
    emotion_hint: str,
    user_input: str,
    word_limit: int = 60
) -> str:
    """获取对话生成 prompt"""
    return format_prompt(
        DIALOGUE_GENERATION_PROMPT,
        scene_name=scene_name,
        speaker_name=speaker_name,
        atmosphere=atmosphere,
        rhetoric=rhetoric,
        personality=personality,
        style=style,
        scene_description=scene_description,
        user_identity=user_identity,
        emotion_hint=emotion_hint,
        user_input=user_input,
        word_limit=word_limit
    )


def get_rescue_master_prompt(
    scene_name: str,
    ai_name: str,
    user_dominance: int,
    ai_dominance: int,
    context: str
) -> str:
    """获取救场大师 prompt"""
    return format_prompt(
        RESCUE_MASTER_PROMPT,
        scene_name=scene_name,
        ai_name=ai_name,
        user_dominance=user_dominance,
        ai_dominance=ai_dominance,
        context=context
    )


def get_dominance_judge_prompt(
    scene_name: str,
    user_dominance: int,
    ai_dominance: int,
    user_text: str,
    ai_text: str,
    ai_name: str
) -> str:
    """获取裁判 prompt"""
    return format_prompt(
        DOMINANCE_JUDGE_PROMPT,
        scene_name=scene_name,
        user_dominance=user_dominance,
        ai_dominance=ai_dominance,
        user_text=user_text,
        ai_text=ai_text,
        ai_name=ai_name
    )


def get_report_scores_prompt(
    scene_name: str,
    npc_list: str,
    history_log: str
) -> str:
    """获取复盘报告评分 prompt"""
    return format_prompt(
        REPORT_SCORES_PROMPT,
        scene_name=scene_name,
        npc_list=npc_list,
        history_log=history_log
    )


def get_report_summary_prompt(
    scene_name: str,
    npc_list: str,
    history_log: str,
    medal: str
) -> str:
    """获取复盘报告总结 prompt"""
    return format_prompt(
        REPORT_SUMMARY_PROMPT,
        scene_name=scene_name,
        npc_list=npc_list,
        history_log=history_log,
        medal=medal
    )


def get_report_npc_inner_voice_prompt(
    scene_name: str,
    npc_list: str,
    history_log: str,
    medal: str
) -> str:
    """获取 NPC 内心 OS prompt"""
    return format_prompt(
        REPORT_NPC_INNER_VOICE_PROMPT,
        scene_name=scene_name,
        npc_list=npc_list,
        history_log=history_log,
        medal=medal
    )


def get_duel_summary_prompt(
    scene_name: str,
    ai_name: str,
    user_dominance: int,
    ai_dominance: int,
    turn_count: int,
    dialogue: str,
    result: str
) -> str:
    """获取对决总结 prompt"""
    return format_prompt(
        DUEL_SUMMARY_PROMPT,
        scene_name=scene_name,
        ai_name=ai_name,
        user_dominance=user_dominance,
        ai_dominance=ai_dominance,
        turn_count=turn_count,
        dialogue=dialogue,
        result=result
    )
