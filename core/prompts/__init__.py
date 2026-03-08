"""
Prompt 管理模块
集中管理所有 LLM 调用的 prompt 模板
"""

from .registry import (
    # 场景生成 prompts
    SCENARIO_GENERATION_PROMPTS,
    # 对话生成 prompt
    DIALOGUE_GENERATION_PROMPT,
    # 救场大师 prompt
    RESCUE_MASTER_PROMPT,
    # 裁判 prompt
    DOMINANCE_JUDGE_PROMPT,
    # 复盘报告 prompts
    REPORT_SCORES_PROMPT,
    REPORT_SUMMARY_PROMPT,
    REPORT_NPC_INNER_VOICE_PROMPT,
    # 对决总结 prompt
    DUEL_SUMMARY_PROMPT,
    # 工具函数
    format_prompt,
    get_scenario_generation_prompt,
    get_dialogue_generation_prompt,
    get_rescue_master_prompt,
    get_dominance_judge_prompt,
    get_report_scores_prompt,
    get_report_summary_prompt,
    get_report_npc_inner_voice_prompt,
    get_duel_summary_prompt,
)

__all__ = [
    # Prompts
    "SCENARIO_GENERATION_PROMPTS",
    "DIALOGUE_GENERATION_PROMPT",
    "RESCUE_MASTER_PROMPT",
    "DOMINANCE_JUDGE_PROMPT",
    "REPORT_SCORES_PROMPT",
    "REPORT_SUMMARY_PROMPT",
    "REPORT_NPC_INNER_VOICE_PROMPT",
    "DUEL_SUMMARY_PROMPT",
    # Functions
    "format_prompt",
    "get_scenario_generation_prompt",
    "get_dialogue_generation_prompt",
    "get_rescue_master_prompt",
    "get_dominance_judge_prompt",
    "get_report_scores_prompt",
    "get_report_summary_prompt",
    "get_report_npc_inner_voice_prompt",
    "get_duel_summary_prompt",
]
