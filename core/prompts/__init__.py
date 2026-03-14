"""
Prompt 管理模块
集中管理所有 LLM 调用的 prompt 模板
"""

from .registry import (
    # 场景生成 prompts
    SCENARIO_GENERATION_PROMPTS,
    # 对话生成 prompts
    DIALOGUE_GENERATION_PROMPT,
    DIALOGUE_GENERATION_PROMPTS,
    # 救场大师 prompts
    RESCUE_MASTER_PROMPT,
    RESCUE_MASTER_PROMPTS,
    RESCUE_MASTER_FALLBACK_PROMPT,
    # 裁判 prompts
    DOMINANCE_JUDGE_PROMPT,
    DOMINANCE_JUDGE_PROMPTS,
    # 复盘报告 prompts
    REPORT_SCORES_PROMPT,
    REPORT_SCORES_PROMPTS,
    REPORT_SUMMARY_PROMPT,
    REPORT_SUMMARY_PROMPTS,
    REPORT_NPC_INNER_VOICE_PROMPT,
    REPORT_NPC_INNER_VOICE_PROMPTS,
    # 对决总结 prompts
    DUEL_SUMMARY_PROMPT,
    DUEL_SUMMARY_PROMPTS,
    # Unified Agent 对话生成 prompt
    UNIFIED_AGENT_DIALOGUE_PROMPT,
    # 场景特定 System Prompts
    SCENE_SYSTEM_PROMPTS,
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
    get_unified_agent_dialogue_prompt,
    get_scene_system_prompt,
)

__all__ = [
    # Prompts
    "SCENARIO_GENERATION_PROMPTS",
    "DIALOGUE_GENERATION_PROMPT",
    "DIALOGUE_GENERATION_PROMPTS",
    "RESCUE_MASTER_PROMPT",
    "RESCUE_MASTER_PROMPTS",
    "RESCUE_MASTER_FALLBACK_PROMPT",
    "DOMINANCE_JUDGE_PROMPT",
    "DOMINANCE_JUDGE_PROMPTS",
    "REPORT_SCORES_PROMPT",
    "REPORT_SCORES_PROMPTS",
    "REPORT_SUMMARY_PROMPT",
    "REPORT_SUMMARY_PROMPTS",
    "REPORT_NPC_INNER_VOICE_PROMPT",
    "REPORT_NPC_INNER_VOICE_PROMPTS",
    "DUEL_SUMMARY_PROMPT",
    "DUEL_SUMMARY_PROMPTS",
    "UNIFIED_AGENT_DIALOGUE_PROMPT",
    "SCENE_SYSTEM_PROMPTS",
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
    "get_unified_agent_dialogue_prompt",
    "get_scene_system_prompt",
]
