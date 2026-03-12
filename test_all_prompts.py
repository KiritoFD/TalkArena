#!/usr/bin/env python3
"""测试所有Prompt是否已正确集中管理在registry.py中"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TalkArena.core.prompts.registry import (
    # 场景生成
    SCENARIO_GENERATION_PROMPTS,
    # 对话生成
    DIALOGUE_GENERATION_PROMPT,
    DIALOGUE_GENERATION_PROMPTS,
    # 救场大师
    RESCUE_MASTER_PROMPT,
    RESCUE_MASTER_PROMPTS,
    RESCUE_MASTER_FALLBACK_PROMPT,
    # 裁判
    DOMINANCE_JUDGE_PROMPT,
    DOMINANCE_JUDGE_PROMPTS,
    # 复盘报告
    REPORT_SCORES_PROMPT,
    REPORT_SCORES_PROMPTS,
    REPORT_SUMMARY_PROMPT,
    REPORT_SUMMARY_PROMPTS,
    REPORT_NPC_INNER_VOICE_PROMPT,
    REPORT_NPC_INNER_VOICE_PROMPTS,
    # 对决总结
    DUEL_SUMMARY_PROMPT,
    DUEL_SUMMARY_PROMPTS,
    # Unified Agent
    UNIFIED_AGENT_DIALOGUE_PROMPT,
    # 场景特定System Prompt
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
    get_rescue_master_fallback_prompt,
)

def check_prompt_exists(name: str, prompt_obj, expected_scenes=None) -> bool:
    """检查Prompt是否存在"""
    print(f"检查: {name}")
    if prompt_obj is None:
        print(f"  ❌ 不存在")
        return False
    if isinstance(prompt_obj, dict):
        print(f"  ✅ 字典类型，包含 {len(prompt_obj)} 个场景")
        if expected_scenes:
            for scene in expected_scenes:
                if scene in prompt_obj:
                    print(f"    ✅ 包含场景: {scene}")
                else:
                    print(f"    ❌ 缺少场景: {scene}")
    elif isinstance(prompt_obj, str):
        print(f"  ✅ 字符串类型，长度: {len(prompt_obj)}")
    else:
        print(f"  ⚠️  未知类型: {type(prompt_obj)}")
    return True

def test_all_prompts_exist():
    """测试所有Prompt是否存在"""
    print("=" * 70)
    print("测试所有Prompt是否已正确集中管理")
    print("=" * 70 + "\n")
    
    all_passed = True
    expected_scenes = ["shandong_dinner", "business_dinner", "interview", "debate"]
    
    # 场景生成
    print("--- 场景生成 Prompts ---")
    all_passed &= check_prompt_exists("SCENARIO_GENERATION_PROMPTS", SCENARIO_GENERATION_PROMPTS, expected_scenes)
    print()
    
    # 对话生成
    print("--- 对话生成 Prompts ---")
    all_passed &= check_prompt_exists("DIALOGUE_GENERATION_PROMPT", DIALOGUE_GENERATION_PROMPT)
    all_passed &= check_prompt_exists("DIALOGUE_GENERATION_PROMPTS", DIALOGUE_GENERATION_PROMPTS, expected_scenes)
    print()
    
    # 救场大师
    print("--- 救场大师 Prompts ---")
    all_passed &= check_prompt_exists("RESCUE_MASTER_PROMPT", RESCUE_MASTER_PROMPT)
    all_passed &= check_prompt_exists("RESCUE_MASTER_PROMPTS", RESCUE_MASTER_PROMPTS, expected_scenes)
    all_passed &= check_prompt_exists("RESCUE_MASTER_FALLBACK_PROMPT", RESCUE_MASTER_FALLBACK_PROMPT)
    print()
    
    # 裁判
    print("--- 裁判 Prompts ---")
    all_passed &= check_prompt_exists("DOMINANCE_JUDGE_PROMPT", DOMINANCE_JUDGE_PROMPT)
    all_passed &= check_prompt_exists("DOMINANCE_JUDGE_PROMPTS", DOMINANCE_JUDGE_PROMPTS, expected_scenes)
    print()
    
    # 复盘报告
    print("--- 复盘报告 Prompts ---")
    all_passed &= check_prompt_exists("REPORT_SCORES_PROMPT", REPORT_SCORES_PROMPT)
    all_passed &= check_prompt_exists("REPORT_SCORES_PROMPTS", REPORT_SCORES_PROMPTS, expected_scenes + ["default"])
    all_passed &= check_prompt_exists("REPORT_SUMMARY_PROMPT", REPORT_SUMMARY_PROMPT)
    all_passed &= check_prompt_exists("REPORT_SUMMARY_PROMPTS", REPORT_SUMMARY_PROMPTS, expected_scenes + ["default"])
    all_passed &= check_prompt_exists("REPORT_NPC_INNER_VOICE_PROMPT", REPORT_NPC_INNER_VOICE_PROMPT)
    all_passed &= check_prompt_exists("REPORT_NPC_INNER_VOICE_PROMPTS", REPORT_NPC_INNER_VOICE_PROMPTS, expected_scenes + ["default"])
    print()
    
    # 对决总结
    print("--- 对决总结 Prompts ---")
    all_passed &= check_prompt_exists("DUEL_SUMMARY_PROMPT", DUEL_SUMMARY_PROMPT)
    all_passed &= check_prompt_exists("DUEL_SUMMARY_PROMPTS", DUEL_SUMMARY_PROMPTS, expected_scenes)
    print()
    
    # Unified Agent
    print("--- Unified Agent Prompts ---")
    all_passed &= check_prompt_exists("UNIFIED_AGENT_DIALOGUE_PROMPT", UNIFIED_AGENT_DIALOGUE_PROMPT)
    print()
    
    # 场景特定System Prompt
    print("--- 场景特定 System Prompts ---")
    all_passed &= check_prompt_exists("SCENE_SYSTEM_PROMPTS", SCENE_SYSTEM_PROMPTS, ["debate", "interview"])
    print()
    
    return all_passed

def test_all_functions_exist():
    """测试所有函数是否存在"""
    print("=" * 70)
    print("测试所有工具函数是否存在")
    print("=" * 70 + "\n")
    
    functions = [
        ("format_prompt", format_prompt),
        ("get_scenario_generation_prompt", get_scenario_generation_prompt),
        ("get_dialogue_generation_prompt", get_dialogue_generation_prompt),
        ("get_rescue_master_prompt", get_rescue_master_prompt),
        ("get_dominance_judge_prompt", get_dominance_judge_prompt),
        ("get_report_scores_prompt", get_report_scores_prompt),
        ("get_report_summary_prompt", get_report_summary_prompt),
        ("get_report_npc_inner_voice_prompt", get_report_npc_inner_voice_prompt),
        ("get_duel_summary_prompt", get_duel_summary_prompt),
        ("get_unified_agent_dialogue_prompt", get_unified_agent_dialogue_prompt),
        ("get_scene_system_prompt", get_scene_system_prompt),
        ("get_rescue_master_fallback_prompt", get_rescue_master_fallback_prompt),
    ]
    
    all_passed = True
    for name, func in functions:
        if func and callable(func):
            print(f"✅ {name}")
        else:
            print(f"❌ {name}")
            all_passed = False
    
    print()
    return all_passed

def test_function_calls():
    """测试函数调用是否正常"""
    print("=" * 70)
    print("测试函数调用")
    print("=" * 70 + "\n")
    
    all_passed = True
    
    # 测试get_scene_system_prompt
    print("测试 get_scene_system_prompt:")
    try:
        debate_prompt = get_scene_system_prompt("debate")
        if debate_prompt and "辩论选手" in debate_prompt:
            print("  ✅ debate场景获取成功")
        else:
            print("  ❌ debate场景获取失败")
            all_passed = False
        
        interview_prompt = get_scene_system_prompt("interview")
        if interview_prompt and "压力面试" in interview_prompt:
            print("  ✅ interview场景获取成功")
        else:
            print("  ❌ interview场景获取失败")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        all_passed = False
    print()
    
    # 测试get_rescue_master_fallback_prompt
    print("测试 get_rescue_master_fallback_prompt:")
    try:
        prompt = get_rescue_master_fallback_prompt(
            scene_name="测试场景",
            ai_name="测试AI",
            user_dominance=50,
            ai_dominance=50,
            context="测试对话内容"
        )
        if prompt and "沟通专家" in prompt:
            print("  ✅ 调用成功")
        else:
            print("  ❌ 调用失败")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        all_passed = False
    print()
    
    # 测试get_unified_agent_dialogue_prompt
    print("测试 get_unified_agent_dialogue_prompt:")
    try:
        prompt = get_unified_agent_dialogue_prompt(
            scene_name="测试场景",
            atmosphere="测试氛围",
            rhetoric="测试修辞",
            pressure_hint="",
            drinking_hint="",
            chars_desc="- 测试NPC",
            history_str="（暂无对话）",
            interrupt_hint="",
            user_input_hint="用户刚说：测试",
            word_limit=60
        )
        if prompt and "总导演" in prompt:
            print("  ✅ 调用成功")
        else:
            print("  ❌ 调用失败")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        all_passed = False
    print()
    
    return all_passed

def generate_prompt_summary():
    """生成Prompt汇总信息"""
    print("=" * 70)
    print("Prompt汇总信息")
    print("=" * 70 + "\n")
    
    total_prompts = 0
    
    print("📊 统计:")
    print(f"- 场景生成: {len(SCENARIO_GENERATION_PROMPTS)} 场景 × 2 模式 = {len(SCENARIO_GENERATION_PROMPTS) * 2}")
    total_prompts += len(SCENARIO_GENERATION_PROMPTS) * 2
    
    print(f"- 对话生成: 1 (默认) + {len(DIALOGUE_GENERATION_PROMPTS)} (场景) = {1 + len(DIALOGUE_GENERATION_PROMPTS)}")
    total_prompts += 1 + len(DIALOGUE_GENERATION_PROMPTS)
    
    print(f"- 救场大师: 1 (默认) + {len(RESCUE_MASTER_PROMPTS)} (场景) + 1 (备用) = {2 + len(RESCUE_MASTER_PROMPTS)}")
    total_prompts += 2 + len(RESCUE_MASTER_PROMPTS)
    
    print(f"- 裁判: 1 (默认) + {len(DOMINANCE_JUDGE_PROMPTS)} (场景) = {1 + len(DOMINANCE_JUDGE_PROMPTS)}")
    total_prompts += 1 + len(DOMINANCE_JUDGE_PROMPTS)
    
    print(f"- 复盘评分: 1 (默认) + {len(REPORT_SCORES_PROMPTS)} (场景) = {1 + len(REPORT_SCORES_PROMPTS)}")
    total_prompts += 1 + len(REPORT_SCORES_PROMPTS)
    
    print(f"- 复盘总结: 1 (默认) + {len(REPORT_SUMMARY_PROMPTS)} (场景) = {1 + len(REPORT_SUMMARY_PROMPTS)}")
    total_prompts += 1 + len(REPORT_SUMMARY_PROMPTS)
    
    print(f"- 复盘-NPC内心OS: 1 (默认) + {len(REPORT_NPC_INNER_VOICE_PROMPTS)} (场景) = {1 + len(REPORT_NPC_INNER_VOICE_PROMPTS)}")
    total_prompts += 1 + len(REPORT_NPC_INNER_VOICE_PROMPTS)
    
    print(f"- 对决总结: 1 (默认) + {len(DUEL_SUMMARY_PROMPTS)} (场景) = {1 + len(DUEL_SUMMARY_PROMPTS)}")
    total_prompts += 1 + len(DUEL_SUMMARY_PROMPTS)
    
    print(f"- Unified Agent: 1")
    total_prompts += 1
    
    print(f"- 场景特定System Prompt: {len(SCENE_SYSTEM_PROMPTS)}")
    total_prompts += len(SCENE_SYSTEM_PROMPTS)
    
    print(f"\n总计: {total_prompts} 个Prompt模板")
    print("\n✅ 所有Prompt已集中管理在 core/prompts/registry.py 中！")
    print("   可以将此文件交给负责优化提示词的同学处理。")

def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("Prompt集中管理验证")
    print("=" * 70 + "\n")
    
    results = []
    results.append(("所有Prompt存在检查", test_all_prompts_exist()))
    results.append(("所有函数存在检查", test_all_functions_exist()))
    results.append(("函数调用测试", test_function_calls()))
    
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        all_passed &= passed
    
    print()
    generate_prompt_summary()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过！所有Prompt已正确集中管理！")
    else:
        print("⚠️  部分测试失败，请检查")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
