#!/usr/bin/env python3
"""测试场景化Prompt定制化功能"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TalkArena.core.prompts.registry import (
    DOMINANCE_JUDGE_PROMPTS,
    DUEL_SUMMARY_PROMPTS,
    get_dominance_judge_prompt,
    get_duel_summary_prompt,
    _get_scene_type_by_name
)

def test_scene_type_mapping():
    """测试场景名称到场景类型的映射"""
    print("=== 测试场景类型映射 ===\n")
    
    test_cases = [
        ("家庭饭桌试炼", "shandong_dinner"),
        ("商务饭局谈判", "business_dinner"),
        ("群面竞争场", "interview"),
        ("立场攻防辩论", "debate"),
        ("未知场景", "default")
    ]
    
    all_passed = True
    for scene_name, expected_type in test_cases:
        result = _get_scene_type_by_name(scene_name)
        passed = result == expected_type
        all_passed &= passed
        status = "✅" if passed else "❌"
        print(f"{status} 场景名称: '{scene_name}' → 类型: '{result}' (期望: '{expected_type}')")
    
    print(f"\n场景类型映射测试: {'✅ 全部通过' if all_passed else '❌ 存在失败'}\n")
    return all_passed

def test_dominance_judge_prompts():
    """测试裁判Prompt场景化"""
    print("=== 测试裁判Prompt场景化 ===\n")
    
    # 检查所有场景都有对应的Prompt
    required_scenes = ["shandong_dinner", "business_dinner", "interview", "debate"]
    all_passed = True
    
    for scene in required_scenes:
        if scene in DOMINANCE_JUDGE_PROMPTS:
            print(f"✅ 裁判Prompt: '{scene}' 存在")
            # 检查Prompt包含场景特定内容
            prompt = DOMINANCE_JUDGE_PROMPTS[scene]
            if scene == "shandong_dinner" and "山东饭桌文化" in prompt:
                print(f"   ✓ 包含山东饭桌文化内容")
            elif scene == "business_dinner" and "商务谈判" in prompt:
                print(f"   ✓ 包含商务谈判内容")
            elif scene == "interview" and "群面招聘" in prompt:
                print(f"   ✓ 包含群面招聘内容")
            elif scene == "debate" and "辩论裁判" in prompt:
                print(f"   ✓ 包含辩论裁判内容")
        else:
            print(f"❌ 裁判Prompt: '{scene}' 缺失")
            all_passed = False
    
    # 测试函数调用
    print("\n测试函数调用:")
    try:
        prompt = get_dominance_judge_prompt(
            scene_name="家庭饭桌试炼",
            user_dominance=50,
            ai_dominance=50,
            user_text="我觉得这样挺好",
            ai_text="是吗？我可不这么认为",
            ai_name="大舅"
        )
        print("✅ get_dominance_judge_prompt 调用成功")
        if "山东饭桌文化" in prompt:
            print("   ✓ 返回了山东饭桌场景的Prompt")
    except Exception as e:
        print(f"❌ get_dominance_judge_prompt 调用失败: {e}")
        all_passed = False
    
    print(f"\n裁判Prompt测试: {'✅ 通过' if all_passed else '❌ 失败'}\n")
    return all_passed

def test_duel_summary_prompts():
    """测试对决总结Prompt场景化"""
    print("=== 测试对决总结Prompt场景化 ===\n")
    
    # 检查所有场景都有对应的Prompt
    required_scenes = ["shandong_dinner", "business_dinner", "interview", "debate"]
    all_passed = True
    
    for scene in required_scenes:
        if scene in DUEL_SUMMARY_PROMPTS:
            print(f"✅ 对决总结Prompt: '{scene}' 存在")
            # 检查Prompt包含场景特定内容
            prompt = DUEL_SUMMARY_PROMPTS[scene]
            if scene == "shandong_dinner" and "饭桌对决" in prompt:
                print(f"   ✓ 包含饭桌对决内容")
            elif scene == "business_dinner" and "商务对决" in prompt:
                print(f"   ✓ 包含商务对决内容")
            elif scene == "interview" and "群面对决" in prompt:
                print(f"   ✓ 包含群面对决内容")
            elif scene == "debate" and "辩论对决" in prompt:
                print(f"   ✓ 包含辩论对决内容")
        else:
            print(f"❌ 对决总结Prompt: '{scene}' 缺失")
            all_passed = False
    
    # 测试函数调用
    print("\n测试函数调用:")
    try:
        prompt = get_duel_summary_prompt(
            scene_name="家庭饭桌试炼",
            ai_name="大舅",
            user_dominance=60,
            ai_dominance=40,
            turn_count=10,
            dialogue="用户：谢谢\n大舅：不客气，来喝一杯",
            result="用户胜"
        )
        print("✅ get_duel_summary_prompt 调用成功")
        if "饭桌对决" in prompt:
            print("   ✓ 返回了山东饭桌场景的Prompt")
    except Exception as e:
        print(f"❌ get_duel_summary_prompt 调用失败: {e}")
        all_passed = False
    
    print(f"\n对决总结Prompt测试: {'✅ 通过' if all_passed else '❌ 失败'}\n")
    return all_passed

def main():
    """主测试函数"""
    print("=" * 60)
    print("场景化Prompt定制化功能测试")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("场景类型映射", test_scene_type_mapping()))
    results.append(("裁判Prompt", test_dominance_judge_prompts()))
    results.append(("对决总结Prompt", test_duel_summary_prompts()))
    
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        all_passed &= passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
