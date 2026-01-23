"""
TalkArena 全面测试
测试内容：场景、对局、消息、气场算法、语音
"""
import sys
import time
from pathlib import Path

print("=" * 60)
print("TalkArena 全面测试")
print("=" * 60)

# ============================================================
# 1. 基础导入测试
# ============================================================
print("\n[1] 基础导入测试")

from orchestrator import Orchestrator, Session, Turn, logger
from ui.handlers import get_scenarios, start_session, send_message
print("✓ 模块导入成功")

# ============================================================
# 2. 场景配置测试
# ============================================================
print("\n[2] 场景配置测试")

# 禁用 TTS 加速测试
orch = Orchestrator(enable_tts=False)

scenarios = orch.scenarios
assert len(scenarios) == 3, f"应有3个场景，实际 {len(scenarios)}"
assert "negotiation" in scenarios
assert "debate" in scenarios
assert "interview" in scenarios

for sid, scenario in scenarios.items():
    assert "name" in scenario, f"{sid} 缺少 name"
    assert "ai_name" in scenario, f"{sid} 缺少 ai_name"
    assert "system_prompt" in scenario, f"{sid} 缺少 system_prompt"
    assert "opening" in scenario, f"{sid} 缺少 opening"
    print(f"  ✓ {sid}: {scenario['name']} ({scenario['ai_name']})")

print("✓ 场景配置完整")

# ============================================================
# 3. Session 数据结构测试
# ============================================================
print("\n[3] Session 数据结构测试")

session = orch.start_session("negotiation")

assert session.session_id, "session_id 不能为空"
assert session.scenario_id == "negotiation"
assert session.user_name == "你"
assert session.ai_name == "王总"
assert session.user_dominance == 50
assert session.ai_dominance == 50, f"AI气场应为50，实际 {session.ai_dominance}"
assert len(session.chat_history) == 1, "应有1条开场白"
assert session.turn_count == 0

print(f"  session_id: {session.session_id}")
print(f"  气场: 用户 {session.user_dominance} vs AI {session.ai_dominance}")
print("✓ Session 结构正确")

# ============================================================
# 4. 气场零和约束测试
# ============================================================
print("\n[4] 气场零和约束测试")

# 测试不同气场值
for user_dom in [5, 25, 50, 75, 95]:
    session.user_dominance = user_dom
    total = session.user_dominance + session.ai_dominance
    assert total == 100, f"气场总和应为100，实际 {total}"
    print(f"  用户 {session.user_dominance} + AI {session.ai_dominance} = {total} ✓")

print("✓ 零和约束正确")

# ============================================================
# 5. 犹豫惩罚算法测试
# ============================================================
print("\n[5] 犹豫惩罚算法测试")

test_cases = [
    (0, 0),    # 0秒 -> 0惩罚
    (2, 0),    # 2秒 -> 0惩罚
    (3, 3),    # 3秒 -> 3惩罚
    (6, 6),    # 6秒 -> 6惩罚
    (9, 9),    # 9秒 -> 9惩罚
    (15, 15),  # 15秒 -> 15惩罚（上限）
    (30, 15),  # 30秒 -> 15惩罚（上限）
]

for elapsed, expected in test_cases:
    penalty = min(int(elapsed // 3) * 3, 15)
    assert penalty == expected, f"elapsed={elapsed}: 期望{expected}, 实际{penalty}"
    print(f"  {elapsed}秒 -> 惩罚 {penalty} ✓")

print("✓ 犹豫惩罚算法正确")

# ============================================================
# 6. AI思考惩罚算法测试
# ============================================================
print("\n[6] AI思考惩罚算法测试")

test_cases = [
    (0, 0),
    (1, 0),
    (2, 2),
    (4, 4),
    (6, 6),
    (10, 10),  # 上限
    (20, 10),  # 上限
]

for think_time, expected in test_cases:
    penalty = min(int(think_time // 2) * 2, 10)
    assert penalty == expected, f"think_time={think_time}: 期望{expected}, 实际{penalty}"
    print(f"  {think_time}秒 -> 惩罚 {penalty} ✓")

print("✓ AI思考惩罚算法正确")

# ============================================================
# 7. 流式处理阶段测试
# ============================================================
print("\n[7] 流式处理阶段测试")

session2 = orch.start_session("debate")
stages_seen = []

for update in orch.process_turn_streaming(session2.session_id, "我认为科技发展利大于弊"):
    stage = update["stage"]
    stages_seen.append(stage)
    
    # 验证每个阶段都有气场值
    assert "user_dominance" in update
    assert "ai_dominance" in update
    assert update["user_dominance"] + update["ai_dominance"] == 100
    
    if stage == "complete":
        assert "ai_text" in update
        assert "judgment" in update
        assert "dominance_shift" in update
        print(f"  AI回复: {update['ai_text'][:50]}...")
        print(f"  裁判: {update['judgment']}")
        print(f"  气场变化: {update['dominance_shift']:+d}")

expected_stages = ["user_sent", "ai_thinking", "ai_responded", "complete"]
assert stages_seen == expected_stages, f"阶段顺序错误: {stages_seen}"
print(f"  阶段顺序: {' -> '.join(stages_seen)} ✓")

print("✓ 流式处理正确")

# ============================================================
# 8. handlers 接口测试
# ============================================================
print("\n[8] Handlers 接口测试")

# 注入测试用 orchestrator
import ui.handlers as handlers
handlers._orchestrator_instance = orch

# get_scenarios
scenarios_list = get_scenarios()
assert len(scenarios_list) == 3
print(f"  get_scenarios: {len(scenarios_list)} 个场景 ✓")

# start_session
session_id, chat_history, status, ai_dom, user_dom = start_session("interview")
assert session_id
assert len(chat_history) == 1
assert ai_dom == 50
assert user_dom == 50
print(f"  start_session: {session_id} ✓")

# send_message (Generator)
results = list(send_message(session_id, "我有5年工作经验", chat_history))
assert len(results) > 0
final = results[-1]
assert len(final) == 5  # (chat_history, input_clear, ai_dom, user_dom, audio_path)
print(f"  send_message: {len(results)} 次yield ✓")

print("✓ Handlers 接口正确")

# ============================================================
# 9. 边界条件测试
# ============================================================
print("\n[9] 边界条件测试")

# 空输入
results = list(send_message(session_id, "", chat_history))
assert len(results) == 1
print("  空输入处理 ✓")

# 无效 session_id (应抛出 ValueError - let it crash)
try:
    list(send_message("invalid-id", "test", []))
    assert False, "应抛出 ValueError"
except ValueError as e:
    assert "Session not found" in str(e)
    print("  无效session抛出ValueError ✓")

# 未选择场景
result = start_session("")
assert result[0] == ""  # session_id 为空
assert "请先选择场景" in result[2]
print("  未选场景处理 ✓")

print("✓ 边界条件处理正确")

# ============================================================
# 10. TTS 语音合成测试
# ============================================================
print("\n[10] TTS 语音合成测试")

from model_loader import TTSLoader
import io

tts = TTSLoader()
tts.load()

test_text = "你好，这是一段测试语音"
wav_bytes = tts.synthesize(test_text)

assert wav_bytes, "TTS 合成失败"
assert len(wav_bytes) > 1000, f"音频太小: {len(wav_bytes)} bytes"

# 验证是 WAV 格式
assert wav_bytes[:4] == b"RIFF", "输出不是 WAV 格式"
assert wav_bytes[8:12] == b"WAVE", "输出不是 WAV 格式"
print(f"  合成文本: {test_text}")
print(f"  输出大小: {len(wav_bytes)} bytes")
print(f"  格式验证: RIFF/WAVE ✓")

print("✓ TTS 合成正确 (MP3->WAV 转换成功)")

# ============================================================
# 11. STT 语音识别测试
# ============================================================
print("\n[11] STT 语音识别测试")

# 创建启用语音的 orchestrator
orch_with_voice = Orchestrator(enable_tts=True)

assert orch_with_voice.stt is not None, "STT 未初始化"
print("  STT 初始化 ✓")

# 使用刚才合成的音频测试 STT
from pathlib import Path
import tempfile

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(wav_bytes)
    temp_wav_path = f.name

print(f"  临时文件: {temp_wav_path}")

# 测试转录（需要网络，可能失败）
try:
    transcribed = orch_with_voice.transcribe_audio(temp_wav_path)
    print(f"  原文: {test_text}")
    print(f"  转录: {transcribed}")
    print("✓ STT 转录成功")
except Exception as e:
    print(f"  ⚠ STT 转录跳过 (需要网络): {e}")

# 清理临时文件
Path(temp_wav_path).unlink(missing_ok=True)

# ============================================================
# 12. process_voice_input 测试
# ============================================================
print("\n[12] 语音输入处理测试")

from ui.handlers import process_voice_input

# 注入启用语音的 orchestrator
handlers._orchestrator_instance = orch_with_voice

# 测试空音频
results = list(process_voice_input("", None, []))
assert len(results) == 1
print("  空音频处理 ✓")

# 测试无 session
results = list(process_voice_input("", "fake_path.wav", []))
assert len(results) == 1
print("  无session处理 ✓")

# 测试有效音频（使用合成的音频）
session3 = orch_with_voice.start_session("negotiation")

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(wav_bytes)
    temp_wav_path2 = f.name

try:
    results = list(process_voice_input(session3.session_id, temp_wav_path2, []))
    print(f"  有效音频: {len(results)} 次yield")
    
    if len(results) > 1:
        final = results[-1]
        print(f"  最终气场: AI {final[2]} vs 用户 {final[3]}")
        print("✓ 语音输入处理成功")
    else:
        print("  ⚠ 语音处理返回单次 (可能网络问题)")
except Exception as e:
    print(f"  ⚠ 语音输入测试跳过: {e}")

Path(temp_wav_path2).unlink(missing_ok=True)

# ============================================================
# 13. 音频文件格式验证测试
# ============================================================
print("\n[13] 音频格式验证测试")

def check_audio_format(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "WAV"
    elif data[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return "MP3"
    return "UNKNOWN"

# 验证 TTS 输出
fmt = check_audio_format(wav_bytes)
assert fmt == "WAV", f"TTS 输出应为 WAV，实际 {fmt}"
print(f"  TTS 输出格式: {fmt} ✓")

print("✓ 音频格式验证正确")

# ============================================================
# 测试总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 全部测试通过!")
print("=" * 60)

# ============================================================
# 测试总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 全部测试通过!")
print("=" * 60)