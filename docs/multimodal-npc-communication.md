# 多模态与 NPC 沟通机制说明

本文档说明 TalkArena 中“表情 + 声音”如何进入系统，并最终影响 NPC 的对话策略与反馈。

## 1. 一句话概览

用户每次发送消息时，前端会把 `emotion`（表情状态）和 `voice_features`（语音特征）一起提交到后端。后端先做多模态融合，得到 `emotion_state` 与 `behavior_cues`，再把多模态信息带入对话引擎，影响 NPC 话术、评分和压迫节奏（取决于当前引擎模式）。

## 2. 前端如何采集与上报

### 2.1 表情数据（摄像头）

前端维护 `emotionData`，核心字段：

- `confidence`
- `nervous`
- `calm`
- `focus`

当前页面中该数据由摄像头逻辑周期更新（示例为模拟值），并在发送消息时并入 `multimodal`。

### 2.2 语音数据（麦克风 + STT）

语音输入通过 `/api/stt` 获取：

- `text`（识别文本）
- `voice_features`（语音情感特征）
- `emotion_state`（语音参与融合后的状态）
- `behavior_cues`

前端会缓存 `lastVoiceFeatures`、`lastVoiceText`，在下一次 `/api/chat/send` 时带上。

### 2.3 聊天请求载荷

发送聊天时的关键结构：

```json
{
  "session_id": "...",
  "message": "用户文本",
  "multimodal": {
    "emotion": {
      "confidence": 80,
      "nervous": 20,
      "calm": 70,
      "focus": 75
    },
    "voice_features": {"...": "..."},
    "voice_text": "..."
  }
}
```

## 3. 后端多模态融合流程

### 3.1 入口

- `/api/chat/send`：主对话入口，会调用 `MultimodalAnalyzer.process_turn(...)`
- `/api/multimodal/analyze`：独立分析入口
- `/api/stt`：语音识别后调用 `analyze_multimodal(...)` 返回状态

### 3.2 融合步骤

`MultimodalAnalyzer` 的核心流程：

1. 将前端 `emotion` 映射为 `MicroExpressionFeatures`
2. 将 `voice_features` 映射为 `VoiceEmotionFeatures`
3. 通过 `UserEmotionStateMachine.update(...)` 融合 face + voice + text sentiment
4. 输出：
   - `emotion_state`（主情绪、强度、valence/arousal/dominance、置信度）
   - `behavior_cues`（建议 NPC 语气、眼神、动作）
   - `patterns`（历史模式）
   - `trend`（趋势）
   - `inconsistencies`（跨模态不一致，如“表情平静但声音紧张”）

### 3.3 状态机做了什么

`UserEmotionStateMachine` 负责：

- 表情向量 + 语音向量融合
- 时间平滑（减少抖动）
- 离散情绪推断（如 confident/nervous/angry）
- 置信度估计
- 不一致性检测（表情与声音冲突）
- 趋势追踪（如 `increasing_stress`）

## 4. 多模态如何影响 NPC

这里分三种运行路径：

### 4.1 经典多 Agent 模式（`use_unified_agent=False`）

多模态会**直接参与 NPC 生成**：

- `DialogueAgent._emotion_hint(...)` 根据 `nervous/confidence/focus/calm` 生成“多模态提示”
- 提示注入 LLM prompt，改变 NPC 语气和推进策略
- `EvaluatorAgent` 用多模态修正评分，特别是 `pressure_handling`
- `DecisionEngine` 在 `multimodal.available=True` 时采用更高的 emotion/voice 权重

结论：这一模式下，多模态对 NPC 行为影响是直接且明显的。

### 4.2 统一 Agent 模式（`use_unified_agent=True`，当前默认）

当前默认引擎使用 `UnifiedAgent`。现状是：

- `process_turn(..., multimodal=...)` 会接收多模态参数
- 但 `UnifiedAgent.process(...)` 当前未接收/使用 `multimodal`
- 因此多模态主要体现在：
  - API 返回中的 `multimodal_analysis`
  - 前端状态展示与陪练反馈

结论：当前默认模式下，多模态对 NPC 文本生成的影响是“间接/弱耦合”，不是强驱动。

### 4.3 回退引擎（Fallback）

在回退引擎中，多模态会参与简单规则评分：

- `response_quality` 受 `focus - nervous` 影响
- `pressure_handling` 受 `confidence - nervous` 影响

## 5. 语音反向影响（TTS）

`/api/tts` 支持按情绪选择音色（如 happy/sad/angry/neutral），即 NPC 语音输出也能根据情绪标签变化，形成“输入情绪 -> NPC策略 -> 输出语气”的闭环体验。

## 6. 时序图（简化）

```text
Camera/Mic -> Frontend emotionData + voice_features
          -> POST /api/chat/send
          -> MultimodalAnalyzer.process_turn
          -> EmotionState + BehaviorCues
          -> Engine.process_turn
              -> (classic multi-agent: multimodal directly affects prompt/scoring)
              -> (unified-agent default: currently weak coupling)
          -> NPC utterances + multimodal_analysis -> Frontend render
```

## 7. 联调与排查建议

1. 先看 `/api/chat/send` 请求体里是否有 `multimodal.emotion` 和 `multimodal.voice_features`。
2. 再看响应里是否有 `data.multimodal_analysis.emotion_state`。
3. 如果你希望“NPC说话内容明显受表情/声音影响”，需确认是否在经典多 Agent 模式运行。
4. 语音链路问题优先检查 `/api/stt` 返回是否包含 `voice_features`。

## 8. 关键结论

- 多模态分析链路是完整的：采集 -> 融合 -> 状态 -> 行为提示。
- 是否真正“驱动 NPC 文本策略”，取决于运行模式。
- 在经典多 Agent 模式中，表情和声音对 NPC 决策影响最直接。
