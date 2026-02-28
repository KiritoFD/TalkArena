# 多模态多 Agent 技术实现

## 模块总览

代码统一位于 `core/multimodal_multiagent/`：

- `contracts.py`：协议结构（UserState、Proposal、FloorDecision、RenderFrame）。
- `user_signal_fusion.py`：多模态融合。
- `scenario_director.py`：面试/饭桌导演规则。
- `npc_policy.py`：NPC 独立行动策略（微状态 + 并行提案）。
- `speaker_selection.py`：话权竞价、冷却、插话窗口。
- `group_coordinator.py`：群体协同（phase / soften / pressure / pace）。
- `orchestrator.py`：顶层编排。

## 关键设计

### 1) UserState（可控而非泛情绪）

融合输出 6 维：
- wants_to_speak
- confusion
- stress_arousal
- valence
- addressee_distribution
- pace_preference

这些量直接驱动导演、话权和非语言渲染。

### 2) NPC 独立行动

每个 NPC 维护 `NPCMicroState`（energy/assertiveness/cooperativeness/turns_silent），
并在每轮独立输出 `NPCProposal`，包括：
- 是否发言
- 紧迫度
- 意图（probe/challenge/rescue 等）
- 是否可插话
- 时长预算与非语言建议

### 3) 场景协同

`GroupCoordinator` 把导演意图 + 用户状态 + 当前话权结果转为统一协同信号，
保证多人行为方向一致，不会“各说各话”。

### 4) 双通道输出

- 台词通道：只给 speaker 生成 `SpeechInstruction`（重路径）。
- 非语言通道：给所有 NPC 输出 `NonverbalInstruction`（轻路径，持续更新）。

## 工程建议

- 前端用 WebSocket 实时消费 `RenderFrame`。
- 将 `SpeechInstruction` 和 TTS/口型管线解耦，避免阻塞非语言帧。
- 用 `tests/test_app_full_process.py` 做回归门禁，确保对话/救场/总结链路不退化。
