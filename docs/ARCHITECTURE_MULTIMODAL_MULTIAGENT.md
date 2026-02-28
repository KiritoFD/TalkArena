# TalkArena 架构说明：多模态多 Agent Talking Head

## 1. 目标

本架构用于单用户对多 NPC 场景（面试/饭桌）：

1. 话权真实：并行提案、竞价决策、冷却与插话窗口。
2. 注意力真实：被点名角色优先响应，旁观角色持续追随 speaker。
3. 非语言真实：未发言角色持续处于 listening/reacting，不“站桩”。

## 2. 核心分层

- `core/multimodal_multiagent/contracts.py`：统一协议数据结构。
- `core/multimodal_multiagent/user_signal_fusion.py`：音频/表情/文本融合为 `UserState`。
- `core/multimodal_multiagent/scenario_director.py`：场景导演规则。
- `core/multimodal_multiagent/npc_policy.py`：每个 NPC 的独立轻策略与 proposal。
- `core/multimodal_multiagent/speaker_selection.py`：话权评分与中断判断。
- `core/multimodal_multiagent/group_coordinator.py`：群体协同信号。
- `core/multimodal_multiagent/orchestrator.py`：完整流水线编排。

## 3. 单轮时序

1. 前端上报 multimodal features。
2. `UserSignalFusion.fuse()` 得到 `UserState`。
3. `ScenarioDirector.plan()` 得到导演意图。
4. `NPCPolicyPlanner.build_proposals()` 并行生成 proposal。
5. `SpeakerSelector.decide()` 选 speaker / runner-up。
6. `GroupCoordinator.coordinate()` 给出协同节奏与群体策略。
7. 输出单人 `SpeechInstruction` + 全员 `RenderFrame.nonverbals`。

## 4. 成本控制

- 并行阶段仅 JSON proposal（轻量）。
- 单轮最多 1 次重生成 + 1 次 TTS（重路径）。
- 插话优先短句模板。
- 非语言由规则驱动，不依赖大模型。

## 5. 测试与验收

- `tests/test_multiagent_runtime.py`：基础运行时行为。
- `tests/test_multiagent_full_flow.py`：自动 proposal + 场景协同阶段。
- `tests/test_multimodal_multiagent_package.py`：新包入口可用性。
- `tests/test_app_full_process.py`：对话/救场/总结 API 全流程与页面钩子。
