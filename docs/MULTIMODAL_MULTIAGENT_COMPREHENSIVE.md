# TalkArena 多模态 + 多 Agent 综合技术文档

更新时间：2026-03-13

## 1. 结论先说

1. 前端是否使用 MediaPipe：**是，但仅在实时演示页使用**。`assets/multimodal_realtime.html` 使用 `@mediapipe/face_mesh`；主业务页 `templates/index.html` 当前是摄像头 + 规则/模拟特征，不直接跑 MediaPipe。
2. 现网主链路：`templates/index.html -> /api/chat/send -> MultimodalAnalyzer -> TalkArenaEngine`。
3. 多 Agent 现状：
   - 现网默认是 `UnifiedAgent`（`use_unified_agent=True`）。
   - `core/multimodal_multiagent/` 为完整“多模态多 NPC 编排包”，目前主要由测试覆盖，尚未接入 `main.py` 主链路。
4. 黑板机制（Blackboard）：当前是“轻量黑板”形态（`context + run_tick 输出包 + session state`），尚无独立 `Blackboard` 类；可平滑升级到显式黑板总线。

---

## 2. 系统分层（按代码事实）

### 2.1 前端采集层

- 主页面：`templates/index.html`
  - 摄像头：`getUserMedia`
  - 语音录制：`AudioContext + ScriptProcessor`
  - 本地重采样：`downsampleBuffer(..., 16000)`
  - 上报：`/api/chat/send`、`/api/stt`
- 实时演示页面：`assets/multimodal_realtime.html`
  - 使用 `FaceMesh`（MediaPipe）进行面部关键点与特征提取
  - 特征缓冲 `emotionBuffer`（`MAX_BUFFER_SIZE = 30`）
  - `requestAnimationFrame` 驱动视觉与音量循环

### 2.2 多模态分析层

- 核心模块：`core/multimodal_analyzer.py`
  - `MultimodalAnalyzer`
  - `MultimodalFusionEngine`
  - `VoiceEmotionAnalyzer`
- 状态机与记忆：`core/emotion_state.py`
  - `UserEmotionStateMachine`
  - `EmotionMemory`
  - `MultimodalEmotionState`

### 2.3 对话决策层

- 主引擎：`core/engine.py`
  - 默认统一代理：`UnifiedAgent`
  - 可切换经典多 Agent：`MultiAgentOrchestrator`
- 经典多 Agent：`core/agents/multi_agent.py`
  - 对话 Agent、评估 Agent、救场 Agent、记忆 Agent

### 2.4 多模态多 Agent 编排包（预备主线）

目录：`core/multimodal_multiagent/`

- `contracts.py`：统一协议
- `user_signal_fusion.py`：多模态融合为 `UserState`
- `scenario_director.py`：场景导演意图
- `npc_policy.py`：每 NPC 独立 proposal
- `speaker_selection.py`：话权选择 + 冷却 + 插话窗口
- `group_coordinator.py`：群体节奏/缓压/施压协同
- `orchestrator.py`：`run_tick()` 总编排

---

## 3. 数据协议与关键对象

### 3.1 聊天请求中的多模态载荷（现网）

```json
{
  "session_id": "...",
  "message": "...",
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

### 3.2 融合输出（现网）

`MultimodalAnalyzer.analyze_multimodal()` 返回：

- `emotion_state`
- `behavior_cues`
- `patterns`
- `trend`
- `inconsistencies`

### 3.3 多模态多 Agent 包协议（预备主线）

`core/multimodal_multiagent/contracts.py` 定义：

- `UserState`：`wants_to_speak/confusion/stress_arousal/valence/addressee_distribution/pace_preference`
- `NPCProposal`：发言意图、紧迫度、可否打断、时长预算、非语言建议
- `FloorDecision`：speaker、runner-up、interrupt window
- `SpeechInstruction`：单 speaker 文本指令
- `RenderFrame` + `NonverbalInstruction`：全员非语言帧

---

## 4. 多模态如何影响 NPC

### 4.1 现网主路径（默认 UnifiedAgent）

`/api/chat/send` 会做两件事：

1. 调 `get_mm_analyzer().process_turn(...)` 生成 `multimodal_analysis`
2. 把 `multimodal` 传给 `engine.process_turn(...)`

注意：当前 `UnifiedAgent.process(...)` 参数不含 `multimodal`，因此默认路径下，多模态主要用于分析回传和状态展示，**对 NPC 文本生成影响较弱**。

### 4.2 经典多 Agent 路径（`use_unified_agent=False`）

多模态会直接进入 `DialogueAgent._emotion_hint(...)` 和 `EvaluatorAgent._evaluate(...)`：

- 紧张高：先稳情绪再推进
- 自信低：给台阶
- 专注低：收束问题
- 评分中 `pressure_handling` 会受 `confidence/nervous` 影响

### 4.3 预备主线：多模态多 Agent 编排包

`run_tick()` 完整链路：

1. `UserSignalFusion.fuse(features, npc_ids)`
2. `ScenarioDirector.plan(scene, user_state, context)`
3. `NPCPolicyPlanner.build_proposals(...)`
4. `SpeakerSelector.decide(...)`
5. `GroupCoordinator.coordinate(...)`
6. 输出 `speech_instruction`（重路径）+ `render_frame.nonverbals`（轻路径）

该链路天然支持“一个人说话、所有人有反应”的真实群体互动。

---

## 5. MediaPipe 使用现状与建议

### 5.1 当前事实

- `assets/multimodal_realtime.html`：已接入 MediaPipe Face Mesh。
- `templates/index.html`：当前不依赖 MediaPipe，使用简化规则/模拟情绪特征。
- `assets/multimodal_fixed.html`：明确是简化版（无 MediaPipe）。

### 5.2 建议

如果要让主业务页也具备真实面部特征：

1. 把 `multimodal_realtime` 的关键点提取模块抽成独立 JS 模块。
2. 在主页面保留“降级模式”（无 MediaPipe 时回退规则模式）。
3. 采样频率做自适应（CPU 高时降低帧率或减少关键点计算）。

---

## 6. 延迟降低机制（现状 + 可加项）

### 6.1 已有机制（代码中已实现）

1. **前端音频重采样到 16k**
   - 减少上传体积与 STT 开销。
2. **双通道思想已具雏形**
   - 文本/语音回复与非语言状态分离（`render_frame.nonverbals` 设计）。
3. **发言时长预算**
   - `NPCProposal.max_duration_ms`（interview 默认 3200ms，dinner 2400ms）。
4. **话权冷却**
   - `SpeakerSelector.cooldown_s=4.0`，避免同 NPC 高频连说。
5. **插话窗口限制**
   - `interrupt_window_ms=700`，控制打断节奏。
6. **只让一个 speaker 走重路径**
   - `speech_instruction` 单发言人，其他 NPC 用低成本 nonverbal 更新。

### 6.2 推荐补强（高收益）

1. 把 `/api/chat/send` 做为流式（SSE/WebSocket）输出：先发 nonverbal，再发语句分片。
2. 增加“短句模板快速路径”：当 `intent=probe/agree` 时优先模板，不走完整 LLM。
3. STT 和多模态分析并行化，合并回传。
4. 在 TTS 前加文本长度裁剪与去动作括号预处理（主页面部分已做）。
5. 添加 tick 级性能指标：`fusion_ms/policy_ms/select_ms/tts_ms/e2e_ms`。

---

## 7. 黑板机制（Blackboard）

### 7.1 当前实现形态（轻量黑板）

当前没有单独 `Blackboard` 类，但存在共享状态中枢：

- 输入侧：`context`（turn、场景上下文）
- 中间态：`user_state/director_intent/proposals/floor_decision/coordination`
- 输出侧：`speech_instruction/render_frame`
- 会话态：`engine.sessions`、`unified_history/history`

可理解为“隐式黑板”：每个子模块读写同一轮共享事实，完成协作决策。

### 7.2 显式黑板化建议（推荐）

建议新增 `core/multimodal_multiagent/blackboard.py`：

- `TickBlackboard`
  - `observations`：audio/face/text 原始观测
  - `inferences`：user_state、emotion_state、risk_flags
  - `decisions`：director/floor/coordination
  - `outputs`：speech/nonverbal/action queue
  - `metrics`：各阶段耗时
- 约束：
  - 只允许写自己的命名空间
  - 统一 schema 版本
  - 每 tick 只读快照，避免竞态

好处：

1. 可追溯（回放每轮决策）
2. 易观测（定位延迟和异常）
3. 易扩展（新 Agent 接入成本低）

---

## 8. 典型时序（端到端）

```text
[Frontend]
  Camera/Mic采集 -> emotion/voice_features
  -> POST /api/chat/send

[Backend API]
  -> MultimodalAnalyzer.process_turn
  -> engine.process_turn
      -> UnifiedAgent(默认) 或 MultiAgentOrchestrator
  -> 返回 ai_text/utterances + multimodal_analysis

[Frontend]
  -> 渲染 speaker + talking head状态
  -> /api/tts(可选)并排队播放
```

预备主线（multimodal_multiagent）对应：

```text
features -> user_fusion -> director -> npc proposals
         -> speaker selection -> group coordination
         -> speech_instruction + render_frame
```

---

## 9. 测试与验证

现有测试覆盖：

- `tests/test_multiagent_runtime.py`
- `tests/test_multiagent_full_flow.py`
- `tests/test_multimodal_multiagent_package.py`
- `tests/test_app_full_process.py`

建议新增：

1. `tests/test_latency_budget.py`：限定 P95 时延门槛。
2. `tests/test_blackboard_schema.py`：黑板字段兼容性。
3. `tests/test_mediapipe_fallback.py`：MediaPipe 不可用时自动降级。

---

## 10. 接入路线图（建议）

1. 第一步：把 `core/multimodal_multiagent` 通过 feature flag 接到 `main.py`（灰度）。
2. 第二步：引入显式 Blackboard 与 tick trace。
3. 第三步：上线流式输出（先 nonverbal，后 speech）。
4. 第四步：主页面接入可降级 MediaPipe，统一特征 schema。

---

## 11. 常见误解澄清

1. “项目前端全部用 MediaPipe” -> 不准确。当前只在实时演示页用。
2. “多模态已经强驱动默认 NPC 文本” -> 不完全准确。默认 UnifiedAgent 路径下驱动仍偏弱。
3. “黑板机制已经独立实现” -> 不准确。当前是隐式共享状态，可升级为显式黑板。

---

## 12. 附：关键文件索引

- API入口：`main.py`
- 主页面：`templates/index.html`
- 实时 MediaPipe 页面：`assets/multimodal_realtime.html`
- 多模态分析：`core/multimodal_analyzer.py`
- 情绪状态机：`core/emotion_state.py`
- 运行引擎：`core/engine.py`
- 经典多 Agent：`core/agents/multi_agent.py`
- 统一 Agent：`core/agents/unified_agent.py`
- 多模态多 Agent 包：`core/multimodal_multiagent/*`
- 核心测试：`tests/test_multiagent_runtime.py`、`tests/test_multiagent_full_flow.py`、`tests/test_multimodal_multiagent_package.py`
