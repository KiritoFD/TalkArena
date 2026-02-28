# TalkArena 后端接口文档

## 核心概念

| 概念 | 说明 |
|------|------|
| **气场值** | 零和博弈，`user_dominance + ai_dominance = 100`，范围 `[5, 95]` |
| **Session** | 一场对局，包含场景、气场、对话历史 |
| **Turn** | 一轮对话交锋 |

---

## 数据结构

### Session（后端存储）
```typescript
{
  session_id: string        // 8位UUID，对局唯一标识
  scenario_id: string       // 场景ID: "negotiation" | "debate" | "interview"
  user_name: string         // 用户显示名，固定为 "你"
  ai_name: string           // AI角色名，如 "王总"、"反方辩手"、"面试官"
  user_dominance: int       // 用户气场值 [5-95]，与AI气场之和恒为100
  ai_dominance: int         // AI气场值，计算属性 = 100 - user_dominance
  chat_history: Array<[string, string]>  // 后端存储格式 [(发言者名, 内容), ...]
  turn_count: int           // 当前回合数，从0开始
  last_activity: float      // 最后活动Unix时间戳，用于计算犹豫惩罚
}
```

### ChatMessage（前端交互格式）
```typescript
// 前端 Gradio Chatbot 使用的消息格式
{
  role: "user" | "assistant"   // user=用户发言, assistant=AI发言
  content: string              // 消息内容，assistant消息包含Markdown格式
}

// assistant消息content格式示例:
// "**王总**: （拍桌子）你这报价太离谱了！\n\n---\n_📊 AI强势反击 (气场+8)_"
```

### Turn（单轮对话）
```typescript
{
  text: string              // 对话文本内容
  audio_path: string | null // 语音文件路径，无语音时为null
  emotion: string           // 情感标签: "neutral" | "angry" | "happy"
}
```

### Scenario（场景配置）
```typescript
{
  name: string              // 场景显示名，如 "商务谈判"
  ai_name: string           // AI角色名，如 "王总"
  system_prompt: string     // 角色设定prompt，定义AI人格和行为模式
  opening: string           // AI开场白，对局开始时自动发送
}
```

---

## 接口列表

### 1. 获取场景列表
```python
get_scenarios() -> List[Tuple[str, str]]
```

**返回**: `[(display_name, scenario_id), ...]`

| scenario_id | display_name | ai_name |
|-------------|--------------|---------|
| `negotiation` | 商务谈判 | 王总 |
| `debate` | 辩论赛 | 反方辩手 |
| `interview` | 压力面试 | 面试官 |

---

### 2. 开始对局
```python
start_session(scenario_id: str) -> Tuple[str, List, str, int, int]
```

**输入**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| scenario_id | string | 是 | 场景ID，必须是有效值 |

**输出**: `(session_id, chat_history, status, ai_dominance, user_dominance)`

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 8位对局ID，后续接口必传 |
| chat_history | ChatMessage[] | 初始对话，包含AI开场白 |
| status | string | 状态信息，如 "✓ 对局开始 \| 场景: 商务谈判" |
| ai_dominance | int | AI初始气场值，固定50 |
| user_dominance | int | 用户初始气场值，固定50 |

**chat_history 示例**:
```json
[{"role": "assistant", "content": "**王总**: （王总靠在椅背上，手指敲着桌面）行，你们公司派你来谈..."}]
```

---

### 3. 发送消息（流式）
```python
send_message(session_id: str, user_input: str, chat_history: List) -> Generator
```

**输入**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 是 | 对局ID |
| user_input | string | 是 | 用户输入文本，空字符串会被忽略 |
| chat_history | ChatMessage[] | 是 | 当前对话历史 |

**输出**: Generator，多次yield `(chat_history, input_clear, ai_dominance, user_dominance, audio_path)`

| 字段 | 类型 | 说明 |
|------|------|------|
| chat_history | ChatMessage[] | 更新后的对话历史 |
| input_clear | string | 空字符串，用于清空输入框 |
| ai_dominance | int | 当前AI气场值 |
| user_dominance | int | 当前用户气场值 |
| audio_path | string \| null | 语音文件路径，TTS关闭时为null |

**流式更新说明**: 该接口会多次yield，前端应实时更新气场值显示。

---

### 4. 语音输入
```python
process_voice_input(session_id: str, audio_file: str | None, chat_history: List) -> Generator
```

**输入**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| session_id | string | 是 | 对局ID |
| audio_file | string \| null | 否 | 录音文件路径，由Gradio Audio组件提供 |
| chat_history | ChatMessage[] | 是 | 当前对话历史 |

**输出**: 同 `send_message`

**处理流程**: audio_file → STT转录 → send_message

---

### 5. 语音转文字
```python
transcribe_audio(audio_path: str) -> str
```

**输入**: WAV格式音频文件路径  
**输出**: 转录后的中文文本  
**异常**: STT未初始化时抛出 RuntimeError

---

## 气场算法

### 核心规则
- **零和约束**: `user_dominance + ai_dominance = 100`，一方涨另一方必跌
- **范围限制**: `[5, 95]`，防止一方完全归零导致游戏失衡
- **单变量维护**: 只存储 `user_dominance`，`ai_dominance` 为计算属性

### 气场变化来源

| 来源 | 触发条件 | 计算公式 | 影响 | 范围 |
|------|----------|----------|------|------|
| 用户犹豫惩罚 | 用户响应时间 > 3秒 | `min(elapsed // 3 * 3, 15)` | 用户↓ | 0~15 |
| AI思考惩罚 | AI生成时间 > 2秒 | `min(think_time // 2 * 2, 10)` | AI↓ | 0~10 |
| LLM裁判评分 | 每轮对话结束 | 由LLM实时判定 | 动态 | -25~+25 |

### 高级配置参数

#### 犹豫惩罚
```python
HESITATION_THRESHOLD = 3      # 开始计算惩罚的秒数阈值
HESITATION_PENALTY_RATE = 3   # 每3秒惩罚3点气场
HESITATION_MAX_PENALTY = 15   # 单次最大惩罚值
```

#### AI思考惩罚
```python
AI_THINK_THRESHOLD = 2        # 开始计算惩罚的秒数阈值
AI_THINK_PENALTY_RATE = 2     # 每2秒惩罚2点气场
AI_THINK_MAX_PENALTY = 10     # 单次最大惩罚值
```

#### 裁判评分
```python
JUDGE_SCORE_RANGE = (-25, 25) # 评分范围，正数=用户占优，负数=AI占优
```

---

## 流式更新阶段

`process_turn_streaming` 内部按以下阶段依次yield：

| stage | 触发时机 | 说明 | 关键字段 |
|-------|----------|------|----------|
| `user_sent` | 用户消息入队后 | 犹豫惩罚已计算完成 | user_dominance, ai_dominance, log |
| `ai_thinking` | AI开始生成前 | 提示前端显示"思考中" | user_dominance, ai_dominance |
| `ai_responded` | AI生成完成后 | AI思考惩罚已计算完成 | user_dominance, ai_dominance, log |
| `complete` | 回合完全结束 | 包含AI回复和裁判点评 | 全部字段 |

### complete阶段完整输出
```typescript
{
  stage: "complete"           // 阶段标识
  user_dominance: int         // 最终用户气场值
  ai_dominance: int           // 最终AI气场值
  ai_text: string             // AI回复文本（含动作描写）
  audio_path: string | null   // 语音文件路径
  judgment: string            // 裁判一句话点评
  dominance_shift: int        // 本轮气场转移值（正=用户涨，负=AI涨）
  log: string                 // 调试日志信息
}
```

---

## 模型配置

### LLM（大语言模型）
```python
{
  "model_id": "Qwen/Qwen2.5-3B-Instruct",  # ModelScope模型ID
  "cache_dir": "./models/qwen",             # 本地缓存路径
  "device": "auto"                          # 设备选择：auto/cpu/cuda
}
```

### TTS（语音合成 - Edge-TTS）
```python
{
  "voices": {
    "happy": "zh-CN-XiaoxiaoNeural",   # 开心语气
    "sad": "zh-CN-YunyangNeural",      # 低落语气
    "neutral": "zh-CN-YunxiNeural",    # 中性语气
    "angry": "zh-CN-YunjianNeural"     # 愤怒语气
  }
}
```

### STT（语音识别）
- **引擎**: Google Speech Recognition（需联网）
- **语言**: zh-CN（简体中文）
- **依赖**: `pip install SpeechRecognition`

---

## 环境变量

| 变量 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `TTS_ENABLED` | bool | 启用TTS语音合成和STT语音识别 | `true` |
| `MODELSCOPE_CACHE` | string | ModelScope模型缓存路径 | `./models` |

**禁用TTS示例**:
```bash
TTS_ENABLED=0 python app.py
```

---

## 输出文件结构

```
outputs/
├── audio/                      # 语音文件目录
│   └── {session_id}/           # 按对局ID分目录
│       ├── turn_1.wav          # 第1回合AI语音
│       ├── turn_2.wav          # 第2回合AI语音
│       └── ...
└── logs/                       # 日志文件目录
    └── talkarena_{YYYYMMDD_HHMMSS}.log  # 按启动时间命名
```

---

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt
pip install edge-tts SpeechRecognition

# 启动服务（含语音，默认）
python app.py

# 启动服务（不含语音）
TTS_ENABLED=0 python app.py
```

**服务地址**: `http://127.0.0.1:1234`

---

## 多人 Talking Head 架构

推荐入口：`core/multimodal_multiagent/orchestrator.py`

协议与文档：

- `core/multimodal_multiagent/contracts.py`
- `docs/ARCHITECTURE_MULTIMODAL_MULTIAGENT.md`
- `docs/MULTIMODAL_MULTIAGENT_TECH_IMPLEMENTATION.md`

该架构用于一对多场景下提升真实感：仅赢家角色执行重生成/TTS，其余角色持续输出非语言反应。
