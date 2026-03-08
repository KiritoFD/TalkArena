# TalkArena Prompt 总览

本文档列出 TalkArena 项目中所有的 LLM prompt 及其位置。

## Prompt 清单

### 1. 场景生成类 (4 个场景 × 2 种模式 = 8 个 prompt)

| 场景类型 | 模式 | Prompt 名称 | 位置 | 说明 |
|---------|------|-----------|------|------|
| 山东饭桌 | 完整版 | `SCENARIO_GENERATION_PROMPTS["shandong_dinner"]["full"]` | `core/prompts/registry.py:13` | 生成场景描述 +3 个角色 + 用户身份 |
| 山东饭桌 | 仅角色 | `SCENARIO_GENERATION_PROMPTS["shandong_dinner"]["characters_only"]` | `core/prompts/registry.py:38` | 仅生成 3 个角色 |
| 商务饭局 | 完整版 | `SCENARIO_GENERATION_PROMPTS["business_dinner"]["full"]` | `core/prompts/registry.py:55` | 生成场景描述 +3 个角色 + 用户身份 |
| 商务饭局 | 仅角色 | `SCENARIO_GENERATION_PROMPTS["business_dinner"]["characters_only"]` | `core/prompts/registry.py:80` | 仅生成 3 个角色 |
| 面试 | 完整版 | `SCENARIO_GENERATION_PROMPTS["interview"]["full"]` | `core/prompts/registry.py:97` | 生成场景描述 +2-3 个角色 |
| 面试 | 仅角色 | `SCENARIO_GENERATION_PROMPTS["interview"]["characters_only"]` | `core/prompts/registry.py:118` | 仅生成 2-3 个角色 |
| 辩论 | 完整版 | `SCENARIO_GENERATION_PROMPTS["debate"]["full"]` | `core/prompts/registry.py:137` | 生成场景描述 +3 个角色 |
| 辩论 | 仅角色 | `SCENARIO_GENERATION_PROMPTS["debate"]["characters_only"]` | `core/prompts/registry.py:158` | 仅生成 3 个角色 |

**调用位置：**
- `main.py:351-466` - `/api/scenario/generate` 接口
- `main.py:526-599` - `/api/scenario/regenerate` 接口

---

### 2. 对话生成类 (1 个 prompt)

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `DIALOGUE_GENERATION_PROMPT` | `core/prompts/registry.py:179` | NPC 对话生成，包含场景、角色、情感等信息 |

**调用位置：**
- `core/agents/multi_agent.py:207-240` - `_build_prompt` 方法

**参数：**
- scene_name: 场景名称
- speaker_name: 说话人名称
- atmosphere: 场景氛围
- rhetoric: 修辞要求
- personality: 角色设定
- style: 说话风格
- scene_description: 补充背景
- user_identity: 用户身份
- emotion_hint: 多模态情感提示
- user_input: 用户输入
- word_limit: 字数限制

---

### 3. 救场大师类 (1 个 prompt)

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `RESCUE_MASTER_PROMPT` | `core/prompts/registry.py:207` | 生成高情商救场建议 |

**调用位置：**
- `orchestrator.py:494-520` - `get_rescue_suggestion` 方法

**参数：**
- scene_name: 场景名称
- ai_name: 对手名称
- user_dominance: 用户气场
- ai_dominance: AI 气场
- context: 对话历史

---

### 4. 裁判类 (1 个 prompt)

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `DOMINANCE_JUDGE_PROMPT` | `core/prompts/registry.py:232` | 判断气场转移 |

**调用位置：**
- `orchestrator.py:626-655` - `_judge_dominance_zero_sum` 方法

**参数：**
- scene_name: 场景名称
- user_dominance: 用户气场
- ai_dominance: AI 气场
- user_text: 用户发言
- ai_text: AI 回应
- ai_name: AI 名称

---

### 5. 复盘报告类 (3 个 prompt)

#### 5.1 评分 Prompt

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `REPORT_SCORES_PROMPT` | `core/prompts/registry.py:257` | 生成五维度评分 |

**调用位置：**
- `orchestrator.py:776-813` - 复盘报告步骤 1

**评分维度：**
- oily: 圆滑度
- friendliness: 亲和力
- logic: 逻辑性
- humor: 幽默感
- respect: 懂规矩

#### 5.2 总结 Prompt

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `REPORT_SUMMARY_PROMPT` | `core/prompts/registry.py:297` | 生成综合点评 |

**调用位置：**
- `orchestrator.py:823-853` - 复盘报告步骤 2

#### 5.3 NPC 内心 OS Prompt

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `REPORT_NPC_INNER_VOICE_PROMPT` | `core/prompts/registry.py:333` | 生成 NPC 心理活动和建议 |

**调用位置：**
- `orchestrator.py:849-885` - 复盘报告步骤 3

---

### 6. 对决总结类 (1 个 prompt)

| Prompt 名称 | 位置 | 说明 |
|-----------|------|------|
| `DUEL_SUMMARY_PROMPT` | `core/prompts/registry.py:378` | 生成对决总结报告 |

**调用位置：**
- `orchestrator.py:699-725` - `_end_session_generate_report` 方法

---

## 统计

- **总计**: 15 个 prompt 模板
- **场景生成**: 8 个（4 场景 × 2 模式）
- **对话生成**: 1 个
- **救场大师**: 1 个
- **裁判**: 1 个
- **复盘报告**: 3 个
- **对决总结**: 1 个

## 文件分布

### 集中管理后
- ✅ `core/prompts/registry.py`: 所有 prompt 模板（15 个）
- ✅ `core/prompts/__init__.py`: 导出接口
- ✅ `core/prompts/README.md`: 使用说明

### 原分布位置（需要迁移）
- ❌ `main.py`: 场景生成 prompts（8 个）
- ❌ `core/agents/multi_agent.py`: 对话生成 prompt（1 个）
- ❌ `orchestrator.py`: 救场大师、裁判、复盘报告、对决总结（5 个）

## 迁移计划

### 阶段 1: 创建 registry ✅
- [x] 创建 `core/prompts/registry.py`
- [x] 创建 `core/prompts/__init__.py`
- [x] 创建 `core/prompts/README.md`
- [x] 验证导入正常

### 阶段 2: 迁移代码（待执行）
- [ ] 迁移 `main.py` 中的场景生成 prompts
- [ ] 迁移 `core/agents/multi_agent.py` 中的对话生成 prompt
- [ ] 迁移 `orchestrator.py` 中的其他 prompts

### 阶段 3: 测试验证（待执行）
- [ ] 测试场景生成功能
- [ ] 测试对话生成功能
- [ ] 测试救场大师功能
- [ ] 测试复盘报告功能
- [ ] 测试对决总结功能

## 使用示例

### 场景生成
```python
from core.prompts import get_scenario_generation_prompt

# 获取模板
prompt = get_scenario_generation_prompt("shandong_dinner", only_characters=False)

# 格式化
formatted = prompt.format(scene_name="家庭饭桌试炼")

# 调用 LLM
response = llm.generate(formatted, max_new_tokens=1500)
```

### 对话生成
```python
from core.prompts import get_dialogue_generation_prompt

prompt = get_dialogue_generation_prompt(
    scene_name="家庭饭桌",
    speaker_name="李大伯",
    atmosphere="温馨热闹",
    rhetoric="幽默风趣",
    personality="开朗热情的长辈",
    style="口语化、喜欢用俗语",
    scene_description="周末家庭聚会",
    user_identity="用户身份：小明 / 晚辈 / 刚工作的大学生",
    emotion_hint="情感：开心，自信度 80",
    user_input="谢谢大伯关心",
    word_limit=60
)

response = llm.generate(prompt, max_new_tokens=120)
```

### 救场大师
```python
from core.prompts import get_rescue_master_prompt

prompt = get_rescue_master_prompt(
    scene_name="商务谈判",
    ai_name="王总",
    user_dominance=45,
    ai_dominance=55,
    context="用户：我同意您的看法\nAI：很好，那我们继续..."
)

suggestion = llm.generate(prompt, max_new_tokens=150)
```

## 优势

1. **集中管理**: 所有 prompt 在一个文件，方便查看和维护
2. **易于更新**: 修改 prompt 不需要改动业务逻辑
3. **可测试**: 可以单独测试每个 prompt 的效果
4. **可复用**: 不同模块可以共享相同的 prompt
5. **版本控制**: 可以追踪 prompt 的历史变更
6. **文档化**: 每个 prompt 都有清晰的注释和说明

## 下一步

1. 将现有代码中的 prompt 调用迁移到使用 registry
2. 测试所有功能确保正常工作
3. 更新相关文档
4. 提交代码
