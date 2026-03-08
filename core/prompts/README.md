# Prompt 管理模块

## 概述

本模块集中管理 TalkArena 项目中所有的 LLM prompt 模板，方便统一查看、维护和更新。

## 文件结构

```
core/prompts/
├── __init__.py          # 模块导出
├── registry.py          # Prompt 模板注册表
└── README.md           # 使用说明
```

## Prompt 分类

### 1. 场景生成 Prompts (`SCENARIO_GENERATION_PROMPTS`)

用于 AI 生成场景设定和角色信息，支持以下场景类型：
- `shandong_dinner`: 山东饭桌场景
- `business_dinner`: 商务饭局场景
- `interview`: 面试场景
- `debate`: 辩论场景

每个场景类型包含两种模式：
- `full`: 生成完整场景（场景描述 + 角色列表 + 用户身份）
- `characters_only`: 仅生成角色列表

**使用示例：**
```python
from core.prompts import get_scenario_generation_prompt

# 获取完整场景生成 prompt
prompt = get_scenario_generation_prompt("shandong_dinner", only_characters=False)
formatted = prompt.format(scene_name="家庭饭桌试炼")

# 获取仅角色生成 prompt
prompt = get_scenario_generation_prompt("shandong_dinner", only_characters=True)
```

### 2. 对话生成 Prompt (`DIALOGUE_GENERATION_PROMPT`)

用于 NPC 对话生成，包含：
- 场景信息
- 角色设定
- 说话风格
- 多模态情感提示
- 用户输入

**使用示例：**
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
```

### 3. 救场大师 Prompt (`RESCUE_MASTER_PROMPT`)

用于生成高情商救场建议，包含：
- 场景信息
- 对手信息
- 当前气场
- 对话历史

**使用示例：**
```python
from core.prompts import get_rescue_master_prompt

prompt = get_rescue_master_prompt(
    scene_name="商务谈判",
    ai_name="王总",
    user_dominance=45,
    ai_dominance=55,
    context="用户：我同意您的看法\nAI：很好，那我们继续..."
)
```

### 4. 裁判 Prompt (`DOMINANCE_JUDGE_PROMPT`)

用于判断对话中气场转移，包含：
- 场景信息
- 当前气场
- 双方发言
- 评判维度

**使用示例：**
```python
from core.prompts import get_dominance_judge_prompt

prompt = get_dominance_judge_prompt(
    scene_name="辩论赛",
    user_dominance=50,
    ai_dominance=50,
    user_text="我认为这个观点有问题...",
    ai_text="我不同意你的看法...",
    ai_name="反方辩手"
)
```

### 5. 复盘报告 Prompts

#### 5.1 评分 Prompt (`REPORT_SCORES_PROMPT`)

用于生成五维度评分：
- 圆滑度 (oily)
- 亲和力 (friendliness)
- 逻辑性 (logic)
- 幽默感 (humor)
- 懂规矩 (respect)

**使用示例：**
```python
from core.prompts import get_report_scores_prompt

prompt = get_report_scores_prompt(
    scene_name="家庭饭桌",
    npc_list='[{"name": "李大伯", "role": "长辈"}]',
    history_log="用户：谢谢\nNPC：不客气"
)
```

#### 5.2 总结 Prompt (`REPORT_SUMMARY_PROMPT`)

用于生成综合点评，要求：
- 犀利度：像严厉的长辈
- 专业深度：基于社交潜规则
- 称号挂钩：匹配玩家称号
- 结构化：定性 - 分析 - 总结

**使用示例：**
```python
from core.prompts import get_report_summary_prompt

prompt = get_report_summary_prompt(
    scene_name="家庭饭桌",
    npc_list='[{"name": "李大伯", "role": "长辈"}]',
    history_log="用户：谢谢\nNPC：不客气",
    medal="社交达人"
)
```

#### 5.3 NPC 内心 OS Prompt (`REPORT_NPC_INNER_VOICE_PROMPT`)

用于生成 NPC 心理活动和高情商建议

**使用示例：**
```python
from core.prompts import get_report_npc_inner_voice_prompt

prompt = get_report_npc_inner_voice_prompt(
    scene_name="家庭饭桌",
    npc_list='[{"name": "李大伯", "role": "长辈"}]',
    history_log="用户：谢谢\nNPC：不客气",
    medal="社交达人"
)
```

### 6. 对决总结 Prompt (`DUEL_SUMMARY_PROMPT`)

用于生成对决总结报告，包含：
- 对决结果
- 表现分析
- 关键回合复盘
- 改进建议

**使用示例：**
```python
from core.prompts import get_duel_summary_prompt

prompt = get_duel_summary_prompt(
    scene_name="商务谈判",
    ai_name="王总",
    user_dominance=60,
    ai_dominance=40,
    turn_count=10,
    dialogue="用户：...\nAI：...",
    result="🏆 用户获胜"
)
```

## 工具函数

### `format_prompt(template, **kwargs)`

通用 prompt 格式化函数

**参数：**
- `template`: prompt 模板字符串
- `**kwargs`: 模板变量

**返回：**
- 格式化后的 prompt

### 各场景专用函数

- `get_scenario_generation_prompt(scene_type, only_characters)`
- `get_dialogue_generation_prompt(...)`
- `get_rescue_master_prompt(...)`
- `get_dominance_judge_prompt(...)`
- `get_report_scores_prompt(...)`
- `get_report_summary_prompt(...)`
- `get_report_npc_inner_voice_prompt(...)`
- `get_duel_summary_prompt(...)`

## 迁移指南

### 从旧代码迁移

**之前（在 main.py 或 orchestrator.py 中）：**
```python
prompt = f"""
请为一场山东饭桌场景生成以下内容：
1. 详细的场景背景描述...
当前场景名称：{req.scene_name}
...
"""
```

**现在（使用 prompt registry）：**
```python
from core.prompts import get_scenario_generation_prompt

prompt_template = get_scenario_generation_prompt("shandong_dinner")
prompt = prompt_template.format(scene_name=req.scene_name)
```

### 迁移步骤

1. 识别代码中的 prompt 字符串
2. 在 `registry.py` 中找到对应的模板
3. 使用工具函数获取模板并格式化
4. 测试确保功能正常

## 最佳实践

1. **统一管理**: 所有 prompt 都应该在 `registry.py` 中定义
2. **命名规范**: prompt 变量使用大写命名，如 `DIALOGUE_GENERATION_PROMPT`
3. **文档化**: 每个 prompt 都应该有清晰的注释说明用途
4. **参数化**: 使用 `{variable}` 占位符，通过 `format()` 填充
5. **版本控制**: prompt 修改应该提交 git 并写清楚变更原因

## 添加新 Prompt

1. 在 `registry.py` 中定义新的 prompt 模板
2. 添加对应的工具函数（如需要）
3. 在 `__init__.py` 中导出
4. 更新本文档

## 优势

- ✅ **集中管理**: 所有 prompt 在一个地方，方便查看和维护
- ✅ **易于更新**: 修改 prompt 不需要改动业务逻辑代码
- ✅ **可测试**: 可以单独测试 prompt 的效果
- ✅ **可复用**: 不同模块可以共享相同的 prompt
- ✅ **版本控制**: 可以追踪 prompt 的历史变更
