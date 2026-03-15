# 🎯 TalkArena 表达训练营

> 一个把“社交压力”变成“表达训练数据”的多模态实战平台。

**在线体验（Vercel）**：https://talk-arena.vercel.app/

---

## 我们最核心的能力：多模态表达训练

TalkArena 不是普通聊天机器人，而是围绕“真实社交表达”打造的多模态训练系统：

- **文本信号**：你说了什么（内容、逻辑、措辞）
- **语音信号**：你怎么说（语速、停顿、情绪张力）
- **表情信号**：你当下状态（紧张、迟疑、自信等面部线索）

系统会把这三路信息汇总成统一状态，用来驱动：
- 对话中的实时反馈（例如紧张度变化）
- NPC 的压力感知与互动节奏
- 局后复盘的多维评分与建议

一句话：**你练的不只是“台词”，而是完整的临场表达能力。**

---

## 🔐 隐私安全（重点）

这是我们非常重视的一点：

- **表情识别在前端本地处理**（浏览器侧完成）
- **不上传原始摄像头视频流** 到服务端
- 上传的是**低维度的情绪/特征结果**，用于对话反馈与评估

这意味着：
- 你能获得实时表情反馈能力；
- 同时最大化保护你的影像隐私。

> 我们的设计目标是：在“可用的多模态智能”和“用户隐私安全”之间取得平衡。

---

## 📸 在线界面预览（Vercel）

### 首页
![TalkArena 首页](browser:/tmp/codex_browser_invocations/5ebafbae49ab221c/artifacts/shots/home.png)

### 第二页：可选场景配置页
![TalkArena 场景配置页](browser:/tmp/codex_browser_invocations/5ebafbae49ab221c/artifacts/shots/scene-config.png)

### 对话页
![TalkArena 对话页](browser:/tmp/codex_browser_invocations/5ebafbae49ab221c/artifacts/shots/dialogue.png)

---

## 场景与训练闭环

### 1) 真实多人场景
- 家庭饭桌
- 商务饭局
- 群面竞争

### 2) 多角色施压互动
- 主导者、观察者、气氛组协同
- 还原真实社交中的“多人动态压力”

### 3) 实时救场与反馈
- 过程评分（自信度/平静度/紧张度等）
- 一键救场建议，避免卡壳

### 4) 局后复盘
- 多维评分 + 综合点评 + NPC 视角反馈
- 从“这句说错了”到“下次怎么说更好”

---

## 技术亮点（简版）

- FastAPI 服务端 + 多 Agent 编排
- 多模态状态融合（文本/语音/表情）
- 结构化输出约束，降低对话跑偏
- TTS 预取与串行播放，增强沉浸感
- 会话快照机制，提升稳定性

---

## 🚀 本地运行

```bash
pip install -r requirements.txt
python main.py
```

必需环境变量：
- `SILICONFLOW_API_KEY`

可选：
- `SILICONFLOW_BASE_URL`
- `LLM_MODEL`
- `TALKARENA_FAST_MODE=1`

默认访问：
- 本地：`http://127.0.0.1:8000`
- 线上：`https://talk-arena.vercel.app/`

---

## 愿景

让表达能力像体能一样：
**可训练、可量化、可持续提升。**

欢迎提 Issue / PR，一起把 TalkArena 打磨成真正有用的表达训练基础设施。
