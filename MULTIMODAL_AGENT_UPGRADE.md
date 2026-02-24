# TalkArena 深度多模态处理与多Agent系统升级技术方案

## 一、概述与设计目标

### 1.1 项目背景

TalkArena 是一个基于 AI 的酒桌情商训练与实战对练平台，当前版本已具备基础的多模态分析与多Agent协同能力。在现有架构中，系统能够通过摄像头捕捉用户表情特征、通过麦克风采集语音信息，并将这些数据作为上下文参考传递给对话Agent。然而，现有实现存在以下局限性：

**当前问题分析：**

| 问题维度 | 具体表现 | 影响程度 |
|---------|---------|---------|
| 情感感知深度不足 | 仅提取自信度、紧张度等表层指标 | ★★★☆☆ |
| 语音分析粗放 | 仅检测音量大小，未分析语速、语调、停顿 | ★★☆☆☆ |
| NPC反应单一 | 多模态信息仅影响对话策略提示，未深入行为控制 | ★★★☆☆ |
| Agent协同表面 | 各Agent独立运作，缺乏共享情感状态与行为协调 | ★★☆☆☆ |
| 沉浸感有限 | 用户情感变化未实时反馈到NPC的表情、动作、语气 | ★★★★☆ |

### 1.2 升级目标

本次技术升级的核心目标是构建**深度情感驱动的多模态NPC行为控制系统**，使AI NPC能够像真人一样感知、理解并响应用户的细微情感变化，从而实现前所未有的沉浸式对话体验。

**具体目标分解：**

**目标一：多模态感知层升级**

构建端到端的多模态特征提取 pipeline，支持：

- **表情微表情分析**：不仅识别基本情绪，更要检测眉毛上扬、嘴角微动、眼眶变化等微观表情
- **语音声纹情感分析**：从语音频谱中提取情感特征，识别用户的真实心理状态（而非仅依赖文本语义）
- **多模态融合**：将表情、语音、生理信号（如有）进行时序融合，生成用户的综合情感画像
- **实时情感追踪**：建立用户情感时序数据库，记录每个回合的情感变化曲线

**目标二：NPC行为控制层升级**

构建情感驱动的NPC行为引擎，实现：

- **情绪传染机制**：NPC能够感知用户情绪，产生相应的共情或对抗反应
- **动态行为选择**：基于用户情感状态，实时选择NPC的表情、动作、语气、台词风格
- **记忆与学习**：NPC记住用户的历史情感模式，在后续对话中加以利用
- **多NPC协同**：多个NPC之间能够共享用户情感信息，协调配合施加压力或给予支持

**目标三：多Agent系统架构升级**

重构现有的多Agent协同框架：

- **情感协调Agent**：新增专门的情感协调器，负责全局情感状态管理
- **行为规划Agent**：负责将情感状态转换为具体的NPC行为指令
- **角色扮演Agent池**：每个NPC角色拥有独立的Agent，具备独特的性格与情感模型
- **评估Agent升级**：情感驱动的评估算法，综合考虑用户的多模态表现

**目标四：沉浸感体验升级**

通过以下技术手段大幅提升用户体验：

- **实时情感可视化**：NPC实时显示对用户情感的理解与反应
- **动态场景渲染**：基于用户情感状态调整场景氛围、光线、背景音乐
- **智能提示系统**：根据用户情感状态提供个性化的情商提升建议

---

## 二、系统架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TalkArena 深度升级架构                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                            前端交互层                                     │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │    │
│  │  │   视频输入   │  │   音频输入   │  │  文本输入    │  │ 情感可视化 │ │    │
│  │  │  (WebRTC)   │  │  (WebRTC)   │  │  (Gradio)   │  │  (实时)    │ │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │    │
│  └─────────┼─────────────────┼──────────────────┼────────────────┼────────┘    │
│            │                 │                  │                │               │
│            ▼                 ▼                  ▼                ▼               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         多模态感知层                                     │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐     │    │
│  │  │  表情分析模块  │  │  语音分析模块  │  │    多模态融合引擎     │     │    │
│  │  │  (Face API)   │  │  (Vosk+情感)   │  │  (时序融合+权重调整)  │     │    │
│  │  └───────┬────────┘  └───────┬────────┘  └──────────┬───────────┘     │    │
│  │          │                   │                      │                  │    │
│  │          ▼                   ▼                      ▼                  │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐    │    │
│  │  │                    情感状态数据库                                 │    │    │
│  │  │  - 当前情感向量 (128维)                                          │    │    │
│  │  │  - 情感时序曲线 (最近N轮)                                        │    │    │
│  │  │  - 情感记忆块 (重要事件标记)                                      │    │    │
│  │  └──────────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      深度情感驱动 NPC 行为引擎                           │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │    │
│  │  │                     情感协调 Agent (新增)                        │   │    │
│  │  │  - 全局情感状态管理                                               │   │    │
│  │  │  - 多NPC情感协调                                                  │   │    │
│  │  │  - 行为策略规划                                                   │   │    │
│  │  └──────────────────────────────────────────────────────────────────┘   │    │
│  │                                    │                                      │    │
│  │          ┌─────────────────────────┼─────────────────────────┐          │    │
│  │          ▼                         ▼                         ▼          │    │
│  │  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐   │    │
│  │  │  角色Agent池   │      │   评估Agent    │      │   记忆Agent    │   │    │
│  │  │  (每角色独立)  │◄────►│  (情感驱动)    │◄────►│  (情感记忆)    │   │    │
│  │  │                │      │                │      │                │   │    │
│  │  │  - 大舅Agent   │      │  - 多模态评分  │      │  - 情感事件    │   │    │
│  │  │  - 大妗子Agent │      │  - 实时判定    │      │  - 偏好学习    │   │    │
│  │  │  - 表哥Agent   │      │  - 策略建议    │      │  - 模式识别    │   │    │
│  │  └────────────────┘      └────────────────┘      └────────────────┘   │    │
│  │                                    │                                      │    │
│  │                                    ▼                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │    │
│  │  │                    NPC 行为指令生成器                             │   │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │   │    │
│  │  │  │  表情指令  │ │  语气指令  │ │  动作指令  │ │  台词生成  │   │   │    │
│  │  │  │ (微表情)   │ │ (语调速度) │ │  (肢体)    │ │ (情感适配) │   │   │    │
│  │  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │   │    │
│  │  └──────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          LLM 决策层                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Qwen3-4B / Qwen2-7B-Instruct (ModelScope)                       │   │    │
│  │  │  - 情感感知Prompt构建                                            │   │    │
│  │  │  - 行为决策生成                                                   │   │    │
│  │  │  - 裁判评分                                                       │   │    │
│  │  └──────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块说明

**模块一：多模态感知层（Multimodal Perception Layer）**

负责从原始输入中提取高质量的情感特征，是整个系统的输入基础。

| 模块 | 功能描述 | 技术选型 |
|-----|---------|---------|
| 表情分析器 | 实时人脸检测、关键点定位、微表情识别 | MediaPipe Face Mesh + 自研微表情分类器 |
| 语音情感分析器 | 语音特征提取、情感分类、语速/语调分析 | librosa特征 + 轻量级情感分类模型 |
| 多模态融合引擎 | 时序对齐、特征融合、权重动态调整 | Transformer Cross-Modal Fusion |
| 情感状态数据库 | 存储用户情感向量、时序数据、记忆块 | SQLite + 内存缓存 |

**模块二：情感驱动NPC行为引擎（Emotion-Driven NPC Behavior Engine）**

这是本次升级的核心模块，负责将情感状态转换为NPC的具体行为。

| 模块 | 功能描述 | 技术选型 |
|-----|---------|---------|
| 情感协调Agent | 全局情感状态管理、多NPC协调、策略规划 | 情感状态机 + 规则引擎 |
| 角色Agent池 | 每个NPC独立Agent、个性化学反应 | 基于Prompt的角色LLM |
| 评估Agent | 多模态综合评分、实时判定 | 多维度加权评分算法 |
| 记忆Agent | 情感事件存储、偏好学习、模式识别 | 向量数据库 + 时序模型 |

**模块三：NPC行为指令生成器（NPC Behavior Instruction Generator）**

负责将高层行为决策转换为具体的执行指令。

| 指令类型 | 说明 | 示例 |
|---------|------|------|
| 表情指令 | NPC需要表现出的微表情 | 微微皱眉、轻蔑一笑、眼睛微眯 |
| 语气指令 | 语音合成的语调参数 | 语速+20%、音调提高、半朗读式 |
| 动作指令 | NPC的肢体动作描述 | 拍桌子、端起酒杯、摇头 |
| 台词生成 | 最终输出的对话文本 | 融合情感色彩的对话内容 |

---

## 三、多模态感知层技术实现

### 3.1 表情分析模块升级

**3.1.1 技术架构**

```
┌─────────────────────────────────────────────────────────────┐
│                    表情分析 Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   视频流 ──► 人脸检测 ──► 关键点 ──► 微表情 ──► 情感向量    │
│   (30fps)   (MediaPipe)  (468点)  (分类器)   (128维)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**3.1.2 微表情特征定义**

在基础情绪（开心、悲伤、愤怒、惊讶、恐惧、厌恶）之上，增加以下微表情维度：

```python
class MicroExpressionFeatures:
    """微表情特征数据结构 - 扩展至50维向量"""
    
    # 眼部区域 (18维)
    eye_openness_left: float      # 左眼睁开度 [0-1]
    eye_openness_right: float     # 右眼睁开度 [0-1]
    eye_aspect_ratio: float       # 眼睛长宽比 (眨眼检测)
    pupil_x_left: float           # 左瞳孔水平位置
    pupil_y_left: float           # 左瞳孔垂直位置
    pupil_x_right: float          # 右瞳孔水平位置
    pupil_y_right: float          # 右瞳孔垂直位置
    brow_inner_up_left: float     # 左内侧眉毛上扬
    brow_inner_up_right: float   # 右内侧眉毛上扬
    brow_outer_up_left: float     # 左外侧眉毛上扬
    brow_outer_up_right: float   # 右外侧眉毛上扬
    brow_tension: float           # 眉毛紧张度
    gaze_direction_x: float       # 视线水平方向 [-1,1]
    gaze_direction_y: float       # 视线垂直方向 [-1,1]
    blink_rate: float             # 眨眼频率 (次/分钟)
    saccade_magnitude: float      # 眼动幅度
    
    # 嘴部区域 (12维)
    mouth_open_ratio: float       # 嘴巴张开程度 [0-1]
    mouth_width: float             # 嘴巴宽度变化
    lip_corner_up_left: float     # 左嘴角上扬
    lip_corner_up_right: float   # 右嘴角上扬
    lip_corner_down_left: float   # 左嘴角下垂
    lip_corner_down_right: float # 右嘴角下垂
    lip_pressure: float            # 嘴唇压力 (咬唇检测)
    smile_asymmetry: float         # 笑容对称度
    smile_genuine_score: float    # 真诚笑容指数 (Duchenne检测)
    jaw_open: float               # 下颌张开度
    jaw_shift: float              # 下颌偏移
    
    # 眉心鼻梁区域 (8维)
    nose_wrinkle: float           # 鼻子皱纹
    nasolabial_fold_left: float   # 左侧鼻唇沟深度
    nasolabial_fold_right: float  # 右侧鼻唇沟深度
    forehead_wrinkle: float        # 额头皱纹
    
    # 整体情感推断 (12维)
    valence: float                # 效价: 消极-积极 [-1,1]
    arousal: float                # 唤醒度: 平静-激动 [0,1]
    dominance: float              # 控制感: 被控-掌控 [0,1]
    happiness: float              # 幸福感
    sadness: float               # 悲伤感
    anger: float                 # 愤怒感
    fear: float                  # 恐惧感
    surprise: float              # 惊讶感
    disgust: float               # 厌恶感
    contempt: float              # 轻蔑感
    interest: float              # 感兴趣程度
    confusion: float             # 困惑程度
    
    # 时序特征 (时序数据)
    emotion_trajectory: List[float]   # 情感变化轨迹
    stress_indicators: Dict           # 压力指标汇总
```

**3.1.3 情感状态机**

基于微表情特征，构建用户的情感状态机：

```python
class UserEmotionStateMachine:
    """用户情感状态机 - 用于平滑情感检测结果"""
    
    EMOTION_STATES = [
        "confident",      # 自信从容
        "nervous",        # 紧张不安
        "angry",          # 愤怒不满
        "happy",          # 开心愉悦
        "sad",            # 伤心失落
        "surprised",      # 惊讶意外
        "confused",       # 困惑迷茫
        "contemptuous",   # 轻蔑不屑
        "neutral",        # 中性平静
    ]
    
    def __init__(self):
        self.current_state = "neutral"
        self.state_confidence = 0.5
        self.state_history = []  # 状态转移历史
        self.frame_buffer = []   # 帧缓冲用于平滑
        
    def update(self, micro_features: MicroExpressionFeatures) -> Tuple[str, float]:
        """更新状态机，返回(状态, 置信度)"""
        
        # 1. 提取当前帧的情感向量
        current_emotion = self._extract_emotion_vector(micro_features)
        
        # 2. 平滑处理 (时序滤波)
        smoothed_emotion = self._temporal_smoothing(current_emotion)
        
        # 3. 状态推断
        new_state, confidence = self._infer_state(smoothed_emotion)
        
        # 4. 状态转移检测
        if new_state != self.current_state:
            if confidence > 0.7:  # 高置信度才转移
                self._record_transition(new_state, confidence)
                
        return self.current_state, self.state_confidence
    
    def _extract_emotion_vector(self, features: MicroExpressionFeatures) -> np.ndarray:
        """从微表情特征提取情感向量"""
        
        # 效价-唤醒度-控制感 (VAD) 推断
        valence = (
            features.happiness * 0.3 +
            features.smile_genuine_score * 0.3 +
            features.brow_tension * (-0.2) +
            features.lip_corner_up_left * 0.2 +
            features.lip_corner_up_right * 0.2 -
            features.sadness * 0.3 -
            features.contempt * 0.2
        )
        
        arousal = (
            features.eye_openness_left * 0.2 +
            features.eye_openness_right * 0.2 +
            features.brow_tension * 0.3 +
            features.jaw_open * 0.2 +
            features.mouth_open_ratio * 0.2 -
            features.blink_rate * (-0.1)
        )
        
        dominance = (
            features.gaze_direction_x * 0.2 +
            features.brow_outer_up_left * 0.2 +
            features.brow_outer_up_right * 0.2 +
            features.smile_asymmetry * (-0.1) +
            features.lip_pressure * 0.2
        )
        
        # 特定情绪检测
        happiness = features.happiness + features.smile_genuine_score * 0.5
        sadness = features.sadness + features.brow_inner_up_left * 0.3
        anger = features.anger + features.brow_tension * 0.4 + features.nose_wrinkle * 0.3
        fear = features.fear + features.eye_openness_left * 0.3
        surprise = features.surprise + features.brow_outer_up_left * 0.3
        disgust = features.disgust + features.nose_wrinkle * 0.3
        contempt = features.contempt + features.lip_corner_down_left * 0.3
        confusion = features.confusion + features.jaw_shift * 0.2
        
        return np.array([
            valence, arousal, dominance,
            happiness, sadness, anger, fear, surprise, disgust, contempt,
            interest, confusion
        ], dtype=np.float32)
    
    def _temporal_smoothing(self, current: np.ndarray) -> np.ndarray:
        """时序平滑 - 使用指数移动平均"""
        
        if len(self.frame_buffer) == 0:
            smoothed = current
        else:
            alpha = 0.3  # 平滑系数
            smoothed = alpha * current + (1 - alpha) * self.frame_buffer[-1]
            
        self.frame_buffer.append(smoothed)
        if len(self.frame_buffer) > 30:  # 保留最近30帧
            self.frame_buffer.pop(0)
            
        return smoothed
    
    def _infer_state(self, emotion_vector: np.ndarray) -> Tuple[str, float]:
        """从情感向量推断离散状态"""
        
        valence, arousal, dominance = emotion_vector[0], emotion_vector[1], emotion_vector[2]
        happiness = emotion_vector[3]
        sadness = emotion_vector[4]
        anger = emotion_vector[5]
        fear = emotion_vector[6]
        surprise = emotion_vector[7]
        contempt = emotion_vector[9]
        confusion = emotion_vector[11]
        
        # 状态推断规则
        rules = []
        
        # 自信状态: 高效价 + 高控制感
        if valence > 0.3 and dominance > 0.5:
            rules.append(("confident", valence * dominance))
            
        # 紧张状态: 高唤醒 + 低效价
        if arousal > 0.6 and valence < 0.2:
            rules.append(("nervous", arousal * (1 - valence)))
            
        # 愤怒状态: 低效价 + 高唤醒 + 低控制感
        if valence < -0.2 and arousal > 0.5 and dominance < 0.5:
            rules.append(("angry", -valence * arousal))
            
        # 开心状态: 高效价
        if happiness > 0.6:
            rules.append(("happy", happiness))
            
        # 悲伤状态: 低效价 + 低唤醒
        if sadness > 0.5 and arousal < 0.4:
            rules.append(("sad", sadness))
            
        # 惊讶状态: 高唤醒 + 中效价
        if surprise > 0.5 and 0.3 > arousal > 0.6:
            rules.append(("surprised", surprise))
            
        # 轻蔑状态: 低效价 + 低唤醒 + 高控制感
        if contempt > 0.4 and dominance > 0.5:
            rules.append(("contemptuous", contempt * dominance))
            
        # 困惑状态: 低控制感
        if confusion > 0.4:
            rules.append(("confused", confusion))
            
        if not rules:
            return "neutral", 0.5
            
        # 选择置信度最高的状态
        best_state, best_score = max(rules, key=lambda x: x[1])
        confidence = min(0.95, best_score)
        
        return best_state, confidence
```

### 3.2 语音情感分析模块升级

**3.2.1 语音情感特征提取**

```python
class VoiceEmotionAnalyzer:
    """语音情感分析器 - 提取多维语音特征"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = 2048
        self.hop_length = 512
        
    def analyze(self, audio_data: np.ndarray) -> VoiceEmotionFeatures:
        """全面分析语音情感特征"""
        
        features = VoiceEmotionFeatures()
        
        # 1. 基础声学特征
        features.loudness = self._compute_loudness(audio_data)
        features.speech_rate = self._compute_speech_rate(audio_data)
        features.pitch_mean, features.pitch_std = self._compute_pitch_stats(audio_data)
        features.energy_variance = self._compute_energy_variance(audio_data)
        
        # 2. 频谱特征
        features.mfcc = self._compute_mfcc(audio_data)  # 13维
        features.spectral_centroid = self._compute_spectral_centroid(audio_data)
        features.spectral_rolloff = self._compute_spectral_rolloff(audio_data)
        features.spectral_flux = self._compute_spectral_flux(audio_data)
        
        # 3. 韵律特征
        features.pitch_range = self._compute_pitch_range(audio_data)
        features.intensity_range = self._compute_intensity_range(audio_data)
        features.stress_indicators = self._detect_stress(audio_data)
        
        # 4. 音质特征
        features.harmonic_ratio = self._compute_harmonic_ratio(audio_data)
        features.noise_ratio = self._compute_noise_ratio(audio_data)
        features.jitter = self._compute_jitter(audio_data)
        features.shimmer = self._compute_shimmer(audio_data)
        
        # 5. 情感推断
        features.emotion_scores = self._infer_emotion(features)
        
        return features
    
    def _infer_emotion(self, features: VoiceEmotionFeatures) -> Dict[str, float]:
        """从语音特征推断情感 - 基于规则 + 轻量模型"""
        
        scores = {}
        
        # 开心: 高音调 + 高能量 + 快语速 + 高谐波比
        happiness_score = (
            min(1.0, features.pitch_mean / 300) * 0.3 +
            min(1.0, features.loudness / 0.8) * 0.3 +
            min(1.0, features.speech_rate / 4.0) * 0.2 +
            min(1.0, features.harmonic_ratio / 0.8) * 0.2
        )
        scores["happy"] = happiness_score
        
        # 悲伤: 低音调 + 低能量 + 慢语速 + 低谐波比
        sadness_score = (
            (1 - min(1.0, features.pitch_mean / 200)) * 0.3 +
            (1 - min(1.0, features.loudness / 0.5)) * 0.3 +
            (1 - min(1.0, features.speech_rate / 3.0)) * 0.2 +
            (1 - min(1.0, features.harmonic_ratio / 0.6)) * 0.2
        )
        scores["sad"] = sadness_score
        
        # 愤怒: 高能量 + 高音调变化 + 快语速 + 高噪声
        anger_score = (
            min(1.0, features.loudness / 0.9) * 0.3 +
            min(1.0, features.pitch_std / 100) * 0.2 +
            min(1.0, features.speech_rate / 4.5) * 0.2 +
            min(1.0, features.noise_ratio / 0.3) * 0.3
        )
        scores["angry"] = anger_score
        
        # 紧张: 高音调 + 不稳定能量 + 频繁停顿
        nervousness_score = (
            min(1.0, features.pitch_mean / 250) * 0.25 +
            min(1.0, features.energy_variance / 0.2) * 0.25 +
            min(1.0, features.stress_indicators["pause_frequency"]) * 0.3 +
            min(1.0, features.jitter / 0.05) * 0.2
        )
        scores["nervous"] = nervousness_score
        
        # 自信: 中高音调 + 稳定能量 + 流畅语速
        confidence_score = (
            min(1.0, (features.pitch_mean - 100) / 150) * 0.2 +
            (1 - min(1.0, features.energy_variance / 0.15)) * 0.3 +
            min(1.0, features.speech_rate / 3.5) * 0.2 +
            (1 - min(1.0, features.jitter / 0.03)) * 0.3
        )
        scores["confident"] = confidence_score
        
        # 犹豫: 低能量 + 不稳定语速 + 高停顿
        hesitation_score = (
            (1 - min(1.0, features.loudness / 0.4)) * 0.25 +
            min(1.0, features.stress_indicators["pause_frequency"]) * 0.35 +
            min(1.0, features.stress_indicators["repetition_rate"]) * 0.2 +
            min(1.0, features.shimmer / 0.1) * 0.2
        )
        scores["hesitant"] = hesitation_score
        
        return scores
    
    def _detect_stress(self, audio_data: np.ndarray) -> Dict[str, float]:
        """检测语音压力指标"""
        
        indicators = {}
        
        # 1. 停顿检测
        energy = np.abs(audio_data)
        threshold = np.percentile(energy, 20)
        is_silence = energy < threshold
        
        # 计算停顿频率
        pause_frames = np.sum(is_silence) / len(is_silence)
        indicators["pause_frequency"] = pause_frames
        
        # 2. 重复检测 (基于MFCC变化)
        mfcc = self._compute_mfcc(audio_data)
        mfcc_diff = np.diff(mfcc, axis=0)
        repetition_rate = np.sum(np.std(mfcc_diff, axis=0) < 0.1) / mfcc.shape[1]
        indicators["repetition_rate"] = repetition_rate
        
        # 3. 语速变化
        indicators["speech_rate_variance"] = np.std(features.speech_rate) if hasattr(features, 'speech_rate') else 0
        
        # 4. 声音颤抖 (jitter相关)
        indicators["tremolo"] = features.jitter if hasattr(features, 'jitter') else 0
        
        return indicators


@dataclass
class VoiceEmotionFeatures:
    """语音情感特征 - 扩展至60维"""
    
    # 基础声学特征 (8维)
    loudness: float = 0.0           # 整体响度 [0-1]
    speech_rate: float = 0.0        # 语速 (字/秒)
    pitch_mean: float = 0.0          # 平均基频 (Hz)
    pitch_std: float = 0.0           # 基频标准差
    energy_variance: float = 0.0    # 能量方差
    
    # 频谱特征 (20维)
    mfcc: np.ndarray = None         # 13维MFCC
    spectral_centroid: float = 0.0  # 频谱质心
    spectral_rolloff: float = 0.0   # 频谱滚降点
    spectral_flux: float = 0.0      # 频谱通量
    
    # 韵律特征 (8维)
    pitch_range: float = 0.0         # 基频范围
    intensity_range: float = 0.0     # 强度范围
    stress_indicators: Dict = None  # 压力指标
    
    # 音质特征 (6维)
    harmonic_ratio: float = 0.0      # 谐波比
    noise_ratio: float = 0.0         # 噪声比
    jitter: float = 0.0             # 频率抖动
    shimmer: float = 0.0             # 振幅抖动
    
    # 情感推断 (6维)
    emotion_scores: Dict[str, float] = None  # 各情感得分
    
    # 综合情感 (3维)
    valence: float = 0.0             # 效价
    arousal: float = 0.0             # 唤醒度
    dominance: float = 0.0           # 控制感
```

### 3.3 多模态融合引擎

**3.3.1 时序对齐与融合**

```python
class MultimodalFusionEngine:
    """多模态融合引擎 - 表情与语音的时序融合"""
    
    def __init__(self):
        self.face_buffer = deque(maxlen=30)   # 30帧 (1秒) 缓冲
        self.voice_buffer = deque(maxlen=30)   # 对应的语音缓冲
        self.fusion_model = self._build_fusion_transformer()
        
    def fuse(
        self, 
        face_features: MicroExpressionFeatures,
        voice_features: VoiceEmotionFeatures,
        timestamp: float
    ) -> MultimodalEmotionState:
        """
        融合表情与语音特征，返回综合情感状态
        """
        
        # 1. 特征编码
        face_encoding = self._encode_face(face_features)
        voice_encoding = self._encode_voice(voice_features)
        
        # 2. 时序对齐
        aligned_face, aligned_voice = self._temporal_align(
            face_encoding, voice_encoding, timestamp
        )
        
        # 3. Cross-Modal Attention 融合
        fused_embedding = self._cross_attention_fuse(aligned_face, aligned_voice)
        
        # 4. 情感推断
        emotion_state = self._infer_emotion(fused_embedding)
        
        # 5. 不一致性检测
        inconsistencies = self._detect_inconsistency(
            face_features, voice_features, emotion_state
        )
        
        # 6. 置信度计算
        confidence = self._compute_confidence(
            face_features, voice_features, inconsistencies
        )
        
        return MultimodalEmotionState(
            primary_emotion=emotion_state["primary"],
            secondary_emotion=emotion_state.get("secondary"),
            emotion_intensity=emotion_state["intensity"],
            valence=emotion_state["valence"],
            arousal=emotion_state["arousal"],
            dominance=emotion_state["dominance"],
            confidence=confidence,
            inconsistencies=inconsistencies,
            face_features=face_features,
            voice_features=voice_features,
            timestamp=timestamp
        )
    
    def _cross_attention_fuse(
        self, 
        face: Tensor, 
        voice: Tensor
    ) -> Tensor:
        """Cross-Modal Transformer 融合"""
        
        batch_size = face.shape[0]
        
        # Q, K, V 投影
        face_q = self.fq_face(face)
        face_k = self.fk_face(face)
        face_v = self.fv_face(face)
        
        voice_k = self.fk_voice(voice)
        voice_v = self.fv_voice(voice)
        
        # 表情查询语音键值
        face_to_voice_attn = self._attention(face_q, voice_k, voice_v)
        
        # 语音查询表情键值
        voice_q = self.fq_voice(voice)
        voice_to_face_attn = self._attention(voice_q, face_k, face_v)
        
        # 融合
        fused = torch.cat([
            face + face_to_voice_attn,
            voice + voice_to_face_attn
        ], dim=-1)
        
        # 最终投影
        output = self.fusion_projection(fused)
        
        return output
    
    def _detect_inconsistency(
        self,
        face: MicroExpressionFeatures,
        voice: VoiceEmotionFeatures,
        emotion_state: Dict
    ) -> List[Dict]:
        """检测多模态不一致性 - 重要的沉浸感增强手段"""
        
        inconsistencies = []
        
        # 1. 表情开心 vs 语音悲伤
        if face.happiness > 0.6 and voice.emotion_scores.get("sad", 0) > 0.5:
            inconsistencies.append({
                "type": "emotion_mismatch",
                "description": "表情开心但语音透露悲伤",
                "severity": "high",
                "interpretation": "可能在强颜欢笑或言不由衷"
            })
        
        # 2. 表情平静 vs 语音紧张
        if face.nervousness < 0.3 and voice.emotion_scores.get("nervous", 0) > 0.5:
            inconsistencies.append({
                "type": "emotion_mismatch",
                "description": "表情平静但声音紧张",
                "severity": "medium",
                "interpretation": "内心紧张但努力保持镇定"
            })
        
        # 3. 表情愤怒 vs 语音犹豫
        if face.anger > 0.5 and voice.emotion_scores.get("hesitant", 0) > 0.4:
            inconsistencies.append({
                "type": "emotion_mismatch",
                "description": "表情愤怒但声音犹豫",
                "severity": "medium",
                "interpretation": "色厉内荏，外强中干"
            })
        
        # 4. 文本与表情/语音严重不匹配
        # (需要在调用时传入文本情感对比)
        
        # 5. 情感突变检测 (可能的欺骗或重大情绪波动)
        if len(self.emotion_history) > 5:
            recent_emotions = [e["primary_emotion"] for e in self.emotion_history[-5:]]
            if len(set(recent_emotions)) > 3:  # 短时间内情绪大幅波动
                inconsistencies.append({
                    "type": "emotion_volatile",
                    "description": "情绪波动剧烈",
                    "severity": "low",
                    "interpretation": "可能内心有重大波动"
                })
        
        return inconsistencies
    
    def _compute_confidence(
        self,
        face: MicroExpressionFeatures,
        voice: VoiceEmotionFeatures,
        inconsistencies: List[Dict]
    ) -> float:
        """计算融合结果的置信度"""
        
        confidence = 0.5
        
        # 表情特征质量
        face_confidence = 1.0 - (face.brow_tension * 0.1)  # 模糊表情降低置信度
        
        # 语音特征质量
        voice_confidence = 1.0 - (voice.noise_ratio * 0.3)  # 噪声高降低置信度
        
        # 不一致性惩罚
        inconsistency_penalty = len(inconsistencies) * 0.1
        
        confidence = (face_confidence * 0.5 + voice_confidence * 0.5) - inconsistency_penalty
        
        return max(0.1, min(0.95, confidence))
```

---

## 四、深度情感驱动NPC行为引擎

### 4.1 情感协调Agent（新增核心模块）

```python
class EmotionCoordinatorAgent:
    """
    情感协调Agent - NPC行为系统的"大脑"
    负责：全局情感状态管理、多NPC协调、行为策略规划
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.global_emotion_state = None
        self.npc_emotion_models = {}  # 每个NPC的情感模型
        self.behavior_strategy = BehaviorStrategy()
        self.coordination_history = []
        
    def coordinate(
        self,
        user_multimodal_state: MultimodalEmotionState,
        dialogue_context: Dict,
        npc_states: Dict[str, Dict]
    ) -> CoordinationResult:
        """
        协调多个NPC的情感与行为
        """
        
        # 1. 分析用户情感态势
        user_sentiment = self._analyze_user_sentiment(user_multimodal_state)
        
        # 2. 更新全局情感状态
        self._update_global_state(user_sentiment, dialogue_context)
        
        # 3. 为每个NPC分配情感目标
        npc_emotion_targets = self._assign_emotion_targets(
            user_sentiment, npc_states, dialogue_context
        )
        
        # 4. 协调多NPC配合策略
        coordination_strategy = self._plan_coordination(
            user_sentiment, npc_states, npc_emotion_targets
        )
        
        # 5. 生成行为指令
        behavior_instructions = self._generate_behavior_instructions(
            coordination_strategy, dialogue_context
        )
        
        return CoordinationResult(
            user_sentiment=user_sentiment,
            npc_emotion_targets=npc_emotion_targets,
            coordination_strategy=coordination_strategy,
            behavior_instructions=behavior_instructions,
            global_intensity=self._compute_global_intensity(user_sentiment)
        )
    
    def _analyze_user_sentiment(self, state: MultimodalEmotionState) -> UserSentiment:
        """分析用户整体情感态势"""
        
        # 综合情感分析
        if state.primary_emotion == "nervous":
            if state.arousal > 0.7:
                sentiment_type = "highly_stressed"
                pressure_strategy = "continual_attack"
                sympathy_level = 0.2
            elif state.arousal > 0.4:
                sentiment_type = "moderately_stressed"
                pressure_strategy = "gradual_escalation"
                sympathy_level = 0.4
            else:
                sentiment_type = "calmly_nervous"
                pressure_strategy = "observation"
                sympathy_level = 0.6
                
        elif state.primary_emotion == "confident":
            if state.arousal > 0.6:
                sentiment_type = "aggressively_confident"
                pressure_strategy = "defensive_counter"
                sympathy_level = 0.3
            else:
                sentiment_type = "calmly_confident"
                pressure_strategy = "measured_approach"
                sympathy_level = 0.5
                
        elif state.primary_emotion == "happy":
            sentiment_type = "positive_mood"
            pressure_strategy = "maintain_momentum"
            sympathy_level = 0.7
            
        elif state.primary_emotion == "angry":
            sentiment_type = "hostile"
            pressure_strategy = "de_escalation"
            sympathy_level = 0.8
            
        elif state.primary_emotion == "sad":
            sentiment_type = "down"
            pressure_strategy = "supportive_approach"
            sympathy_level = 0.9
            
        else:  # neutral, confused, etc.
            sentiment_type = "neutral"
            pressure_strategy = "probing"
            sympathy_level = 0.5
        
        # 检测不一致性带来的额外信息
        hidden_sentiment = None
        if state.inconsistencies:
            for inc in state.inconsistencies:
                if inc.get("interpretation"):
                    hidden_sentiment = inc["interpretation"]
                    break
        
        return UserSentiment(
            type=sentiment_type,
            pressure_strategy=pressure_strategy,
            sympathy_level=sympathy_level,
            hidden_sentiment=hidden_sentiment,
            intensity=state.emotion_intensity,
            valence=state.valence,
            arousal=state.arousal,
            dominance=state.dominance
        )
    
    def _assign_emotion_targets(
        self,
        user_sentiment: UserSentiment,
        npc_states: Dict[str, Dict],
        dialogue_context: Dict
    ) -> Dict[str, EmotionTarget]:
        """为每个NPC分配情感目标 - 基于角色性格和当前态势"""
        
        targets = {}
        
        # 获取当前说话角色
        current_speaker = dialogue_context.get("current_speaker", "大舅")
        
        for npc_name, npc_state in npc_states.items():
            npc_personality = npc_state.get("personality", {})
            is_active = (npc_name == current_speaker)
            
            # 基于用户态势确定NPC目标情感
            if user_sentiment.type == "highly_stressed":
                if npc_personality.get("role") == "aggressor":
                    # 进攻型角色继续施压
                    target_emotion = "dominant"
                    target_intensity = 0.9
                elif npc_personality.get("role") == "supporter":
                    # 支持型角色适当缓和
                    target_emotion = "sympathetic"
                    target_intensity = 0.6
                else:
                    target_emotion = "observing"
                    target_intensity = 0.4
                    
            elif user_sentiment.type == "aggressively_confident":
                if is_active:
                    # 当前说话者要反击
                    target_emotion = "defensive"
                    target_intensity = 0.8
                else:
                    # 其他角色观察等待
                    target_emotion = "alert"
                    target_intensity = 0.5
                    
            elif user_sentiment.type == "hostile":
                # 用户愤怒时，所有NPC适当收敛
                target_emotion = "conciliatory"
                target_intensity = 0.7
                
            elif user_sentiment.type == "down":
                # 用户情绪低落，给台阶
                target_emotion = "encouraging"
                target_intensity = 0.6
                
            else:
                # 默认按角色设定行动
                target_emotion = npc_personality.get("default_emotion", "neutral")
                target_intensity = 0.5
            
            # 个性化调整
            target_emotion = self._personalize_target(
                target_emotion, npc_personality, user_sentiment
            )
            
            targets[npc_name] = EmotionTarget(
                target_emotion=target_emotion,
                intensity=target_intensity,
                is_active=is_active,
                speaking_priority=self._calculate_speaking_priority(
                    npc_name, user_sentiment, npc_states
                )
            )
        
        return targets
    
    def _plan_coordination(
        self,
        user_sentiment: UserSentiment,
        npc_states: Dict[str, Dict],
        emotion_targets: Dict[str, EmotionTarget]
    ) -> CoordinationStrategy:
        """规划多NPC协同策略"""
        
        # 1. 确定整体战术
        if user_sentiment.pressure_strategy == "continual_attack":
            # 持续施压战术
            strategy = {
                "tactic": "overwhelming_force",
                "description": "多NPC轮番施压，不给喘息机会",
                "roles": {
                    "aggressor": "主攻手，持续输出压力",
                    "supporter": "侧面补刀，填补逻辑漏洞",
                    "observer": "偶尔插话，制造尴尬"
                },
                "timing": "快速轮转，每个角色不超过1轮"
            }
            
        elif user_sentiment.pressure_strategy == "gradual_escalation":
            # 渐进施压战术
            strategy = {
                "tactic": "gradual_escalation",
                "description": "逐步提升压力，温水煮青蛙",
                "roles": {
                    "aggressor": "先礼后兵，逐步收紧",
                    "supporter": "表面劝和，实则拱火",
                    "observer": "适时加入，推波助澜"
                },
                "timing": "每个角色2-3轮，情感逐渐升级"
            }
            
        elif user_sentiment.pressure_strategy == "de_escalation":
            # 缓和战术 (用户愤怒时)
            strategy = {
                "tactic": "de_escalation",
                "description": "适当让步，避免冲突升级",
                "roles": {
                    "aggressor": "暂时收声，降低攻击性",
                    "supporter": "打圆场，给台阶",
                    "observer": "转移话题"
                },
                "timing": "放缓节奏，给用户冷静时间"
            }
            
        elif user_sentiment.pressure_strategy == "supportive_approach":
            # 支持性战术 (用户情绪低落)
            strategy = {
                "tactic": "supportive_approach",
                "description": "给鼓励和认可，提升用户信心",
                "roles": {
                    "aggressor": "减少施压",
                    "supporter": "正面鼓励，给正面反馈",
                    "observer": "适时夸奖"
                },
                "timing": "温和互动，逐步建立用户信心"
            }
            
        else:
            # 试探性战术
            strategy = {
                "tactic": "probing",
                "description": "试探用户底线，观察反应",
                "roles": {
                    "aggressor": "轻量级试探",
                    "supporter": "补充问题细节",
                    "observer": "记录反应"
                },
                "timing": "慢节奏，多观察"
            }
        
        return CoordinationStrategy(**strategy)
    
    def _generate_behavior_instructions(
        self,
        strategy: CoordinationStrategy,
        dialogue_context: Dict
    ) -> List[BehaviorInstruction]:
        """生成具体的行为指令"""
        
        instructions = []
        
        # 为每个角色生成指令
        for role_name, role_desc in strategy.roles.items():
            instruction = BehaviorInstruction(
                target_role=role_name,
                tactic=strategy.tactic,
                timing=strategy.timing,
                description=role_desc,
                emotion_adjustment=self._get_emotion_adjustment(role_name, strategy.tactic),
                verbal_style=self._get_verbal_style(role_name, strategy.tactic),
                nonverbal_cues=self._get_nonverbal_cues(role_name, strategy.tactic)
            )
            instructions.append(instruction)
        
        return instructions
    
    def _get_nonverbal_cues(self, role: str, tactic: str) -> NonVerbalCues:
        """获取非语言提示 - NPC的微表情和动作"""
        
        cue_map = {
            "overwhelming_force": {
                "aggressor": NonVerbalCues(
                    facial_expression="严肃",
                    eye_contact="直视",
                    body_posture="前倾",
                    hand_gesture="频繁指点",
                    voice_tone="严厉"
                ),
                "supporter": NonVerbalCues(
                    facial_expression="假笑",
                    eye_contact="斜视",
                    body_posture="放松旁观",
                    hand_gesture="抱臂",
                    voice_tone="阴阳怪气"
                )
            },
            "de_escalation": {
                "aggressor": NonVerbalCues(
                    facial_expression="尴尬赔笑",
                    eye_contact="回避",
                    body_posture="后靠放松",
                    hand_gesture="摆动",
                    voice_tone="缓和"
                ),
                "supporter": NonVerbalCues(
                    facial_expression="关切",
                    eye_contact="直视",
                    body_posture="前倾倾听",
                    hand_gesture="轻拍",
                    voice_tone="温和"
                )
            }
        }
        
        return cue_map.get(tactic, {}).get(role, NonVerbalCues())


@dataclass
class UserSentiment:
    """用户情感态势"""
    type: str                          # 情感类型
    pressure_strategy: str              # 施压策略
    sympathy_level: float               # 同理心水平 [0-1]
    hidden_sentiment: str = None        # 隐藏情感 (从不一致性推断)
    intensity: float = 0.5              # 情感强度
    valence: float = 0.0                # 效价
    arousal: float = 0.5                # 唤醒度
    dominance: float = 0.5              # 控制感


@dataclass
class EmotionTarget:
    """NPC情感目标"""
    target_emotion: str                 # 目标情感
    intensity: float                    # 情感强度
    is_active: bool                    # 是否当前说话
    speaking_priority: float            # 说话优先级


@dataclass
class CoordinationStrategy:
    """协调策略"""
    tactic: str                        # 战术名称
    description: str                    # 描述
    roles: Dict[str, str]               # 角色分配
    timing: str                         # 时序安排


@dataclass
class BehaviorInstruction:
    """行为指令"""
    target_role: str
    tactic: str
    timing: str
    description: str
    emotion_adjustment: Dict
    verbal_style: Dict
    nonverbal_cues: NonVerbalCues


@dataclass
class NonVerbalCues:
    """非语言提示"""
    facial_expression: str = "中性"
    eye_contact: str = "正常"
    body_posture: str = "自然"
    hand_gesture: str = "无"
    voice_tone: str = "正常"
```

### 4.2 角色Agent池升级

```python
class EmpatheticDialogueAgent(BaseAgent):
    """
    升级版对话Agent - 深度情感驱动
    每个NPC角色拥有独立的Agent，具备：
    1. 独特的性格模型
    2. 情感记忆
    3. 行为倾向
    """
    
    def __init__(self, llm=None, character_config: Dict = None):
        super().__init__(AgentRole.DIALOGUE, llm)
        
        self.character = character_config or {}
        self.personality = self._build_personality_model()
        self.emotion_memory = EmotionMemory()
        self.behavior_tendency = BehaviorTendency()
        
    def think(self, context: Dict) -> AgentMessage:
        """核心思考方法 - 情感驱动的对话生成"""
        
        # 1. 获取上下文信息
        user_input = context.get("user_input", "")
        user_emotion = context.get("user_multimodal_state")
        emotion_target = context.get("emotion_target", {})
        coordination = context.get("coordination_strategy", {})
        dialogue_history = context.get("dialogue_history", [])
        dominance = context.get("dominance", {"user": 50, "ai": 50})
        
        # 2. 分析用户当前状态
        user_analysis = self._analyze_user(user_input, user_emotion)
        
        # 3. 更新NPC情感状态
        npc_emotion = self._update_npc_emotion(user_analysis, emotion_target)
        
        # 4. 检索相关记忆
        relevant_memories = self.emotion_memory.retrieve(
            user_input, user_emotion, dialogue_history
        )
        
        # 5. 确定行为策略
        strategy = self._determine_strategy(
            user_analysis, npc_emotion, coordination, dominance
        )
        
        # 6. 构建情感感知Prompt
        prompt = self._build_empathy_prompt(
            user_analysis=user_analysis,
            npc_emotion=npc_emotion,
            memories=relevant_memories,
            strategy=strategy,
            dialogue_history=dialogue_history,
            dominance=dominance
        )
        
        # 7. 生成回复
        response = self._generate_response(prompt, strategy)
        
        # 8. 存储本次交互到记忆
        self.emotion_memory.store(
            user_input=user_input,
            user_emotion=user_emotion,
            npc_response=response,
            outcome=strategy.get("outcome")
        )
        
        # 9. 生成行为指令
        behavior_cues = self._generate_behavior_cues(
            npc_emotion, strategy, coordination
        )
        
        return AgentMessage(
            self.role,
            response,
            metadata={
                "speaker": self.character.get("name", "NPC"),
                "npc_emotion": npc_emotion,
                "strategy": strategy,
                "behavior_cues": behavior_cues,
                "memories_used": len(relevant_memories)
            },
            confidence=strategy.get("confidence", 0.8)
        )
    
    def _build_empathy_prompt(
        self,
        user_analysis: Dict,
        npc_emotion: Dict,
        memories: List[Dict],
        strategy: Dict,
        dialogue_history: List,
        dominance: Dict
    ) -> str:
        """构建情感感知的Prompt - 核心差异化手段"""
        
        char_name = self.character.get("name", "NPC")
        char_personality = self.personality
        scenario = self.character.get("scenario", "shandong_dinner")
        
        # 用户情感分析摘要
        user_emotion_summary = f"""
【用户当前状态分析】
- 主要情感: {user_analysis.get('primary_emotion', 'neutral')}
- 情感强度: {user_analysis.get('intensity', 0.5):.0%}
- 自信度: {user_analysis.get('confidence', 0.5):.0%}
- 紧张度: {user_analysis.get('nervousness', 0.5):.0%}
- 隐藏情感: {user_analysis.get('hidden_sentiment', '无')}
- 语音特征: {user_analysis.get('voice_summary', '正常')}
"""
        
        # NPC自身状态
        npc_emotion_summary = f"""
【你的当前状态】
- 情感状态: {npc_emotion.get('current_emotion', 'neutral')}
- 情绪强度: {npc_emotion.get('intensity', 0.5):.0%}
- 对用户的态度: {npc_emotion.get('attitude', '中性')}
"""
        
        # 记忆影响
        memory_section = ""
        if memories:
            memory_section = f"""
【相关记忆】
用户之前的表现：
{chr(10).join([f"- {m['summary']}" for m in memories[:3]])}
你需要根据这些记忆，保持对话的连贯性。
"""
        
        # 策略指导
        strategy_section = f"""
【当前策略】
- 战术: {strategy.get('tactic', 'normal')}
- 语气: {strategy.get('tone', 'normal')}
- 目标: {strategy.get('goal', '正常对话')}
- 情感基调: {strategy.get('emotional_tone', '中性')}
"""
        
        # 行为指导
        behavior_section = f"""
【行为指导】
- 表情: {strategy.get('facial_expression', '自然')}
- 眼神: {strategy.get('eye_contact', '正常')}
- 语气: {strategy.get('voice_tone', '自然')}
- 动作: {strategy.get('gesture', '无')}
"""
        
        prompt = f"""<角色设定>
你是{char_name}，{char_personality['bio']}
说话风格: {char_personality['style']}
常用策略: {char_personality['strategy']}
说话特点: {char_personality.get('verbal_traits', '')}
</角色设定>

<当前局势>
- 用户气场: {dominance['user']}/100
- 你的气场: {dominance['ai']}/100
- 你是{strategy.get('speaker', char_name)}
</局势>

{user_emotion_summary}
{npc_emotion_summary}
{memory_section}
{strategy_section}
{behavior_section}

【最近对话】
{self._format_history(dialogue_history[-6:])}

【用户刚才说】
"{user_input}"

【输出要求】
1. 严格保持{char_name}的说话风格和性格特点
2. 根据用户的真实情感状态（不是TA说的话，而是TA的表现）来调整你的回应
3. 如果检测到用户紧张，可以适当放松或给台阶；如果用户自信，要更加谨慎应对
4. 深度结合情感分析结果，让回复有差异化 - 同样的内容，不同的语气
5. 包含适当的表情动作描述（用括号）
6. 30-80字以内，只输出对话内容
7. 如果需要多个角色说话，只输出你负责的那部分

请以{char_name}的身份回复："""
        
        return prompt
    
    def _generate_behavior_cues(
        self,
        npc_emotion: Dict,
        strategy: Dict,
        coordination: Dict
    ) -> BehaviorCues:
        """生成行为提示 - 用于前端渲染NPC表情和动作"""
        
        return BehaviorCues(
            facial_expression=strategy.get("facial_expression", "neutral"),
            eye_contact=strategy.get("eye_contact", "normal"),
            body_language=strategy.get("body_language", "natural"),
            hand_gesture=strategy.get("hand_gesture", "none"),
            voice_modulation=VoiceModulation(
                speed=strategy.get("voice_speed", 1.0),
                pitch=strategy.get("voice_pitch", 0),
                volume=strategy.get("voice_volume", 1.0),
                tone=strategy.get("voice_tone", "natural")
            ),
            timing=TimingCues(
                pause_before=float(strategy.get("pause_before", 0)),
                pause_after=float(strategy.get("pause_after", 0)),
                hesitation=int(strategy.get("hesitation", 0))
            )
        )
```

### 4.3 评估Agent升级

```python
class EmpatheticEvaluatorAgent(BaseAgent):
    """
    升级版评估Agent - 情感驱动的多模态评分
    """
    
    EVAL_CRITERIA = {
        "emotional_intelligence": {"weight": 0.25, "description": "情商表现"},
        "response_quality": {"weight": 0.20, "description": "回复质量"},
        "pressure_handling": {"weight": 0.20, "description": "压力应对"},
        "cultural_fit": {"weight": 0.15, "description": "文化适配"},
        "authenticity": {"weight": 0.20, "description": "真诚度 (多模态)"},
    }
    
    def think(self, context: Dict) -> AgentMessage:
        """综合多模态信息的评估"""
        
        user_input = context.get("user_input", "")
        user_multimodal = context.get("user_multimodal_state")
        ai_response = context.get("ai_response", "")
        prev_dominance = context.get("dominance", {"user": 50, "ai": 50})
        
        # 多维度评分
        scores = {}
        
        # 1. 情商评分
        scores["emotional_intelligence"] = self._eval_emotional_intelligence(
            user_input, user_multimodal
        )
        
        # 2. 回复质量
        scores["response_quality"] = self._eval_response_quality(
            user_input, ai_response
        )
        
        # 3. 压力应对
        scores["pressure_handling"] = self._eval_pressure_handling(
            user_input, user_multimodal, prev_dominance
        )
        
        # 4. 文化适配
        scores["cultural_fit"] = self._eval_cultural_fit(
            user_input, context.get("scenario")
        )
        
        # 5. 真诚度评分 (新增 - 基于多模态)
        scores["authenticity"] = self._eval_authenticity(
            user_input, user_multimodal
        )
        
        # 综合评分
        total = sum(scores[k] * v["weight"] for k, v in self.EVAL_CRITERIA.items())
        
        # 气场变化计算
        delta = self._calculate_dominance_shift(scores, prev_dominance)
        new_user = max(10, min(90, prev_dominance["user"] + delta))
        new_ai = 100 - new_user
        
        # 生成点评
        judgment = self._generate_judgment(scores, delta, user_multimodal)
        
        return AgentMessage(
            self.role,
            judgment,
            metadata={
                "scores": scores,
                "total": total,
                "new_dominance": {"user": new_user, "ai": new_ai},
                "delta": delta,
                "authenticity_analysis": self._get_authenticity_analysis(user_multimodal)
            },
            confidence=0.9
        )
    
    def _eval_authenticity(self, text: str, multimodal: MultimodalEmotionState) -> float:
        """
        评估用户的真诚度 - 核心创新点
        检测用户表达与真实情感的一致性
        """
        
        if not multimodal:
            return 0.5
        
        base_score = 0.5
        
        # 1. 文本-表情一致性
        if multimodal.inconsistencies:
            for inc in multimodal.inconsistencies:
                if inc["type"] == "emotion_mismatch":
                    # 严重不一致扣分
                    if inc.get("severity") == "high":
                        base_score -= 0.2
                    elif inc.get("severity") == "medium":
                        base_score -= 0.1
        
        # 2. 文本-语音一致性
        text_emotion = self._infer_text_emotion(text)
        voice_emotion = multimodal.primary_emotion
        
        if text_emotion != voice_emotion:
            # 文本和语音情感不匹配
            if text_emotion in ["happy", "confident"] and voice_emotion in ["sad", "nervous"]:
                # 可能的言不由衷
                base_score -= 0.15
        
        # 3. 情感稳定性
        if multimodal.arousal > 0.8:
            # 情绪激动时更容易不真诚
            base_score -= 0.1
        elif multimodal.arousal < 0.3:
            # 过于平静可能是在伪装
            base_score -= 0.05
        
        # 4. 隐藏情感检测
        if multimodal.primary_emotion == "confident" and multimodal.dominance < 0.4:
            # 外强中干
            base_score -= 0.15
        
        return max(0.1, min(1.0, base_score))
    
    def _get_authenticity_analysis(self, multimodal: MultimodalEmotionState) -> Dict:
        """生成真诚度分析报告"""
        
        if not multimodal:
            return {"status": "no_data"}
        
        analysis = {
            "status": "analyzed",
            "inconsistencies": len(multimodal.inconsistencies),
            "hidden_sentiment": multimodal.primary_emotion,
            "true_feeling": self._interpret_hidden_sentiment(multimodal),
            "sincere":(multimodal len.inconsistencies) < 2
        }
        
        return analysis
    
    def _interpret_hidden_sentiment(self, state: MultimodalEmotionState) -> str:
        """解读用户真实情感"""
        
        if not state.inconsistencies:
            return "表达真诚，情感一致"
        
        interpretations = []
        for inc in state.inconsistencies:
            if "interpretation" in inc:
                interpretations.append(inc["interpretation"])
        
        return "; ".join(interpretations) if interpretations else "情感复杂"
```

---

## 五、NPC行为控制实现

### 5.1 行为指令系统

```python
class NPCBehaviorController:
    """
    NPC行为控制器
    将高层决策转换为可执行的行为指令
    """
    
    def __init__(self):
        self.behavior_templates = self._load_behavior_templates()
        
    def generate_behavior_package(
        self,
        agent_message: AgentMessage,
        emotion_state: MultimodalEmotionState,
        strategy: Dict
    ) -> BehaviorPackage:
        """生成完整的行为包 - 前端渲染用"""
        
        speaker = agent_message.metadata.get("speaker", "NPC")
        behavior_cues = agent_message.metadata.get("behavior_cues", {})
        
        # 1. 文本回复
        text_response = agent_message.content
        
        # 2. 表情动画指令
        facial_instruction = self._generate_facial_instruction(
            behavior_cues.get("facial_expression", "neutral"),
            emotion_state
        )
        
        # 3. 语音合成参数
        voice_params = self._generate_voice_params(
            behavior_cues.get("voice_modulation", {}),
            emotion_state
        )
        
        # 4. 动作动画指令
        motion_instruction = self._generate_motion_instruction(
            behavior_cues.get("body_language", "natural"),
            behavior_cues.get("hand_gesture", "none")
        )
        
        # 5. 情感状态显示
        emotion_display = {
            "shown_emotion": strategy.get("shown_emotion", emotion_state.primary_emotion),
            "actual_emotion": emotion_state.primary_emotion,
            "intensity": emotion_state.emotion_intensity,
            "thought_bubble": self._generate_thought_bubble(emotion_state, strategy)
        }
        
        return BehaviorPackage(
            text=text_response,
            facial=facial_instruction,
            voice=voice_params,
            motion=motion_instruction,
            display=emotion_display,
            speaker=speaker
        )
    
    def _generate_facial_instruction(
        self,
        base_expression: str,
        emotion_state: MultimodalEmotionState
    ) -> FacialInstruction:
        """生成面部表情动画指令"""
        
        expression_map = {
            "neutral": {
                "eyebrows": "normal",
                "eyes": "normal",
                "mouth": "slight_smile",
                "intensity": 0.3
            },
            "happy": {
                "eyebrows": "raised",
                "eyes": "squinting",
                "mouth": "big_smile",
                "intensity": 0.8
            },
            "angry": {
                "eyebrows": "furrowed",
                "eyes": "narrowed",
                "mouth": "tight",
                "intensity": 0.9
            },
            "contemptuous": {
                "eyebrows": "one_raised",
                "eyes": "side_look",
                "mouth": "smirk",
                "intensity": 0.6
            },
            "sympathetic": {
                "eyebrows": "inner_raised",
                "eyes": "soft",
                "mouth": "gentle",
                "intensity": 0.5
            },
            "skeptical": {
                "eyebrows": "one_raised",
                "eyes": "narrowed_one",
                "mouth": "side",
                "intensity": 0.5
            }
        }
        
        # 基于用户情感调整
        if emotion_state.primary_emotion == "nervous" and base_expression == "neutral":
            # 用户紧张时，NPC可以表现出轻蔑或观察
            expression_map["neutral"]["mouth"] = "slight_smirk"
        
        base = expression_map.get(base_expression, expression_map["neutral"])
        
        return FacialInstruction(
            primary=base_expression,
            eyebrows=base["eyebrows"],
            eyes=base["eyes"],
            mouth=base["mouth"],
            intensity=min(1.0, base["intensity"] * emotion_state.emotion_intensity),
            micro_expressions=self._generate_micro_expressions(emotion_state)
        )
    
    def _generate_voice_params(
        self,
        voice_modulation: Dict,
        emotion_state: MultimodalEmotionState
    ) -> VoiceParams:
        """生成语音合成参数"""
        
        # 基础参数
        base_speed = voice_modulation.get("speed", 1.0)
        base_pitch = voice_modulation.get("pitch", 0)
        base_volume = voice_modulation.get("volume", 1.0)
        
        # 情感调整
        if emotion_state.primary_emotion == "angry":
            base_speed = min(1.3, base_speed * 1.1)
            base_pitch += 2
            base_volume = min(1.2, base_volume * 1.1)
            
        elif emotion_state.primary_emotion == "nervous":
            # 用户紧张时，NPC可能更温和
            base_speed = max(0.8, base_speed * 0.9)
            base_pitch -= 1
            
        elif emotion_state.primary_emotion == "confident":
            base_speed = min(1.1, base_speed * 1.05)
            base_pitch += 1
            
        return VoiceParams(
            speed=base_speed,
            pitch=base_pitch,
            volume=base_volume,
            tone=voice_modulation.get("tone", "natural"),
            pause_before=voice_modulation.get("pause_before", 0),
            pause_after=voice_modulation.get("pause_after", 0)
        )
    
    def _generate_thought_bubble(
        self,
        emotion_state: MultimodalEmotionState,
        strategy: Dict
    ) -> str:
        """生成NPC心理活动气泡 - 增强沉浸感"""
        
        # 基于用户情感生成NPC内心OS
        if emotion_state.primary_emotion == "nervous":
            thoughts = [
                "这娃子明显紧张了，再加把劲",
                "哈哈，撑不住了吧",
                "看来今天是来对了"
            ]
        elif emotion_state.primary_emotion == "confident":
            thoughts = [
                "口气不小嘛",
                "有意思，让我试试你的深浅",
                "这年轻人有点东西"
            ]
        elif emotion_state.primary_emotion == "angry":
            thoughts = [
                "呦，急了",
                "看来踩到痛点了",
                "这就受不了了？"
            ]
        else:
            thoughts = [
                "在想着什么呢",
                "下一句会说什么",
                "嗯..."
            ]
        
        return random.choice(thoughts)


@dataclass
class BehaviorPackage:
    """完整的行为包"""
    text: str
    facial: FacialInstruction
    voice: VoiceParams
    motion: MotionInstruction
    display: Dict
    speaker: str


@dataclass
class FacialInstruction:
    """面部表情指令"""
    primary: str
    eyebrows: str
    eyes: str
    mouth: str
    intensity: float
    micro_expressions: List[Dict]
```

---

## 六、前端交互与可视化

### 6.1 实时情感可视化

```javascript
// 前端 - 实时情感状态面板
class EmotionVisualization {
    constructor() {
        this.currentEmotion = null;
        this.emotionHistory = [];
        this.maxHistoryLength = 30;
    }
    
    // 更新情感显示
    updateEmotion(state) {
        this.currentEmotion = state;
        this.emotionHistory.push({
            timestamp: Date.now(),
            ...state
        });
        
        if (this.emotionHistory.length > this.maxHistoryLength) {
            this.emotionHistory.shift();
        }
        
        this.render();
    }
    
    // 渲染情感状态
    render() {
        if (!this.currentEmotion) return;
        
        // 1. 更新主要情感指标
        this.updateMetricBars();
        
        // 2. 更新情感时序图
        this.renderEmotionTimeline();
        
        // 3. 更新NPC反应面板
        this.renderNPCReactions();
        
        // 4. 更新不一致性警告
        this.renderInconsistencies();
    }
    
    // 更新NPC对用户的理解
    renderNPCReactions() {
        const reactions = this.currentEmotion.npc_understanding || {};
        
        // 显示NPC看到了什么
        Object.entries(reactions).forEach(([npc, understanding]) => {
            const el = document.getElementById(`npc-understanding-${npc}`);
            if (el) {
                el.innerHTML = `
                    <div class="npc-thought">
                        <span class="npc-name">${npc}:</span>
                        <span class="npc-thought-text">${understanding.detected}</span>
                        <span class="npc-strategy">策略: ${understanding.strategy}</span>
                    </div>
                `;
            }
        });
    }
    
    // 渲染不一致性警告
    renderInconsistencies() {
        const inconsistencies = this.currentEmotion.inconsistencies || [];
        
        const container = document.getElementById('inconsistency-panel');
        if (!container) return;
        
        if (inconsistencies.length === 0) {
            container.innerHTML = '';
            return;
        }
        
        container.innerHTML = inconsistencies.map(inc => `
            <div class="inconsistency-alert ${inc.severity}">
                <span class="icon">⚠️</span>
                <span class="text">${inc.description}</span>
                <span class="interpretation">(${inc.interpretation})</span>
            </div>
        `).join('');
    }
}
```

### 6.2 NPC情感状态显示

```javascript
// NPC情感状态面板
class NPCEmotionPanel {
    constructor() {
        this.npcs = {};
    }
    
    // 更新NPC情感状态
    updateNPC(npcName, emotionState, strategy) {
        this.npcs[npcName] = {
            currentEmotion: emotionState,
            strategy: strategy,
            thoughtBubble: strategy.thought_bubble,
            lastUpdate: Date.now()
        };
        
        this.renderNPC(npcName);
    }
    
    // 渲染单个NPC
    renderNPC(npcName) {
        const npc = this.npcs[npcName];
        if (!npc) return;
        
        const el = document.getElementById(`npc-card-${npcName}`);
        if (!el) return;
        
        // 1. 表情动画
        const faceEl = el.querySelector('.npc-face');
        this.applyFacialAnimation(faceEl, npc.currentEmotion.facial);
        
        // 2. 心理气泡
        const bubbleEl = el.querySelector('.thought-bubble');
        if (bubbleEl && npc.thoughtBubble) {
            bubbleEl.textContent = npc.thoughtBubble;
            bubbleEl.classList.add('visible');
        }
        
        // 3. 状态标签
        const statusEl = el.querySelector('.npc-status');
        statusEl.textContent = `策略: ${npc.strategy.tactic}`;
    }
}
```

---

## 七、数据流与接口设计

### 7.1 核心数据流

```
用户输入流程：

┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  前端摄像头  │───►│  表情分析    │───►│  情感状态机     │    │                  │
│  (30fps)    │    │  (MediaPipe) │    │  (平滑+推断)    │    │                  │
└─────────────┘    └──────────────┘    └─────────────────┘    │                  │
                                                                 │                  │
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │  融合引擎        │
│  前端麦克风  │───►│  语音情感    │───►│  语音特征提取   │───►│  (Cross-Modal)  │
│  (实时)     │    │  (情感分类)  │    │  (librosa)      │    │                  │
└─────────────┘    └──────────────┘    └─────────────────┘    │                  │
                                                                 │                  │
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │                  │
│  用户文本    │───►│  文本情感    │───►│  语义理解       │───►│                  │
│  (输入框)   │    │  (关键词)    │    │  (LLM)         │    │                  │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              融合后的情感状态                                    │
│  { primary_emotion, intensity, inconsistencies, confidence, ... }              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            情感协调 Agent                                        │
│  - 分析用户态势                                                                 │
│  - 规划NPC策略                                                                  │
│  - 协调多NPC配合                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
        ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
        │  角色Agent   │    │  评估Agent   │    │  记忆Agent   │
        │  (生成回复)  │    │  (评分)      │    │  (存储)      │
        └──────────────┘    └──────────────┘    └──────────────┘
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   行为指令生成器              │
                    │  - 文本回复                  │
                    │  - NPC表情动画               │
                    │  - 语音参数                  │
                    │  - 心理气泡                  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   前端渲染                    │
                    │  - 对话显示                   │
                    │  - NPC表情变化               │
                    │  - 情感面板更新              │
                    │  - 气场值更新                │
                    └───────────────────────────────┘
```

### 7.2 API 接口扩展

| 接口 | 方法 | 说明 | 新增参数 |
|-----|------|------|---------|
| `/api/chat/send` | POST | 发送消息 | `multimodal_state`: 融合后的情感状态 |
| `/api/multimodal/stream` | WebSocket | 实时多模态流 | 持续推送情感状态更新 |
| `/api/npc/understand` | GET | 获取NPC理解 | 返回NPC对用户当前状态的理解 |
| `/api/session/emotion_history` | GET | 情感历史 | 返回会话的情感变化曲线 |

---

## 八、技术指标与性能优化

### 8.1 性能目标

| 指标 | 目标值 | 说明 |
|-----|-------|------|
| 表情分析延迟 | < 50ms | 端到端延迟 |
| 语音情感延迟 | < 200ms | 片段级延迟 |
| 融合延迟 | < 30ms | 特征融合 |
| NPC响应延迟 | < 1.5s | LLM生成 + 行为生成 |
| 帧率 | 30 FPS | 表情动画流畅度 |
| 内存占用 | < 2GB | 整体系统 |

### 8.2 优化策略

**1. 表情分析优化**

- 使用 MediaPipe 轻量级模型 (face_mesh)
- 降采样处理：每3帧处理1帧
- GPU加速：使用 TensorFlow Lite GPU delegate

**2. 语音分析优化**

- 滑动窗口：200ms窗口，50ms步长
- 特征缓存：复用MFCC等计算结果
- 轻量模型：使用小型CNN进行情感分类

**3. 融合推理优化**

- 特征量化：INT8量化减少计算
- 批处理：批量处理历史帧
- 缓存策略：相似情感状态复用结果

**4. LLM响应优化**

- Prompt模板化：减少token消耗
- 流式输出：减少感知延迟
- 本地缓存：常用回复缓存

---

## 九、部署配置

### 9.1 依赖清单

```txt
# 核心依赖
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0

# 多模态处理
mediapipe>=0.10.0        # 人脸关键点
librosa>=0.10.0          # 音频特征提取
soundfile>=0.12.0        # 音频IO
scipy>=1.10.0            # 信号处理

# LLM & TTS
modelscope>=1.9.0        # 模型加载
edge-tts>=6.1.0          # 语音合成

# 存储
sqlalchemy>=2.0.0        # 数据库
redis>=4.5.0             # 缓存

# 服务
fastapi>=0.100.0
uvicorn>=0.22.0
gradio>=3.35.0
```

### 9.2 配置示例

```yaml
# config/multimodal.yaml
multimodal:
  face:
    enabled: true
    model: "face_mesh_light"
    process_every_n_frames: 3
    
  voice:
    enabled: true
    sample_rate: 16000
    window_size: 200
    hop_size: 50
    
  fusion:
    enabled: true
    cross_attention: true
    cache_enabled: true
    
npc:
  coordination:
    enabled: true
    strategy_update_interval: 1
    
  behavior:
    show_thought_bubble: true
    facial_animation: true
    voice_modulation: true
    
performance:
  target_fps: 30
  max_latency_ms: 2000
  cache_size_mb: 512
```

---

## 十、总结与展望

### 10.1 技术升级要点总结

本次深度升级在以下方面实现了技术突破：

| 升级维度 | 原有能力 | 升级后能力 | 技术价值 |
|---------|---------|-----------|---------|
| 表情分析 | 基础情绪识别 | 微表情50维特征提取 + 时序状态机 | 深度理解用户真实情感 |
| 语音分析 | 音量检测 | 60维声学特征 + 情感推断 | 识别言外之意 |
| 情感融合 | 简单加权 | Cross-Modal Transformer + 不一致性检测 | 发现用户伪装 |
| NPC协调 | 顺序轮转 | 情感协调Agent + 策略规划 | 智能团队配合 |
| 角色Agent | 规则Prompt | 情感记忆 + 个性化策略 | 有温度的AI角色 |
| 评估系统 | 文本评分 | 多模态综合评估 + 真诚度检测 | 全面客观评价 |
| 沉浸体验 | 文本对话 | 多维度行为反馈 + 心理气泡 | 身临其境 |

### 10.2 未来扩展方向

**短期扩展**

- 支持更多情感维度（疲劳、困惑、尴尬等）
- 增加生理信号输入（心率、皮肤电导）
- 优化NPC微表情动画

**中期目标**

- 引入强化学习优化NPC策略
- 支持个性化NPC性格养成
- 实现跨会话用户情感建模

**长期愿景**

- 构建通用情感智能框架
- 支持VR/AR沉浸式场景
- 实现真正的共情AI助手

---

**文档版本**: v1.0  
**更新日期**: 2026-02-21  
**维护团队**: TalkArena Dev Team
