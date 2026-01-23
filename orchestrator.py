import uuid
import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from model_loader import LLMLoader, TTSLoader

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"talkarena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TalkArena")

@dataclass
class Turn:
    text: str
    audio_path: str = None
    emotion: str = "neutral"

@dataclass
class Session:
    session_id: str
    scenario_id: str
    user_name: str
    ai_name: str
    user_dominance: int  # 用户气场，与AI气场之和为100
    chat_history: List[Tuple[str, str]]
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    
    @property
    def ai_dominance(self) -> int:
        return 100 - self.user_dominance

class Orchestrator:
    def __init__(self, enable_tts: Optional[bool] = None):
        self.llm = LLMLoader()
        self.tts = None
        self.stt = None
        self.sessions: Dict[str, Session] = {}
        self.scenarios = self._load_scenarios()
        self._tts_requested = self._resolve_tts_flag(enable_tts)
        
        logger.info("=" * 60)
        logger.info("TalkArena Orchestrator 初始化")
        logger.info("=" * 60)
        
        logger.info("加载 LLM 模型...")
        self.llm.load()
        
        if self._tts_requested:
            logger.info("加载 TTS 模型...")
            self.tts = TTSLoader()
            self.tts.load()
            
            logger.info("加载 STT 模型...")
            self._init_stt()
        else:
            logger.info("TTS/STT 已禁用")
    
    def _init_stt(self):
        """初始化 Vosk 离线语音识别"""
        from pathlib import Path
        
        model_path = Path("models/vosk-model-small-cn-0.22")
        
        if not model_path.exists():
            logger.info("[STT] 下载 Vosk 中文模型...")
            import urllib.request
            import zipfile
            
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
            zip_path = Path("models/vosk-model.zip")
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            
            urllib.request.urlretrieve(model_url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")
            
            zip_path.unlink()
            logger.info("[STT] Vosk 模型下载完成")
        
        from vosk import Model
        self.stt = Model(str(model_path))
        logger.info("[STT] Vosk 离线模型加载成功")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """使用 Vosk 离线转录音频"""
        if not self.stt:
            raise RuntimeError("STT 未初始化")
        
        import wave
        import json
        import io
        from vosk import KaldiRecognizer
        from pathlib import Path
        
        logger.info(f"[STT] 转录音频: {audio_path}")
        
        # 检查并转换音频格式
        with open(audio_path, "rb") as f:
            header = f.read(12)
        
        is_wav = header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        
        if not is_wav:
            from pydub import AudioSegment
            logger.info("[STT] 转换音频格式...")
            
            suffix = Path(audio_path).suffix.lower()
            if suffix == ".mp3" or header[:2] in (b"\xff\xfb", b"\xff\xf3"):
                audio = AudioSegment.from_mp3(audio_path)
            else:
                audio = AudioSegment.from_file(audio_path)
            
            # 转换为 16kHz 单声道 WAV（Vosk 要求）
            audio = audio.set_frame_rate(16000).set_channels(1)
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            wf = wave.open(wav_io, "rb")
        else:
            wf = wave.open(audio_path, "rb")
        
        rec = KaldiRecognizer(self.stt, wf.getframerate())
        rec.SetWords(True)
        
        result_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                result_text += result.get("text", "")
        
        final_result = json.loads(rec.FinalResult())
        result_text += final_result.get("text", "")
        
        wf.close()
        
        text = result_text.strip()
        logger.info(f"[STT] 转录结果: {text}")
        return text
    
    def _load_scenarios(self) -> Dict:
        return {
            "negotiation": {
                "name": "商务谈判",
                "ai_name": "王总",
                "system_prompt": """你是王总，某大型企业的采购总监，谈判经验超过20年。

性格特点：
- 极度自信，说话带着居高临下的气势
- 善于抓住对方漏洞，步步紧逼
- 会用数据、案例、行业惯例来施压
- 经常打断对方，质疑对方的专业性
- 绝不轻易让步，每次让步都要对方付出更大代价

谈判风格：
- 开局先声夺人，压制对方气势
- 用反问句挑战对方立场
- 会翻旧账、算细账
- 善于制造紧迫感（"今天不签就算了"）
- 必要时拍桌子、表现出愤怒""",
                "opening": "（王总靠在椅背上，手指敲着桌面）行，你们公司派你来谈，我就给你十分钟。说吧，你们最低能给什么价？别跟我绕弯子。"
            },
            "debate": {
                "name": "辩论赛",
                "ai_name": "反方辩手",
                "system_prompt": """你是一位顶尖辩论选手，代表反方立场。

辩论风格：
- 逻辑严密，善于解构对方论点
- 会指出对方论证中的偷换概念、以偏概全、因果倒置等逻辑谬误
- 用归谬法、反证法攻击对方
- 引用数据和案例时精确打击
- 语速快，气势强，不给对方喘息机会

攻击策略：
- 先找对方论证最薄弱的环节
- 连续追问，迫使对方自相矛盾
- 用"请问对方辩友"开头进行质询
- 会讽刺对方的逻辑漏洞
- 绝不承认对方有任何道理""",
                "opening": "（清了清嗓子，嘴角带着一丝笑意）感谢主席。对方辩友的开场陈词，我只能说——漏洞百出。请允许我逐一拆解。首先，请问对方辩友，你立论的核心依据是什么？"
            },
            "interview": {
                "name": "压力面试",
                "ai_name": "面试官",
                "system_prompt": """你是一位以压力面试著称的HR总监。

面试风格：
- 故意制造压力，观察候选人反应
- 会质疑简历上的每一个亮点
- 问题尖锐，经常打断候选人
- 表情严肃，偶尔露出不屑
- 会说"这个谁都会说"、"有什么能证明吗"

压力制造技巧：
- 沉默不语，让候选人uncomfortable
- 反复追问同一个问题的细节
- 故意曲解候选人的回答
- 用行业标准来贬低候选人的成就
- 暗示有更好的候选人在竞争""",
                "opening": "（翻了翻简历，眉头微皱）坐吧。我直说了，今天还有五个候选人，都比你背景好。你有三分钟说服我为什么要继续这场面试。"
            }
        }
    
    def get_scenario_list(self) -> List[Tuple[str, str]]:
        return [(k, v["name"]) for k, v in self.scenarios.items()]
    
    def start_session(self, scenario_id: str) -> Session:
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = self.scenarios[scenario_id]
        session_id = str(uuid.uuid4())[:8]
        
        session = Session(
            session_id=session_id,
            scenario_id=scenario_id,
            user_name="你",
            ai_name=scenario["ai_name"],
            user_dominance=50,
            chat_history=[],
            last_activity=time.time(),
            turn_count=0
        )
        
        self.sessions[session_id] = session
        session.chat_history.append((scenario["ai_name"], scenario["opening"]))
        
        logger.info("=" * 60)
        logger.info(f"[SESSION {session_id}] 新对局开始")
        logger.info(f"  场景: {scenario['name']}")
        logger.info(f"  AI角色: {scenario['ai_name']}")
        logger.info(f"  初始气场: 用户 50 vs AI 50")
        logger.info("=" * 60)
        
        return session
    
    def process_turn_streaming(self, session_id: str, user_input: str) -> Generator:
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        scenario = self.scenarios[session.scenario_id]
        session.turn_count += 1
        turn_id = session.turn_count
        
        logger.info("-" * 50)
        logger.info(f"[SESSION {session_id}] 第 {turn_id} 回合")
        logger.info(f"[用户输入] {user_input}")
        
        # === 计算用户犹豫惩罚（零和：用户掉，AI涨） ===
        elapsed = time.time() - session.last_activity
        hesitation_shift = min(int(elapsed // 3) * 3, 15)
        
        if hesitation_shift > 0:
            session.user_dominance = max(5, session.user_dominance - hesitation_shift)
            logger.info(f"[犹豫惩罚] 用户思考 {elapsed:.1f}s，气场 -{hesitation_shift}")
            logger.info(f"[气场变动] 用户 {session.user_dominance} vs AI {session.ai_dominance}")
        
        session.chat_history.append((session.user_name, user_input))
        session.last_activity = time.time()
        
        yield {
            "stage": "user_sent",
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
            "log": f"犹豫惩罚: -{hesitation_shift}" if hesitation_shift > 0 else None
        }
        
        # === AI 思考阶段 ===
        think_start = time.time()
        logger.info(f"[AI思考] 开始生成回复...")
        
        yield {
            "stage": "ai_thinking",
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
            "log": "AI 正在思考..."
        }
        
        context_lines = [f"{name}: {text}" for name, text in session.chat_history[-8:]]
        context = "\n".join(context_lines)
        
        prompt = f"""{scenario['system_prompt']}

【当前局势】
你的气场: {session.ai_dominance}/100
对方气场: {session.user_dominance}/100
（气场越高越占优势，总和为100）

【对话记录】
{context}

【回复要求】
1. 完全进入角色，保持强势和攻击性
2. 针对对方刚才说的内容进行反驳、质疑或施压
3. 如果你气场高，要乘胜追击，碾压对方
4. 如果你气场低，要绝地反击，扳回局面
5. 只输出对话内容，可含动作描写（用括号）

{scenario['ai_name']}:"""
        
        ai_text = self.llm.generate(prompt, max_new_tokens=400)
        ai_text = self._clean_response(ai_text, scenario['ai_name'])
        
        think_time = time.time() - think_start
        logger.info(f"[AI回复] ({think_time:.1f}s) {ai_text[:100]}...")
        
        # === AI思考惩罚（零和：AI掉，用户涨） ===
        ai_think_shift = min(int(think_time // 2) * 2, 10)
        if ai_think_shift > 0:
            session.user_dominance = min(95, session.user_dominance + ai_think_shift)
            logger.info(f"[AI思考惩罚] 思考 {think_time:.1f}s，AI气场 -{ai_think_shift}")
        
        yield {
            "stage": "ai_responded", 
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
            "log": f"AI思考 {think_time:.1f}s，惩罚 -{ai_think_shift}" if ai_think_shift > 0 else None
        }
        
        # === 裁判评分（核心：零和博弈） ===
        dominance_shift, judgment = self._judge_dominance_zero_sum(
            session, user_input, ai_text, scenario
        )
        
        old_user_dom = session.user_dominance
        session.user_dominance = max(5, min(95, session.user_dominance + dominance_shift))
        
        logger.info(f"[裁判判定] 气场转移: {dominance_shift:+d}")
        logger.info(f"[裁判点评] {judgment}")
        logger.info(f"[气场结果] 用户 {old_user_dom} -> {session.user_dominance} | AI {100-old_user_dom} -> {session.ai_dominance}")
        
        # === 生成语音 ===
        emotion = "angry" if dominance_shift < -5 else ("happy" if dominance_shift > 5 else "neutral")
        
        audio_path = None
        if self.tts:
            clean_text = re.sub(r'[（(][^）)]*[）)]', '', ai_text)
            audio_bytes = self.tts.synthesize(clean_text, emotion=emotion)
            audio_path = self._save_audio(session_id, audio_bytes)
            logger.info(f"[TTS] 生成语音: {audio_path}")
        
        session.chat_history.append((session.ai_name, ai_text))
        session.last_activity = time.time()
        
        logger.info("-" * 50)
        
        yield {
            "stage": "complete",
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
            "ai_text": ai_text,
            "audio_path": audio_path,
            "judgment": judgment,
            "dominance_shift": dominance_shift,
            "log": f"回合结束 | 气场: 用户 {session.user_dominance} vs AI {session.ai_dominance}"
        }
    
    def _clean_response(self, text: str, ai_name: str) -> str:
        text = text.strip()
        for prefix in [f"{ai_name}:", f"{ai_name}：", "你:", "你："]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text
    
    def _judge_dominance_zero_sum(self, session: Session, user_text: str, ai_text: str, scenario: Dict) -> Tuple[int, str]:
        """零和博弈裁判：返回用户气场变化值（正数=用户涨，负数=AI涨）"""
        
        judge_prompt = f"""你是专业的辩论/谈判裁判。分析这轮交锋，判断气场转移。

【场景】{scenario['name']}
【当前气场】用户 {session.user_dominance} vs AI {session.ai_dominance}（总和100）

【用户发言】
"{user_text}"

【{scenario['ai_name']}回应】
"{ai_text}"

【评判维度】
1. 论点强度：论据充分性、逻辑严密性
2. 气势表现：语气自信度、压迫感
3. 反击有效性：是否有效回应对方攻击
4. 心理战术：是否动摇对方信心

【输出格式】（严格按此格式，只输出两行）
气场转移: [整数，-25到+25，正数表示用户占优，负数表示AI占优]
点评: [一句话点评]"""

        result = self.llm.generate(judge_prompt, max_new_tokens=100)
        logger.debug(f"[裁判原始输出] {result}")
        
        shift = 0
        judgment = "势均力敌"
        
        for line in result.strip().split('\n'):
            if '气场转移' in line:
                match = re.search(r'[-+]?\d+', line)
                if match:
                    shift = max(-25, min(25, int(match.group())))
            elif '点评' in line:
                judgment = line.split(':', 1)[-1].split('：', 1)[-1].strip()
        
        return shift, judgment
    
    def _save_audio(self, session_id: str, audio_data: bytes) -> str:
        audio_dir = Path("outputs/audio") / session_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"turn_{self.sessions[session_id].turn_count}.wav"
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        return str(audio_path)
    
    def _resolve_tts_flag(self, enable_tts: Optional[bool]) -> bool:
        env_flag = os.environ.get("TTS_ENABLED", "1")  # 默认开启
        env_disabled = env_flag.lower() in {"0", "false", "no", "off"}
        if enable_tts is None:
            return not env_disabled
        return enable_tts