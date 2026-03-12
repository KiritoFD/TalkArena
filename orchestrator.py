import uuid
import os
import re
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from model_loader import LLMLoader, TTSLoader
from core.prompts import get_rescue_master_fallback_prompt, get_scene_system_prompt

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
                "theme_color": "#4A90E2",
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
                "theme_color": "#D0021B",
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
                "theme_color": "#4A4A4A",
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
            },
            "shandong_dinner": {
                "name": "山东人的饭桌",
                "theme_color": "#F5A623",
                "characters": [
                    {
                        "name": "大舅",
                        "bio": "鲁中地区德高望重的长辈，担任“主陪”。热情但极讲规矩，擅长情感绑架和逻辑劝酒。",
                        "avatar": "👴"
                    },
                    {
                        "name": "大妗子",
                        "bio": "大舅的老伴，负责在旁边敲边鼓。明着是劝你别喝了，实则是在数你到底喝了几杯，并以此为由让大舅再敬你一个。",
                        "avatar": "👵"
                    },
                    {
                        "name": "表哥",
                        "bio": "大舅的儿子，酒桌上的“副陪”。负责起哄和活跃气氛，最擅长说‘我陪一个’然后让你干了。",
                        "avatar": "👨"
                    }
                ],
                "system_prompt": """场景：过年期间的家族聚餐，鲁中地区。用户（你）作为晚辈坐在这场酒局中。
酒桌角色：
1. 大舅（主陪）：灵魂人物，强势慈祥，极讲规矩。
2. 大妗子：在旁边‘明劝实激’，数着杯数。
3. 表哥（副陪）：起哄能手，最爱‘陪一个’。

任务：你现在要同时扮演这三个AI角色与用户对决。

【严格规则 - 必须遵守】：
1. **每一轮只能1个角色说话**
2. **禁止替用户说话！绝对不能出现"你:"或"用户:"开头的内容**
3. 角色要轮流随机发言，避免每次都是同一个人
4. 每个角色台词简短有力，不超过60字
5. 适当使用鲁中方言特色（如：昂、木有、杠好等），但要自然，不要刻意堆砌

【输出格式】：
大舅: [台词内容]

**严禁多个角色同时发言！只能1个角色！**
**绝对禁止**：你: [任何内容]""",
                "opening": "大舅:（站起来，红光满面）哎！那个谁，刚考上研那个外甥，别在那扣手机了！往主宾位坐坐。来，大舅先起个头，这第一杯酒，咱得全干了，这叫'开门红'，不喝就是不给大舅面子昂！"
            }
        }
    
    def get_scenario_list(self) -> List[Tuple[str, str]]:
        return [(k, v["name"]) for k, v in self.scenarios.items()]
    
    def start_session(self, scenario_id: str) -> Session:
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = self.scenarios[scenario_id]
        session_id = str(uuid.uuid4())[:8]
        
        # 处理多角色
        ai_name = scenario.get("ai_name")
        if not ai_name and "characters" in scenario:
            ai_name = " / ".join([c["name"] for c in scenario["characters"]])
        
        session = Session(
            session_id=session_id,
            scenario_id=scenario_id,
            user_name="你",
            ai_name=ai_name or "对手",
            user_dominance=50,
            chat_history=[],
            last_activity=time.time(),
            turn_count=0
        )
        
        self.sessions[session_id] = session
        
        # 处理开场白（可能包含多个角色的对话）
        opening = scenario["opening"]
        if "\n" in opening:
            for line in opening.split("\n"):
                if ":" in line:
                    name, text = line.split(":", 1)
                    session.chat_history.append((name.strip(), text.strip()))
                else:
                    session.chat_history.append((ai_name, line.strip()))
        else:
            session.chat_history.append((ai_name, opening))
        
        logger.info("=" * 60)
        logger.info(f"[SESSION {session_id}] 新对局开始")
        logger.info(f"  场景: {scenario['name']}")
        logger.info(f"  AI角色: {session.ai_name}")
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
        model_name = self.llm.get_model_name()
        logger.info(f"[AI思考] 开始生成回复... (模型: {model_name})")
        
        yield {
            "stage": "ai_thinking",
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
            "model_name": model_name,
            "think_start": think_start,
            "log": "AI 正在思考..."
        }
        
        context_lines = [f"{name}: {text}" for name, text in session.chat_history[-8:]]
        context = "\n".join(context_lines)

        # 获取当前场景的角色列表
        characters = scenario.get("characters", [])
        character_list_str = ""
        if characters:
            char_names = [f"{c.get('avatar', '')} {c['name']}" for c in characters]
            character_list_str = f"\n【可用角色列表】（你只能扮演以下角色，不能编造其他角色）\n" + "\n".join([f"- {name}" for name in char_names])

        ai_prompt_name = session.ai_name
        if "characters" in scenario:
            ai_prompt_name = "请根据场景角色进行回复"

        prompt = f"""{scenario['system_prompt']}
{character_list_str}

【当前局势】
你的气场: {session.ai_dominance}/100
对方气场: {session.user_dominance}/100
（气场越高越占优势，总和为100）

【对话记录】
{context}

【本轮回复要求】
1. **只能1个角色说话！严禁多个角色！**
2. **只能使用上面【可用角色列表】中的角色名，不能编造其他角色**
3. **绝对禁止替用户说话，不能出现"你:"开头的内容**
4. 完全进入角色，保持强势和攻击性
5. 针对对方刚才说的内容进行反驳、质疑或施压
6. 只输出对话内容，可含动作描写（用括号）
7. 格式："角色名: 内容"

{ai_prompt_name}:"""
        
        logger.debug(f"[AI思考] Prompt长度: {len(prompt)}字符")
        
        ai_text = self.llm.generate(prompt, max_new_tokens=400)
        ai_text = self._clean_response(ai_text, session.ai_name)
        
        # 如果 AI 返回空，使用 fallback 回复
        if not ai_text:
            logger.warning("[AI思考] LLM返回空，使用fallback回复")
            ai_text = "（沉默片刻）你说得很有意思，但我不同意。"
        
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

        # 检查是否达到游戏结束条件
        game_over = False
        game_result = None
        if session.user_dominance <= 5:
            game_over = True
            game_result = "ai_win"
            logger.info(f"[游戏结束] AI气场达到95，用户失败！")
        elif session.user_dominance >= 95:
            game_over = True
            game_result = "user_win"
            logger.info(f"[游戏结束] 用户气场达到95，用户胜利！")

        logger.info(f"[裁判判定] 气场转移: {dominance_shift:+d}")
        logger.info(f"[裁判点评] {judgment}")
        logger.info(f"[气场结果] 用户 {old_user_dom} -> {session.user_dominance} | AI {100-old_user_dom} -> {session.ai_dominance}")

        # === 生成语音 ===
        emotion = "angry" if dominance_shift < -5 else ("happy" if dominance_shift > 5 else "neutral")
        
        audio_path = None
        if self.tts:
            clean_text = re.sub(r'[（(][^）)]*[）)]', '', ai_text).strip()
            if clean_text:
                audio_bytes = self.tts.synthesize(clean_text, emotion=emotion)
                if audio_bytes:
                    audio_path = self._save_audio(session_id, audio_bytes)
                    logger.info(f"[TTS] 生成语音: {audio_path}")
                else:
                    logger.warning("[TTS] 语音合成失败，跳过")
            else:
                logger.warning(f"[TTS] 清理后文本为空，跳过 (ai_text={ai_text[:50] if ai_text else 'None'}...)")
        
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
            "game_over": game_over,
            "game_result": game_result,
            "log": f"回合结束 | 气场: 用户 {session.user_dominance} vs AI {session.ai_dominance}"
        }
    
    def get_rescue_suggestion(self, session_id: str) -> str:
        """救场逻辑：根据当前场景和对话历史，生成高情商回复供用户参考"""
        if session_id not in self.sessions:
            return "对局已结束"
        
        session = self.sessions[session_id]
        scenario = self.scenarios[session.scenario_id]
        
        context_lines = [f"{name}: {text}" for name, text in session.chat_history[-10:]]
        context = "\n".join(context_lines)
        
        prompt = get_rescue_master_fallback_prompt(
            scene_name=scenario['name'],
            ai_name=session.ai_name,
            user_dominance=session.user_dominance,
            ai_dominance=session.ai_dominance,
            context=context
        )
        
        suggestion = self.llm.generate(prompt, max_new_tokens=150)
        logger.info(f"[救场] Session {session_id} 生成建议: {suggestion[:50]}...")
        return suggestion
    
    def process_rescue_turn(self, session_id: str, rescue_text: str) -> Generator:
        """处理救场大师发言后，AI对手的回应"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        scenario = self.scenarios[session.scenario_id]
        session.turn_count += 1
        
        logger.info(f"[SESSION {session_id}] 救场大师介入，对手回应中...")
        
        session.last_activity = time.time()
        
        yield {
            "stage": "ai_thinking",
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
        }
        
        think_start = time.time()
        
        context_lines = [f"{name}: {text}" for name, text in session.chat_history[-8:]]
        context = "\n".join(context_lines)

        # 获取当前场景的角色列表
        characters = scenario.get("characters", [])
        character_list_str = ""
        if characters:
            char_names = [f"{c.get('avatar', '')} {c['name']}" for c in characters]
            character_list_str = f"\n【可用角色列表】（你只能扮演以下角色，不能编造其他角色）\n" + "\n".join([f"- {name}" for name in char_names])

        ai_prompt_name = session.ai_name
        if "characters" in scenario:
            ai_prompt_name = "请根据场景角色进行回复"

        prompt = f"""{scenario['system_prompt']}
{character_list_str}

【当前局势】
你的气场: {session.ai_dominance}/100
对方气场: {session.user_dominance}/100

【对话记录】
{context}

【特别说明】
刚才有一位"救场大师"介入帮助对方说话了。你需要回应这位救场大师的发言。
可以表现出对外援介入的不满，继续保持攻势。

【本轮回复要求】
1. **只能1个角色说话！严禁多个角色！**
2. **只能使用上面【可用角色列表】中的角色名，不能编造其他角色**
3. **绝对禁止替用户说话，不能出现"你:"开头的内容**
4. 完全进入角色，保持强势
5. 回应救场大师的发言内容
6. 只输出对话内容，可含动作描写（用括号）
7. 格式："角色名: 内容"

{ai_prompt_name}:"""
        
        ai_text = self.llm.generate(prompt, max_new_tokens=400)
        ai_text = self._clean_response(ai_text, session.ai_name)
        
        if not ai_text:
            ai_text = "（冷笑）哦？还请外援了？那也没用。"
        
        think_time = time.time() - think_start
        logger.info(f"[AI回复] ({think_time:.1f}s) {ai_text[:100]}...")
        
        audio_path = None
        if self.tts:
            clean_text = re.sub(r'[（(][^）)]*[）)]', '', ai_text).strip()
            if clean_text:
                audio_bytes = self.tts.synthesize(clean_text, emotion="angry")
                if audio_bytes:
                    audio_path = self._save_audio(session_id, audio_bytes)
        
        session.chat_history.append((session.ai_name, ai_text))
        session.last_activity = time.time()
        
        yield {
            "stage": "complete",
            "user_dominance": session.user_dominance,
            "ai_dominance": session.ai_dominance,
            "ai_text": ai_text,
            "audio_path": audio_path,
        }

    def _clean_response(self, text: str, ai_name: str) -> str:
        if not text:
            logger.warning(f"[_clean_response] 输入文本为空")
            return ""
        text = text.strip()
        
        # 如果包含多个冒号换行，说明是多角色模式，不删除前缀
        lines = text.split('\n')
        if len(lines) > 1 and all(':' in l or '：' in l for l in lines if l.strip()):
            logger.debug(f"[_clean_response] 检测到多角色回复，保留格式")
            return text
            
        for prefix in [f"{ai_name}:", f"{ai_name}：", "你:", "你：", "助手:", "AI:", "Assistant:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        logger.debug(f"[_clean_response] 清理后: {len(text)}字符")
        return text
    
    def _judge_dominance_zero_sum(self, session: Session, user_text: str, ai_text: str, scenario: Dict) -> Tuple[int, str]:
        """零和博弈裁判：返回用户气场变化值（正数=用户涨，负数=AI涨）"""
        
        judge_prompt = f"""你是专业的辩论/谈判裁判。分析这轮交锋，判断气场转移。

【场景】{scenario['name']}
【当前气场】用户 {session.user_dominance} vs AI {session.ai_dominance}（总和100）

【用户发言】
"{user_text}"

【{session.ai_name}回应】
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
    
    def end_session_with_summary(self, session_id: str) -> Tuple[str, str]:
        """结束对决，生成总结、建议并保存文件"""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        scenario = self.scenarios[session.scenario_id]
        
        # 构建对话记录
        dialogue = "\n".join([f"{name}: {text}" for name, text in session.chat_history])
        
        # 计算结果
        if session.user_dominance > 60:
            result = "🏆 用户胜出"
        elif session.user_dominance < 40:
            result = "💢 AI 胜出"
        else:
            result = "🤝 势均力敌"
        
        # 让 LLM 生成总结
        summary_prompt = f"""你是一位专业的沟通教练。分析以下对决并给出详细点评和改进建议。

【场景】{scenario['name']}
【对手】{session.ai_name}
【最终气场】用户 {session.user_dominance} vs AI {session.ai_dominance}
【回合数】{session.turn_count}

【对话记录】
{dialogue}

请 output（严格按以下 format）：

## 🎯 对决结果
[{result}，最终气场比分]

## 📊 表现分析
- 优势: [列举2-3个亮点]
- 不足: [列举2-3个问题]

## 🔑 关键回合复盘
[指出1-2个关键转折点，分析为什么赢/输]

## 💡 改进建议
[给出3条具体可操作的建议]"""
        
        summary = self.llm.generate(summary_prompt, max_new_tokens=800)
        
        logger.info("=" * 60)
        logger.info(f"[SESSION {session_id}] 对决结束")
        logger.info(f"  结果: {result}")
        logger.info(f"  最终气场: 用户 {session.user_dominance} vs AI {session.ai_dominance}")
        logger.info(f"  总回合数: {session.turn_count}")
        logger.info("=" * 60)
        
        # 保存对决记录
        file_content = f"""# TalkArena 对决记录

## 基本信息
- 场景: {scenario['name']}
- 对手: {session.ai_name}
- 回合数: {session.turn_count}
- 最终气场: 用户 {session.user_dominance} vs AI {session.ai_dominance}
- 结果: {result}

## 对话记录
{dialogue}

## 总结与建议
{summary}
"""
        
        output_dir = Path("outputs/sessions")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{session_id}_summary.md"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        logger.info(f"[保存] 对决记录: {file_path}")
        
        # 清理 session
        del self.sessions[session_id]
        
        return summary, str(file_path)
    
    def generate_game_report(self, session_id: str, scene_name: str, npc_list: List[Dict]) -> Dict:
        """生成游戏结束后的全面复盘报告"""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        scenario = self.scenarios.get(session.scenario_id, {})
        
        # 构建对话历史
        history_log = "\n".join([f"{name}: {text}" for name, text in session.chat_history])
        
        # 第一次调用：生成五维度得分
        scores_prompt = f"""# Role
你是“山东人饭局情商大挑战”的打分裁判，负责给玩家在饭局对话中的表现从五个维度打分。

# Input
- 场景描述：{scene_name}
- NPC设定列表：{json.dumps(npc_list, ensure_ascii=False)}
- 历史对话：
{history_log}

# Task
分析对话，给出玩家在五个维度的客观得分，满分10，输出从0-100的数值。5个指标如下：
1. "oily": 圆滑度：避重就轻、推诱话题的能力,
2. "friendliness": 亲和力：共情与情绪价值提供,
3. "logic": 逻辑性：论据支撑与表达条理,
4. "humor": 幽默感：破冰与自嘲能力,
5. "respect": 懂规矩：礼仪遵守与分寸感。

# Output Format (JSON Only)
{{
  "metrics": {{
    "oily": int,
    "friendliness": int,
    "logic": int,
    "humor": int,
    "respect": int
  }}
}}

# Constraints
只输出 JSON格式，不得输出任何额外解释文字"""
        
        logger.info("[复盘报告] 步骤1: 生成五维度得分...")
        scores_result = self.llm.generate(scores_prompt, max_new_tokens=200)
        
        # 解析JSON
        try:
            scores_data = json.loads(scores_result.strip())
            scores = scores_data.get("metrics", {})
        except:
            logger.warning("[复盘报告] JSON解析失败，使用默认分数")
            scores = {"oily": 50, "friendliness": 50, "logic": 50, "humor": 50, "respect": 50}
        
        # 计算勋章
        from ui.report import get_medal_by_scores
        medal = get_medal_by_scores(scores)
        
        # 第二次调用：综合点评
        summary_prompt = f"""# Role
你是一位在山东饭局混迹三十年、眼光毒辣的人情世故宗师。你的任务是根据玩家在“山东人饭局情商大挑战”中的对话表现，给出一份既专业又扎心的总结陈词。

# Input
- 场景描述：{scene_name}
- NPC设定列表：{json.dumps(npc_list, ensure_ascii=False)}
- 历史对话：
{history_log}
- 玩家称号：{medal}

# Task 
分析对话历史，撰写一段 100 字以内的玩家表现综合点评。

# Writing Constraints
- 犠利度：不要客气，要像一位严厉的长辈或刻薄的职场前辈。如果表现差，请使用“社交自杀”、“拆迁队”、“冷场王”等词汇。
- 专业深度：点评必须基于真实的社交潜规则。
- 称号挂钩：点评必须匹配生成的玩家称号。
- 结构化：第一句：定性评价；中间语句：逻辑分析；结尾句：总结。

# Constraints
直接输出总结陈词内容，不得输出任何额外解释文字"""
        
        logger.info("[复盘报告] 步骤2: 生成综合点评...")
        summary = self.llm.generate(summary_prompt, max_new_tokens=300)
        
        # 第三次调用：NPC OS + 改进建议
        npc_prompt = f"""# Role
你是一位在山东饭局混迹三十年、毒舌且看透世事的“人情世故大宗师”。

# Input Data
- 场景描述：{scene_name}
- NPC设定列表：{json.dumps(npc_list, ensure_ascii=False)}
- 历史对话：
{history_log}
- 玩家称号：{medal}

# Tasks
1. 生成 NPC 内心 OS：为 NPC 列表中的每人生成一段 20 字以内的心理活动。要求口语化，符合人设。
2. 生成改进建议：针对玩家最不合时宜的一句话，给出高情商台词改写及避坑逻辑。

# Output Format (Strict JSON)
{{
  "npc_inner_voice": [
    {{"name": "...", "os": "..."}},
    {{"name": "...", "os": "..."}}
  ],
  "high_light_suggestion": "..."
}}

# Constraints
只输出 JSON格式，不得输出任何额外解释文字"""
        
        logger.info("[复盘报告] 步骤3: 生成NPC OS和建议...")
        npc_result = self.llm.generate(npc_prompt, max_new_tokens=500)
        
        # 解析JSON
        try:
            npc_data = json.loads(npc_result.strip())
            npc_os_list = npc_data.get("npc_inner_voice", [])
            suggestion = npc_data.get("high_light_suggestion", "没有具体建议")
        except:
            logger.warning("[复盘报告] NPC JSON解析失败")
            npc_os_list = [{"name": npc["name"], "os": "表现一般", "avatar": npc.get("avatar", "👤")} for npc in npc_list[:3]]
            suggestion = "多观察，少说话。"
        
        # 添加avatar到NPC OS
        for os_item in npc_os_list:
            npc_name = os_item.get("name", "")
            for npc in npc_list:
                if npc.get("name") == npc_name:
                    os_item["avatar"] = npc.get("avatar", "👤")
                    break
        
        logger.info("[复盘报告] 生成完成")
        
        return {
            "scene_name": scene_name,
            "medal": medal,
            "scores": scores,
            "summary": summary,
            "npc_os_list": npc_os_list,
            "suggestion": suggestion
        }