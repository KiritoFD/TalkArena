import json
import uuid
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from model_loader import LLMLoader, TTSLoader

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
    user_dominance: int
    ai_dominance: int
    chat_history: List[Tuple[str, str]]

class Orchestrator:
    def __init__(self, enable_tts: bool = True):
        self.llm = LLMLoader()
        self.tts = TTSLoader() if enable_tts else None
        self.sessions: Dict[str, Session] = {}
        self.scenarios = self._load_scenarios()
        
        print("[Orchestrator] 初始化模型加载器...")
        self.llm.load()
        if self.tts:
            self.tts.load()
    
    def _load_scenarios(self) -> Dict:
        return {
            "negotiation": {
                "name": "商务谈判",
                "ai_name": "王总",
                "user_prompt": "你是一位商务代表",
                "ai_prompt": "你是一位强势的商业谈判对手"
            },
            "debate": {
                "name": "辩论赛",
                "ai_name": "对手",
                "user_prompt": "你是辩手，立场是正方",
                "ai_prompt": "你是辩手，立场是反方，性格犀利"
            },
            "interview": {
                "name": "面试官",
                "ai_name": "面试官",
                "user_prompt": "你是求职者",
                "ai_prompt": "你是面试官，需要严格评估候选人"
            }
        }
    
    def get_scenario_list(self) -> List[Tuple[str, str]]:
        return [(k, v["name"]) for k, v in self.scenarios.items()]
    
    def start_session(self, scenario_id: str) -> Session:
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = self.scenarios[scenario_id]
        session_id = str(uuid.uuid4())
        
        session = Session(
            session_id=session_id,
            scenario_id=scenario_id,
            user_name="你",
            ai_name=scenario["ai_name"],
            user_dominance=50,
            ai_dominance=50,
            chat_history=[]
        )
        
        self.sessions[session_id] = session
        
        # AI开场白
        opening = f"我是{scenario['ai_name']}，很高兴见到你。准备好了吗？"
        session.chat_history.append(("系统", opening))
        
        return session
    
    def process_turn(self, session_id: str, user_input: str) -> Tuple[Turn, Turn]:
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        scenario = self.scenarios[session.scenario_id]
        
        # 更新用户气场
        session.user_dominance = min(100, session.user_dominance + 2)
        
        # 生成AI回复
        context = "\n".join([f"{name}: {text}" for name, text in session.chat_history[-4:]])
        prompt = f"{scenario['ai_prompt']}\n\n对话记录:\n{context}\n\n你的回复:"
        
        ai_text = self.llm.generate(prompt, max_length=150)
        
        # 情感分析（简单版）
        emotion = self._analyze_emotion(ai_text)
        
        # TTS合成
        audio_path = None
        if self.tts:
            audio = self.tts.synthesize(ai_text, emotion=emotion)
            if audio:
                audio_path = self._save_audio(session_id, audio)
        
        # 更新AI气场
        session.ai_dominance = min(100, session.ai_dominance + 1)
        
        # 记录到历史
        session.chat_history.append((session.user_name, user_input))
        session.chat_history.append((session.ai_name, ai_text))
        
        user_turn = Turn(text=user_input)
        ai_turn = Turn(text=ai_text, audio_path=audio_path, emotion=emotion)
        
        return user_turn, ai_turn
    
    def _analyze_emotion(self, text: str) -> str:
        """简单的情感分析"""
        if any(w in text for w in ["！", "很好", "棒", "太好"]):
            return "happy"
        elif any(w in text for w in ["？", "不", "错"]):
            return "neutral"
        elif any(w in text for w in ["呃", "嗯", "不过"]):
            return "sad"
        return "neutral"
    
    def _save_audio(self, session_id: str, audio_data) -> str:
        from pathlib import Path
        import shutil
        
        audio_dir = Path("outputs/audio") / session_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"response_{len(self.sessions[session_id].chat_history)}.wav"
        
        # 如果是文件路径，复制文件
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            shutil.copy(audio_data, audio_path)
        # 如果是字节数据，直接写入
        elif isinstance(audio_data, bytes):
            with open(audio_path, "wb") as f:
                f.write(audio_data)
        else:
            return None
        
        return str(audio_path)
