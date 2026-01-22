import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .structs import (
    AgentProfile,
    SessionState,
    DialogueTurn,
    EmotionMetadata,
    compute_dominance_delta,
    clamp_dominance,
)
from .llm_engine import LLMEngine, EMOTION_INSTRUCTION
from .tts_engine import create_tts_engine


class Orchestrator:
    def __init__(
        self,
        llm_engine: Optional[LLMEngine] = None,
        tts_engine = None,
        agents_config_path: str = "data/default_agents.json",
        enable_tts: bool = True,
    ):
        self.llm_engine = llm_engine or LLMEngine()
        self.tts_engine = tts_engine if tts_engine else (create_tts_engine() if enable_tts else None)
        self.agents_config_path = Path(agents_config_path)
        self.scenarios: Dict[str, dict] = {}
        self.sessions: Dict[str, SessionState] = {}
        
        self._load_scenarios()
    
    def _load_scenarios(self):
        if self.agents_config_path.exists():
            with open(self.agents_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for scenario in data.get("scenarios", []):
                    self.scenarios[scenario["scenario_id"]] = scenario
    
    def get_scenario_list(self) -> List[Tuple[str, str]]:
        """Return list of (scenario_id, display_name) tuples."""
        return [(sid, s["name"]) for sid, s in self.scenarios.items()]
    
    def start_session(self, scenario_id: str) -> SessionState:
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        agents = [AgentProfile(**agent_data) for agent_data in scenario["agents"]]
        
        session = SessionState(
            scenario_id=scenario_id,
            agents=agents,
        )
        session.initialize_dominance()
        
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)
    
    def process_turn(
        self,
        session_id: str,
        user_input: str,
        user_agent_id: Optional[str] = None,
    ) -> Tuple[DialogueTurn, DialogueTurn]:
        """Process a turn: user input -> AI response. Returns (user_turn, ai_turn)."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        
        user_agent = None
        ai_agent = None
        for agent in session.agents:
            if user_agent_id and agent.agent_id == user_agent_id:
                user_agent = agent
            elif not user_agent_id and agent.role in ["user", "求职者", "应聘者", "丈夫", "妻子"]:
                user_agent = agent
            else:
                ai_agent = agent
        
        if not ai_agent:
            ai_agent = session.agents[0]
        if not user_agent:
            user_agent = session.agents[1] if len(session.agents) > 1 else session.agents[0]
        
        user_turn = DialogueTurn(
            speaker=user_agent.agent_id,
            speaker_name=user_agent.name,
            text=user_input,
        )
        session.history.append(user_turn)
        
        system_prompt = self._build_system_prompt(ai_agent, session)
        history_for_llm = self._format_history_for_llm(session.history[:-1], ai_agent.agent_id)
        
        raw_response = self.llm_engine.generate(
            user_input=user_input,
            system_prompt=system_prompt,
            history=history_for_llm,
        )
        
        clean_text, emotion = LLMEngine.parse_emotion(raw_response)
        
        dominance_delta = compute_dominance_delta(emotion)
        old_dominance = session.current_dominance.get(ai_agent.agent_id, 50.0)
        new_dominance = clamp_dominance(old_dominance + dominance_delta)
        session.current_dominance[ai_agent.agent_id] = new_dominance
        
        user_dominance = session.current_dominance.get(user_agent.agent_id, 50.0)
        user_new_dominance = clamp_dominance(user_dominance - dominance_delta * 0.5)
        session.current_dominance[user_agent.agent_id] = user_new_dominance
        
        audio_path = None
        if self.tts_engine:
            try:
                audio_path = self.tts_engine.synthesize(
                    text=clean_text,
                    speaker_id=ai_agent.tts_speaker_id,
                )
            except Exception as e:
                print(f"TTS synthesis failed: {e}")
        
        ai_turn = DialogueTurn(
            speaker=ai_agent.agent_id,
            speaker_name=ai_agent.name,
            text=clean_text,
            emotion=emotion,
            dominance_delta=dominance_delta,
            audio_path=audio_path,
        )
        session.history.append(ai_turn)
        session.turn_count += 1
        
        return user_turn, ai_turn
    
    def _build_system_prompt(self, agent: AgentProfile, session: SessionState) -> str:
        base_prompt = agent.system_prompt
        traits_str = "、".join(agent.personality_traits) if agent.personality_traits else "无特殊设定"
        
        prompt = f"""{base_prompt}

你的性格特点：{traits_str}
当前气场值：{session.current_dominance.get(agent.agent_id, 50):.1f}/100

{EMOTION_INSTRUCTION}"""
        
        return prompt
    
    def _format_history_for_llm(self, history: List[DialogueTurn], ai_agent_id: str) -> List[dict]:
        formatted = []
        for turn in history[-10:]:
            formatted.append({
                "text": turn.text,
                "is_ai": turn.speaker == ai_agent_id,
            })
        return formatted
    
    def get_dominance_display(self, session_id: str) -> Dict[str, float]:
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        result = {}
        for agent in session.agents:
            result[agent.name] = session.current_dominance.get(agent.agent_id, 50.0)
        return result
    
    def cleanup(self):
        if self.llm_engine:
            self.llm_engine.unload()
        if self.tts_engine:
            self.tts_engine.unload()
