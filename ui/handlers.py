from typing import List, Tuple, Generator
from orchestrator import Orchestrator, logger

_orchestrator_instance = None

def init_models():
    logger.info("Handlers 初始化模型...")
    global _orchestrator_instance
    _orchestrator_instance = Orchestrator(enable_tts=True)
    logger.info("Handlers 模型初始化完成")

def get_orchestrator() -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        init_models()
    return _orchestrator_instance

def get_scenarios() -> List[Tuple[str, str]]:
    orch = get_orchestrator()
    return [(scenario.get('name', sid), sid) for sid, scenario in orch.scenarios.items()]

def start_session(scenario_id: str):
    if not scenario_id:
        return "", [], "❌ 请先选择场景", 50, 50
    
    orch = get_orchestrator()
    session = orch.start_session(scenario_id)
    
    chat_history = []
    for name, text in session.chat_history:
        chat_history.append({"role": "assistant", "content": f"**{name}**: {text}"})
    
    status = f"✓ 对局开始 | 场景: {orch.scenarios[scenario_id]['name']}"
    
    return session.session_id, chat_history, status, session.ai_dominance, session.user_dominance

def process_voice_input(session_id: str, audio_file, chat_history: List) -> Generator:
    if not session_id:
        yield chat_history, "", 50, 50, None
        return
    
    if audio_file is None:
        yield chat_history, "", 50, 50, None
        return
    
    orch = get_orchestrator()
    
    try:
        user_text = orch.transcribe_audio(audio_file)
        logger.info(f"[语音输入] 转录成功: {user_text}")
    except Exception as e:
        logger.error(f"[语音输入] 转录失败: {e}")
        yield chat_history, f"语音识别失败: {e}", 50, 50, None
        return
    
    yield from send_message(session_id, user_text, chat_history)

def send_message(session_id: str, user_input: str, chat_history: List) -> Generator:
    if not session_id or not user_input.strip():
        yield chat_history, "", 50, 50, None
        return
    
    orch = get_orchestrator()
    
    chat_history = list(chat_history)
    chat_history.append({"role": "user", "content": user_input})
    
    for update in orch.process_turn_streaming(session_id, user_input):
        stage = update["stage"]
        ai_dom = update["ai_dominance"]
        user_dom = update["user_dominance"]
        
        if stage in ("user_sent", "ai_thinking", "ai_responded"):
            yield chat_history, "", ai_dom, user_dom, None
        
        elif stage == "complete":
            ai_text = update["ai_text"]
            audio_path = update["audio_path"]
            judgment = update.get("judgment", "")
            shift = update.get("dominance_shift", 0)
            
            shift_str = f"+{shift}" if shift > 0 else str(shift)
            ai_name = orch.scenarios[orch.sessions[session_id].scenario_id]['ai_name']
            
            display_text = f"**{ai_name}**: {ai_text}\n\n---\n_📊 {judgment} (气场{shift_str})_"
            
            chat_history.append({"role": "assistant", "content": display_text})
            
            yield chat_history, "", ai_dom, user_dom, audio_path