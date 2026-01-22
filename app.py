import gradio as gr
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict

print("[TalkArena] 正在导入模块...")

from core.orchestrator import Orchestrator

print("[TalkArena] 模块导入完成")

orchestrator: Optional[Orchestrator] = None

VISUALIZER_JS = Path("assets/visualizer.js").read_text(encoding="utf-8") if Path("assets/visualizer.js").exists() else ""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

* {
    font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
}

.subtitle {
    text-align: center;
    color: #a0aec0 !important;
    font-size: 1.1rem !important;
    margin-bottom: 2rem !important;
}

.arena-container {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.control-panel {
    background: linear-gradient(180deg, #1e1e30 0%, #252540 100%);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.section-title {
    color: #e2e8f0 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title::before {
    content: '';
    width: 4px;
    height: 20px;
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    border-radius: 2px;
}

#arena-canvas-container {
    width: 100%;
    height: 350px;
    border-radius: 16px;
    overflow: hidden;
    background: #0a0a1a;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
}

#arena-canvas {
    width: 100%;
    height: 100%;
}

.status-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #252540 100%);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    color: #cbd5e0;
    line-height: 1.8;
}

.dominance-bar {
    height: 12px !important;
    border-radius: 6px !important;
    background: linear-gradient(90deg, #1a1a2e 0%, #252540 100%) !important;
    overflow: hidden;
}

.dominance-fill-ai {
    background: linear-gradient(90deg, #ef4444 0%, #f97316 100%) !important;
    transition: width 0.5s ease-out;
}

.dominance-fill-user {
    background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%) !important;
    transition: width 0.5s ease-out;
}

.chat-container {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-radius: 16px;
    border: 1px solid rgba(102, 126, 234, 0.2);
}

.message-user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border-radius: 16px 16px 4px 16px !important;
    color: white !important;
    padding: 12px 16px !important;
    margin: 8px 0 !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.message-ai {
    background: linear-gradient(135deg, #374151 0%, #1f2937 100%) !important;
    border-radius: 16px 16px 16px 4px !important;
    color: #e5e7eb !important;
    padding: 12px 16px !important;
    margin: 8px 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.input-area {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.scenario-dropdown {
    background: #252540 !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

.start-btn {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 12px !important;
    margin-top: 12px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
}

.emotion-tag {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    margin-top: 8px;
    background: rgba(139, 92, 246, 0.2);
    color: #c4b5fd;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    margin: 4px 0;
}

.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.legend-dot-ai { background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); }
.legend-dot-user { background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); }

.tip-card {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 12px 16px;
    margin-top: 16px;
    color: #c4b5fd;
    font-size: 0.9rem;
}
"""


def get_orchestrator() -> Orchestrator:
    global orchestrator
    if orchestrator is None:
        print("[TalkArena] 初始化 Orchestrator...")
        orchestrator = Orchestrator(enable_tts=True)
        print("[TalkArena] Orchestrator 初始化完成")
    return orchestrator


def get_scenarios() -> List[Tuple[str, str]]:
    print("[TalkArena] 获取场景列表...")
    orch = get_orchestrator()
    scenarios = orch.get_scenario_list()
    print(f"[TalkArena] 获取到 {len(scenarios)} 个场景")
    return scenarios


def start_session(scenario_id: str) -> Tuple[str, List, str, float, float, str, str, str]:
    print(f"[TalkArena] 开始会话，场景ID: {scenario_id}")
    if not scenario_id:
        return "", [], "请先选择场景", 50, 50, "", "", "{}"
    
    orch = get_orchestrator()
    session = orch.start_session(scenario_id)
    scenario = orch.scenarios[scenario_id]
    
    ai_agent = None
    user_agent = None
    for agent in session.agents:
        if "用户" in agent.system_prompt or agent.role in ["求职者", "丈夫", "候选人", "客服"]:
            user_agent = agent
        else:
            ai_agent = agent
    
    if not ai_agent:
        ai_agent = session.agents[0]
    if not user_agent:
        user_agent = session.agents[1] if len(session.agents) > 1 else session.agents[0]
    
    ai_dominance = session.current_dominance.get(ai_agent.agent_id, 50)
    user_dominance = session.current_dominance.get(user_agent.agent_id, 50)
    
    status = f"""**{scenario['name']}**
{scenario['description']}

**你扮演**: {user_agent.name} ({user_agent.role})
**对方**: {ai_agent.name} ({ai_agent.role})"""
    
    agent_data = json.dumps([
        {"id": ai_agent.agent_id, "name": ai_agent.name, "dominance": ai_dominance},
        {"id": user_agent.agent_id, "name": user_agent.name, "dominance": user_dominance},
    ])
    
    return session.session_id, [], status, ai_dominance, user_dominance, ai_agent.name, user_agent.name, agent_data


def send_message(
    session_id: str,
    user_input: str,
    chat_history: List,
) -> Tuple[List, str, float, float, Optional[str], str]:
    print(f"[TalkArena] 发送消息，session_id: {session_id}, input: {user_input[:50] if user_input else 'None'}...")
    if not session_id:
        return chat_history, "请先开始会话", 50, 50, None, "{}"
    
    if not user_input.strip():
        return chat_history, "", 50, 50, None, "{}"
    
    orch = get_orchestrator()
    
    try:
        print("[TalkArena] 调用 process_turn...")
        user_turn, ai_turn = orch.process_turn(session_id, user_input)
        print(f"[TalkArena] AI 回复: {ai_turn.text[:50]}...")
        
        emotion_info = ""
        if ai_turn.emotion:
            emotion_info = f"\n[攻击性: {ai_turn.emotion.aggression_level.value}] [自信: {ai_turn.emotion.confidence_level}] [压力: {ai_turn.emotion.stress_level}]"
        
        ai_message = f"{ai_turn.speaker_name}: {ai_turn.text}{emotion_info}"
        chat_history.append((user_input, ai_message))
        
        dominance = orch.get_dominance_display(session_id)
        dominance_json = json.dumps(dominance)
        
        ai_dom = list(dominance.values())[0] if dominance else 50
        user_dom = list(dominance.values())[1] if len(dominance) > 1 else 50
        
        audio_path = ai_turn.audio_path
        print(f"[TalkArena] 消息处理完成，气场值: AI={ai_dom}, User={user_dom}")
        
        return chat_history, "", ai_dom, user_dom, audio_path, dominance_json
        
    except Exception as e:
        print(f"[TalkArena] 错误: {str(e)}")
        traceback.print_exc()
        error_msg = f"处理出错: {str(e)}"
        chat_history.append((user_input, error_msg))
        return chat_history, "", 50, 50, None, "{}"


CUSTOM_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
)


def create_ui():
    print("[TalkArena] 创建 UI...")
    with gr.Blocks(title="TalkArena - 动态社交博弈场") as demo:
        
        gr.HTML("""
        <h1 class="main-title">TalkArena</h1>
        <p class="subtitle">动态社交博弈场 - 训练你的高压沟通技巧</p>
        """)
        
        session_id = gr.State("")
        agent_data_state = gr.State("[]")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">场景设置</div>')
                
                with gr.Group(elem_classes="control-panel"):
                    scenarios = get_scenarios()
                    scenario_choices = [(name, sid) for sid, name in scenarios]
                    
                    scenario_dropdown = gr.Dropdown(
                        choices=scenario_choices,
                        label="选择场景",
                        value=scenario_choices[0][1] if scenario_choices else None,
                        elem_classes="scenario-dropdown",
                    )
                    
                    start_btn = gr.Button(
                        "开始对决",
                        variant="primary",
                        elem_classes="start-btn",
                    )
                    
                    status_text = gr.Markdown(
                        value="选择场景后点击开始",
                        elem_classes="status-card",
                    )
                
                gr.HTML('<div class="section-title" style="margin-top: 20px;">气场对决</div>')
                
                with gr.Group(elem_classes="control-panel"):
                    gr.HTML("""
                    <div id="arena-canvas-container">
                        <canvas id="arena-canvas"></canvas>
                    </div>
                    """)
                    
                    ai_name = gr.State("对方")
                    user_name = gr.State("你")
                    
                    gr.HTML('<div style="margin-top: 16px;">')
                    gr.HTML("""
                    <div class="legend-item">
                        <div class="legend-dot legend-dot-ai"></div>
                        <span style="color: #e2e8f0;">对方气场</span>
                    </div>
                    """)
                    ai_dominance = gr.Slider(
                        minimum=0, maximum=100, value=50,
                        label="", show_label=False,
                        interactive=False,
                        elem_classes="dominance-bar",
                    )
                    
                    gr.HTML("""
                    <div class="legend-item">
                        <div class="legend-dot legend-dot-user"></div>
                        <span style="color: #e2e8f0;">你的气场</span>
                    </div>
                    """)
                    user_dominance = gr.Slider(
                        minimum=0, maximum=100, value=50,
                        label="", show_label=False,
                        interactive=False,
                        elem_classes="dominance-bar",
                    )
                    gr.HTML('</div>')
                    
                    gr.HTML("""
                    <div class="tip-card">
                        <strong>提示:</strong> 气场值反映对话中的主导权。自信且有理有据的回复会提升你的气场！
                    </div>
                    """)
            
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">对话区域</div>')
                
                with gr.Group(elem_classes="arena-container"):
                    chatbot = gr.Chatbot(
                        label="",
                        height=450,
                        elem_classes="chat-container",
                    )
                    
                    with gr.Row(elem_classes="input-area"):
                        user_input = gr.Textbox(
                            label="",
                            placeholder="输入你的回复，展示你的沟通技巧...",
                            lines=2,
                            scale=5,
                            show_label=False,
                        )
                        send_btn = gr.Button(
                            "发送",
                            variant="primary",
                            scale=1,
                            elem_classes="send-btn",
                        )
                    
                    audio_output = gr.Audio(
                        label="语音回复",
                        visible=True,
                        autoplay=True,
                    )
        
        dominance_json = gr.State("{}")
        
        gr.HTML(f"""
        <script>
        {VISUALIZER_JS}
        
        // Initialize visualizer after DOM is ready
        setTimeout(() => {{
            initVisualizer();
        }}, 500);
        </script>
        """)
        
        start_btn.click(
            fn=start_session,
            inputs=[scenario_dropdown],
            outputs=[session_id, chatbot, status_text, ai_dominance, user_dominance, ai_name, user_name, agent_data_state],
        ).then(
            fn=None,
            inputs=[agent_data_state],
            outputs=None,
            js="(agentData) => { if(agentData) { updateVisualizerAgents(JSON.parse(agentData)); } }"
        )
        
        send_btn.click(
            fn=send_message,
            inputs=[session_id, user_input, chatbot],
            outputs=[chatbot, user_input, ai_dominance, user_dominance, audio_output, dominance_json],
        ).then(
            fn=None,
            inputs=[dominance_json],
            outputs=None,
            js="(dominanceMap) => { if(dominanceMap) { updateVisualizerDominance(JSON.parse(dominanceMap)); } }"
        )
        
        user_input.submit(
            fn=send_message,
            inputs=[session_id, user_input, chatbot],
            outputs=[chatbot, user_input, ai_dominance, user_dominance, audio_output, dominance_json],
        ).then(
            fn=None,
            inputs=[dominance_json],
            outputs=None,
            js="(dominanceMap) => { if(dominanceMap) { updateVisualizerDominance(JSON.parse(dominanceMap)); } }"
        )
    
    print("[TalkArena] UI 创建完成")
    return demo


if __name__ == "__main__":
    print("[TalkArena] ========================================")
    print("[TalkArena] TalkArena 启动中...")
    print("[TalkArena] ========================================")
    try:
        demo = create_ui()
        print("[TalkArena] 启动 Gradio 服务...")
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            theme=CUSTOM_THEME, 
            css=CUSTOM_CSS,
            show_error=True,
        )
    except Exception as e:
        print(f"[TalkArena] 启动失败: {str(e)}")
        traceback.print_exc()
