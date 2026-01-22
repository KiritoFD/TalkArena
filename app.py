import gradio as gr
from pathlib import Path
import sys
import os

print("[TalkArena] 正在导入模块...")

try:
    from ui.handlers import get_scenarios, start_session, send_message, process_voice_input, init_models
    from ui.theme import CUSTOM_CSS, CUSTOM_THEME
except ImportError as e:
    print(f"[TalkArena] 导入失败: {e}")
    os.system(f"{sys.executable} -m pip install torch transformers modelscope gradio SpeechRecognition -q")
    from ui.handlers import get_scenarios, start_session, send_message, process_voice_input, init_models
    from ui.theme import CUSTOM_CSS, CUSTOM_THEME

def create_ui():
    with gr.Blocks(title="TalkArena - 动态社交博弈场") as demo:
        gr.HTML("""
        <h1 style='text-align:center'>🎭 TalkArena - 动态社交博弈场</h1>
        <p style='text-align:center;color:#666'>气场零和博弈 | 犹豫就会败北 | 快速反击制胜</p>
        """)
        
        session_id = gr.State("")
        
        with gr.Row():
            with gr.Column(scale=1):
                scenario_dropdown = gr.Dropdown(
                    choices=get_scenarios(),
                    label="🎬 选择场景",
                    value=None,
                    type="value"
                )
                start_btn = gr.Button("⚔️ 开始对决", variant="primary")
                status_text = gr.Markdown("选择场景后点击开始")
                
                gr.Markdown("### 📊 气场对决 (总和100)")
                ai_dominance = gr.Slider(0, 100, 50, label="👔 对方气场", interactive=False)
                user_dominance = gr.Slider(0, 100, 50, label="💪 你的气场", interactive=False)
                
                gr.Markdown("""
                _⚡ 规则提示:_
                - _思考超过3秒开始掉气场_
                - _对方思考也会掉气场_
                - _裁判实时评判每轮交锋_
                """)
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=400, elem_classes="chat-container")
                
                with gr.Row():
                    user_input = gr.Textbox(
                        scale=4, 
                        placeholder="快速输入你的回击！犹豫就会掉气场...", 
                        show_label=False
                    )
                    send_btn = gr.Button("发送", scale=1, variant="primary")
                
                with gr.Row():
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤 语音输入（录音后自动发送）"
                    )
                
                audio_output = gr.Audio(label="🔊 对方回复", autoplay=True)
        
        start_btn.click(
            start_session,
            [scenario_dropdown],
            [session_id, chatbot, status_text, ai_dominance, user_dominance]
        )
        
        send_btn.click(
            send_message,
            [session_id, user_input, chatbot],
            [chatbot, user_input, ai_dominance, user_dominance, audio_output]
        )
        
        user_input.submit(
            send_message,
            [session_id, user_input, chatbot],
            [chatbot, user_input, ai_dominance, user_dominance, audio_output]
        )
        
        voice_input.stop_recording(
            process_voice_input,
            [session_id, voice_input, chatbot],
            [chatbot, user_input, ai_dominance, user_dominance, audio_output]
        )
    
    return demo

if __name__ == "__main__":
    print("=" * 60)
    print("TalkArena 启动中...")
    print("=" * 60)
    
    init_models()
    demo = create_ui()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=1234,
        show_error=True,
        share=True,
        inbrowser=True,
        css=CUSTOM_CSS
    )