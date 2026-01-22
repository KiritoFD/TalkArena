import gradio as gr

# 自定义CSS样式
CUSTOM_CSS = """
.main-title {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 0.5em;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 2em;
}
.scenario-dropdown {
    font-size: 1.1em;
}
.start-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 1.1em;
    padding: 10px;
}
.send-btn {
    background: #667eea;
    color: white;
}
.chat-container {
    border-radius: 10px;
}
.input-area {
    margin-top: 10px;
}
"""

# 使用默认主题，避免版本兼容问题
CUSTOM_THEME = None
