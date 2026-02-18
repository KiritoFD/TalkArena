#!/bin/bash
# 快速重启脚本

echo "正在重启应用..."
lsof -ti:7860 | xargs kill -9 2>/dev/null
sleep 1
source venv/bin/activate
python app.py
