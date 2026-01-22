# TalkArena - 动态社交博弈场

## 安装依赖

```bash
# 创建虚拟环境
conda create -n arena python=3.10
conda activate arena

# 安装核心依赖
pip install -r requirements.txt

# 如果需要GPU加速
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import modelscope; import transformers; import gradio; print('✓ 依赖安装成功')"
```

## 运行

```bash
python app.py
```

访问: http://127.0.0.1:1234

## 故障排除

### datasets版本问题
```bash
pip install "datasets<3.0.0" --force-reinstall
```

### TTS依赖问题
```bash
pip install addict scipy soundfile librosa
```

### ModelScope下载慢
设置镜像源:
```bash
export MODELSCOPE_CACHE=./models
```
