# API Tracer 测试说明

## 环境依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers huggingface_hub

# image video
pip install diffusers decord

# moonshotai
pip install tiktoken blobfile

# RWKV
pip install flash-linear-attention

# baidu
pip install sentencepiece moviepy

# OpenGVLab
pip install timm

# MonkeyOCR
# refer to https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support

# Wan-AI
pip install ftfy

# Keye
pip install "keye-vl-utils[decord]==1.0.0"

# deepseek-vl2
# refer to https://github.com/deepseek-ai/DeepSeek-VL2?tab=readme-ov-file#4-quick-start

# deepseek-v2
pip install flash_attn

# mistral
pip install mistral-common

```

## 项目调用

```python
    import sys
    import os

    sys.path.append(
        os.path.join(os.path.dirname(__file__), "../PaddleAPITest/tools/api_tracer")
    )
    from api_tracer import APITracer
```
