import os
from dotenv import load_dotenv
import gradio as gr
from openai import AsyncOpenAI
import base64
from io import BytesIO
from PIL import Image
import asyncio

# 环境加载，如晨光渗入窗帘
load_dotenv()
TIMEOUT = 60
STIMA_KEY = os.getenv("STIMA_API_KEY")
STIMA_URL = "https://api.stima.tech/v1"

# 模型目录，如一本古老的魔法书
MODELS = {
    "OpenAI / GPT-4o Image Generation": ("stima", "gpt-4o-image"),
    "OpenAI / DALL·E 3": ("stima", "dall-e-3"),
    "Google / Gemini 2.0 Flash Exp Image Generation": ("stima", "gemini-2.0-flash-exp-image"),
    "Google / Veo 3 Fast": ("stima", "veo3-fast"),
    "Google / Veo 3": ("stima", "veo3"),
    "Google / Veo 3 Pro": ("stima", "veo3-pro"),
    "Google / Veo 3 Pro Frames": ("stima", "veo3-pro-frames"),
    "xAI / Grok 3 Image Generation": ("stima", "grok-3-image"),
    "NanoBanana / Nano Banana": ("stima", "fal-ai/nano-banana"),
}

def get_client():
    """客户端的诞生，如精灵从瓶中释放"""
    return AsyncOpenAI(
        api_key=STIMA_KEY,
        base_url=STIMA_URL,
        timeout=TIMEOUT
    )
