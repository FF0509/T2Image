import os
from dotenv import load_dotenv
import gradio as gr
from openai import AsyncOpenAI
import base64
from io import BytesIO
from PIL import Image
import asyncio

load_dotenv()
TIMEOUT = 60
STIMA_KEY = os.getenv("STIMA_API_KEY")
STIMA_URL = "https://api.stima.tech/v1"

MODELS = {
    "OpenAI / GPT-4o Image Generation": ("stima", "gpt-4o-image"),
    "OpenAI / DALL·E 3": ("stima", "dall-e-3"),
    "Google / Gemini 2.0 Flash Exp Image Generation": ("stima", "gemini-2.0-flash-exp-image"),
    # "Google / Veo 3 Fast": ("stima", "veo3-fast"),
    # "Google / Veo 3": ("stima", "veo3"),
    # "Google / Veo 3 Pro": ("stima", "veo3-pro"),
    # "Google / Veo 3 Pro Frames": ("stima", "veo3-pro-frames"),
    "xAI / Grok 3 Image Generation": ("stima", "grok-3-image"),
    # "NanoBanana / Nano Banana": ("stima", "fal-ai/nano-banana"),
}

def get_client():
    return AsyncOpenAI(
        api_key=STIMA_KEY,
        base_url=STIMA_URL,
        timeout=TIMEOUT
    )


async def generate_image(prompt, model_key):
    if not STIMA_KEY:
        raise gr.Error("API Key not find in enviroment")
    
    client = get_client()
    provider, model = MODELS[model_key]
    
    try:
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
            user="huggingface-user"
        )
        
        if response.data and response.data[0].b64_json:
            # base64解碼
            image_data = base64.b64decode(response.data[0].b64_json)
            image = Image.open(BytesIO(image_data))
            return image
        else:
            raise gr.Error("Error")
    
    except Exception as e:
        raise gr.Error(f"生成失敗：{str(e)}")

def sync_generate_image(prompt, model_key):
    return asyncio.run(generate_image(prompt, model_key))
    
with gr.Blocks(title="AI圖像生成工坊", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI圖像生成工坊\n\n選擇模型，輸入prompt，靜待奇蹟。")
    
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="OpenAI / GPT-4o Image Generation",
                label="選擇模型"
            )
            prompt = gr.Textbox(
                label="你的prompt",
                lines=3,
                placeholder="例如：一個在秋葉中飛舞的機械蝴蝶，背景是廢棄的鐘塔。"
            )
            submit = gr.Button("生成圖像", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Image(label="生成的夢境")
    
    examples = gr.Examples(
        examples=[["a boy standing in the forest, smile and close eyes。"]],
        inputs=[prompt],
        label="靈感示例"
    )
    
    submit.click(
        fn=sync_generate_image,
        inputs=[prompt, model],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch()
