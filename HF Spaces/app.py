import os
import openai
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionPipeline

# OpenAI text
openai.api_key = os.getenv('OPEN_AI_KEY')
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

tokens = os.getenv('ACCESS_TOKEN')
from huggingface_hub import login
login(token=tokens)

# Load model
def load_pipeline(repo_id_embeds):
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", revision="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.load_textual_inversion(repo_id_embeds)
    return pipe

repo_id_embeds_gopro = "sd-concepts-library/gopro-camera"
repo_id_embeds_nike = "sd-concepts-library/nike-shoes"
repo_id_embeds_echo = "sd-concepts-library/echo-speaker"

# Load both pipelines initially
pipelines={"<nike-shoes>": load_pipeline(repo_id_embeds_nike),
    "<gopro-camera>": load_pipeline(repo_id_embeds_gopro),
    "<echo-speaker>": load_pipeline(repo_id_embeds_echo)
    }

def generate_content_and_image(prompt):
    selected_pipeline = None
    for keyword, pipeline in pipelines.items():
        if keyword in prompt:
            selected_pipeline = pipeline
            break
    
    if selected_pipeline is None:
        # Default pipeline if no keyword matches
        selected_pipeline = pipelines["<gopro-camera>"]
    gpt_prompt = f"""Generate a text ad for a new product from the prompt within the "< >" below:
                    prompt = {prompt}.
                    Be short and creative. Not too short more like in 50 words"""
  
    response = get_completion(gpt_prompt)
    print("Response from OpenAI:", response)
    image = selected_pipeline(prompt).images[0]
    print("Image generated! Converting image ...")
    image_np = np.array(image)
    
    return response, image

iface = gr.Interface(
    fn=generate_content_and_image,
    inputs=gr.inputs.Textbox(label="Enter the prompt"),
    outputs=[
        gr.outputs.Textbox(label="Generated Response"),
        gr.outputs.Image(type="numpy",label="Generated Image")
    ],
    title="Adminds AI",
    description="#### Loaded models: GoPro, Nike shoes and Amazon echo speaker. Make sure to use '< gopro-camera >','< nike-shoes >' and '< echo-speaker >' as tokens into the prompt.\n\nEnter a prompt and get a creative ad response and image generated based on the prompt.",
    theme='xiaobaiyuan/theme_demo'
)

iface.launch()