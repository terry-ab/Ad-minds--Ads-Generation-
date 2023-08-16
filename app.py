from flask import Flask, render_template, request
from dotenv import load_dotenv
load_dotenv()
import os
import openai

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

#Openai text
openai.api_key = os.getenv('OPEN_AI_KEY')
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

#Login to the Hugging Face Hub
tokens= os.getenv('ACCESS_TOKEN')
from huggingface_hub import login
login(token=tokens)

# Load model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", revision="fp16", torch_dtype=torch.float16).to("cuda")
repo_id_embeds = "sd-concepts-library/gopro-camera"
#Load the concept into pipeline
pipe.load_textual_inversion(repo_id_embeds)

# Start flask app and set to ngrok
app = Flask(__name__)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")
  gpt_prompt=f"""Generate a text ad for a new product  from the prompt within the "< >" below:
                prompt = {prompt}.
                Be short and creative. Not too short more like in 50 words"""
  
  response = get_completion(gpt_prompt)
  print("Response from OpenAI:", response)
  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  print("Sending image ...")
  return render_template('submit.html', generated_image=img_str,generated_response=response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)