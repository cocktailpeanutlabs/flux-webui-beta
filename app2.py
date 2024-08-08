import gradio as gr
import numpy as np
import random
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import QuantizedDiffusersModel, QuantizedTransformersModel
import json
import devicetorch
import os
dtype = torch.bfloat16
#dtype = torch.float32
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
selected = None
css="""
nav {
  text-align: center;
-q8}
#logo{
  width: 50px;
  display: inline;
}
"""
def infer(prompt, checkpoint="black-forest-labs/FLUX.1-schnell", seed=42, num_images_per_prompt=1, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    global pipe
    global selected
    # if the new checkpoint is different from the selected one, re-instantiate the pipe
    print(f"device={device}")
    if selected != checkpoint:
        if checkpoint == "sayakpaul/FLUX.1-merged":
            #transformer = FluxTransformer2DModel.from_pretrained("sayakpaul/FLUX.1-merged", torch_dtype=dtype)
            #pipe = FluxPipeline.from_pretrained("cocktailpeanut/xulf-d", transformer=transformer, torch_dtype=dtype)
            bfl_repo = "cocktailpeanut/xulf-d"
            print("1")
            class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
                base_class = FluxTransformer2DModel
            print("2")
            if device == "mps":
                transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-merged-qint8")
            else:
                transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-merged-q8")
            transformer.to(device=device, dtype=dtype)
            print("3")
            pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, torch_dtype=dtype)
            print("4")
            pipe.transformer = transformer
            print("5")
        else:
            bfl_repo = "cocktailpeanut/xulf-s"
            print("1")
            class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
                base_class = FluxTransformer2DModel
            print("2")
            if device == "mps":
                transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-schnell-qint8")
            else:
                transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-schnell-q8")
            transformer.to(device=device, dtype=dtype)
            print("3")
            pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, torch_dtype=dtype)
            print("4")
            pipe.transformer = transformer
            print("5")
        pipe.to(device)
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        print("9")
        selected = checkpoint
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    print("10")
    images = pipe(
            prompt = prompt,
            width = width,
            height = height,
            num_inference_steps = num_inference_steps,
            generator = generator,
            num_images_per_prompt = num_images_per_prompt,
            guidance_scale=0.0
    ).images
    print("11")
    devicetorch.empty_cache(torch)
    print("12")
    return images, seed
def update_slider(checkpoint, num_inference_steps):
    if checkpoint == "sayakpaul/FLUX.1-merged":
        return 8
    else:
        return 4
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("<nav><img id='logo' src='file/icon.png'/></nav>")
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", show_label=False, object_fit: "contain")
        checkpoint = gr.Dropdown(
          label="Model",
          value= "black-forest-labs/FLUX.1-schnell",
          choices=[
            "black-forest-labs/FLUX.1-schnell",
            "sayakpaul/FLUX.1-merged"
          ]
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=576,
            )
        with gr.Row():
            num_images_per_prompt = gr.Slider(
                label="Number of images",
                minimum=1,
                maximum=50,
                step=1,
                value=1,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=50,
                step=1,
                value=4,
            )
        checkpoint.change(fn=update_slider, inputs=[checkpoint], outputs=[num_inference_steps])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, checkpoint, seed, num_images_per_prompt, randomize_seed, width, height, num_inference_steps],
        outputs = [result, seed]
    )
demo.launch()

