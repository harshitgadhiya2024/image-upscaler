from pathlib import Path

import gradio as gr
import pillow_heif
import spaces
import torch
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from PIL import Image
from refiners.fluxion.utils import manual_seed
from refiners.foundationals.latent_diffusion import Solver, solvers

from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints

pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

TITLE = """
<center>

  <h1 style="font-size: 1.5rem; margin-bottom: 0.5rem;">
    Image Enhancer Powered By Refiners
  </h1>

  <div style="
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 1.25rem;
    flex-wrap: wrap;
  ">
    <a href="https://blog.finegrain.ai/posts/reproducing-clarity-upscaler/" target="_blank">[Blog Post]</a>
    <a href="https://github.com/finegrain-ai/refiners" target="_blank">[Refiners]</a>
    <a href="https://finegrain.ai/" target="_blank">[Finegrain]</a>
    <a href="https://huggingface.co/spaces/finegrain/finegrain-object-eraser" target="_blank">
        [Finegrain Object Eraser]
    </a>
    <a href="https://huggingface.co/spaces/finegrain/finegrain-object-cutter" target="_blank">
        [Finegrain Object Cutter]
    </a>
  </div>

  <p>
    Turn low resolution images into high resolution versions with added generated details (your image will be modified).
  </p>

  <p>
    This space is powered by Refiners, our open source micro-framework for simple foundation model adaptation.
    If you enjoyed it, please consider starring Refiners on GitHub!
  </p>

  <a href="https://github.com/finegrain-ai/refiners" target="_blank">
    <img src="https://img.shields.io/github/stars/finegrain-ai/refiners?style=social" />
  </a>

</center>
"""

CHECKPOINTS = ESRGANUpscalerCheckpoints(
    unet=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.unet",
            filename="model.safetensors",
            revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
        )
    ),
    clip_text_encoder=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
            filename="model.safetensors",
            revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
        )
    ),
    lda=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
            filename="model.safetensors",
            revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
        )
    ),
    controlnet_tile=Path(
        hf_hub_download(
            repo_id="refiners/controlnet.sd1_5.tile",
            filename="model.safetensors",
            revision="48ced6ff8bfa873a8976fa467c3629a240643387",
        )
    ),
    esrgan=Path(
        hf_hub_download(
            repo_id="philz1337x/upscaler",
            filename="4x-UltraSharp.pth",
            revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        )
    ),
    negative_embedding=Path(
        hf_hub_download(
            repo_id="philz1337x/embeddings",
            filename="JuggernautNegative-neg.pt",
            revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
        )
    ),
    negative_embedding_key="string_to_param.*",
    loras={
        "more_details": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="more_details.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
        "sdxl_render": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
    },
)

# initialize the enhancer, on the cpu
DEVICE_CPU = torch.device("cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=DEVICE_CPU, dtype=DTYPE)

# "move" the enhancer to the gpu, this is handled by Zero GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhancer.to(device=DEVICE, dtype=DTYPE)


@spaces.GPU
def process(
    input_image: Image.Image,
    prompt: str = "masterpiece, best quality, highres",
    negative_prompt: str = "worst quality, low quality, normal quality",
    seed: int = 42,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
) -> tuple[Image.Image, Image.Image]:
    manual_seed(seed)

    solver_type: type[Solver] = getattr(solvers, solver)

    enhanced_image = enhancer.upscale(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        upscale_factor=upscale_factor,
        controlnet_scale=controlnet_scale,
        controlnet_scale_decay=controlnet_decay,
        condition_scale=condition_scale,
        tile_size=(tile_height, tile_width),
        denoise_strength=denoise_strength,
        num_inference_steps=num_inference_steps,
        loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
        solver_type=solver_type,
    )

    return (input_image, enhanced_image)


with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            run_button = gr.ClearButton(components=None, value="Enhance Image")
        with gr.Column():
            output_slider = ImageSlider(label="Before / After")
            run_button.add(output_slider)

    with gr.Accordion("Advanced Options", open=False):
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="masterpiece, best quality, highres",
        )
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            placeholder="worst quality, low quality, normal quality",
        )
        seed = gr.Slider(
            minimum=0,
            maximum=10_000,
            value=42,
            step=1,
            label="Seed",
        )
        upscale_factor = gr.Slider(
            minimum=1,
            maximum=4,
            value=2,
            step=0.2,
            label="Upscale Factor",
        )
        controlnet_scale = gr.Slider(
            minimum=0,
            maximum=1.5,
            value=0.6,
            step=0.1,
            label="ControlNet Scale",
        )
        controlnet_decay = gr.Slider(
            minimum=0.5,
            maximum=1,
            value=1.0,
            step=0.025,
            label="ControlNet Scale Decay",
        )
        condition_scale = gr.Slider(
            minimum=2,
            maximum=20,
            value=6,
            step=1,
            label="Condition Scale",
        )
        tile_width = gr.Slider(
            minimum=64,
            maximum=200,
            value=112,
            step=1,
            label="Latent Tile Width",
        )
        tile_height = gr.Slider(
            minimum=64,
            maximum=200,
            value=144,
            step=1,
            label="Latent Tile Height",
        )
        denoise_strength = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.35,
            step=0.1,
            label="Denoise Strength",
        )
        num_inference_steps = gr.Slider(
            minimum=1,
            maximum=30,
            value=18,
            step=1,
            label="Number of Inference Steps",
        )
        solver = gr.Radio(
            choices=["DDIM", "DPMSolver"],
            value="DDIM",
            label="Solver",
        )

    run_button.click(
        fn=process,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            seed,
            upscale_factor,
            controlnet_scale,
            controlnet_decay,
            condition_scale,
            tile_width,
            tile_height,
            denoise_strength,
            num_inference_steps,
            solver,
        ],
        outputs=output_slider,
    )

    gr.Examples(
        examples=[
            "examples/kara-eads-L7EwHkq1B2s-unsplash.jpg",
            "examples/clarity_bird.webp",
            "examples/edgar-infocus-gJH8AqpiSEU-unsplash.jpg",
            "examples/jeremy-wallace-_XjW3oN8UOE-unsplash.jpg",
            "examples/karina-vorozheeva-rW-I87aPY5Y-unsplash.jpg",
            "examples/karographix-photography-hIaOPjYCEj4-unsplash.jpg",
            "examples/melissa-walker-horn-gtDYwUIr9Vg-unsplash.jpg",
            "examples/ryoji-iwata-X53e51WfjlE-unsplash.jpg",
            "examples/tadeusz-lakota-jggQZkITXng-unsplash.jpg",
        ],
        inputs=[input_image],
        outputs=output_slider,
        fn=process,
        cache_examples="lazy",
        run_on_click=False,
    )

demo.launch(share=False)
