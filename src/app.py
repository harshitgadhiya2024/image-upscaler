from pathlib import Path

import pillow_heif
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from refiners.fluxion.utils import manual_seed
from refiners.foundationals.latent_diffusion import Solver, solvers
from flask import (flash, Flask, redirect, render_template, request,
                   session, url_for, send_file, jsonify, send_from_directory)
from flask_cors import CORS
import uuid
import os

# create a flask app instance
app = Flask(__name__)

# Apply cors policy in our app instance
CORS(app)

# setup all config variable
app.config["enviroment"] = "prod"
app.config["SECRET_KEY"] = uuid.uuid4().hex

# handling our application secure type like http or https
secure_type = "http"

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# allow only that image file extension
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg', 'webp'}

def allowed_photos(filename):
    """
    checking file extension is correct or not

    :param filename: file name
    :return: True, False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints

pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

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


@app.route("/stylic/enhance_photo", methods=["GET", "POST"])
def enhance_photo():
    """
    In this route we can handling superadmin data
    :return: superadmin template
    """
    try:
        files_uploaded = []
        folder_person_name = request.form.get("folder_name")
        folder_image_store_path = f"static/uploads/{folder_person_name}"
        os.makedirs(folder_image_store_path, exist_ok=True)
        print("request are coming")
        file1 = request.files.get("garment_file")
        if file1 and file1.filename != "":
            exten = file1.filename.split(".")[-1]
            file1_path = os.path.join(folder_image_store_path, f"garment.{exten}")
            file1.save(file1_path)
            files_uploaded.append(file1_path.replace("\\", "/"))
        print("upload garment successfully")
        
        from PIL import Image
        human_image = Image.open(files_uploaded[0])
        enhanced_image = process(human_image, "", "", 42, 2, 0.6, 1.0, 6, 112, 144, 0.35, 18, "DDIM")
        output_folder_image_store_path = os.path.join(folder_image_store_path, "output_enhancer.jpg")
        enhanced_image[1].save(output_folder_image_store_path)
        response = {"status_code": 200, "data": {"output_file": f"http://139.84.138.54:80/download_photo/{folder_image_store_path.replace('/', '---')}***output_enhancer.jpg"}}
        return response

    except Exception as e:
        return {"message": "data is not present"}

# from PIL import Image
# human_image = Image.open("")
# process(human_image, "", "", 42, 2, 0.6, 1.0, 6, 112, 144, 0.35, 18, "DDIM")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
    
    
