from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from refiners.foundationals.clip.concepts import ConceptExtender
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
    MultiUpscaler,
    UpscalerCheckpoints,
)

from esrgan_model import UpscalerESRGAN


@dataclass(kw_only=True)
class ESRGANUpscalerCheckpoints(UpscalerCheckpoints):
    esrgan: Path | None = None


class ESRGANUpscaler(MultiUpscaler):
    def __init__(
        self,
        checkpoints: ESRGANUpscalerCheckpoints,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(checkpoints=checkpoints, device=device, dtype=dtype)
        self.esrgan = self.load_esrgan(checkpoints.esrgan)

    def to(self, device: torch.device, dtype: torch.dtype):
        self.esrgan.to(device=device, dtype=dtype)
        self.sd = self.sd.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def load_esrgan(self, path: Path | None) -> UpscalerESRGAN | None:
        if path is None:
            return None
        return UpscalerESRGAN(path, device=self.device, dtype=self.dtype)

    def load_negative_embedding(self, path: Path | None, key: str | None) -> str:
        if path is None:
            return ""

        embeddings: torch.Tensor | dict[str, Any] = torch.load(  # type: ignore
            path, weights_only=True, map_location=self.device
        )

        if isinstance(embeddings, dict):
            assert (
                key is not None
            ), "Key must be provided to access the negative embedding."
            key_sequence = key.split(".")
            for key in key_sequence:
                assert (
                    key in embeddings
                ), f"Key {key} not found in the negative embedding dictionary. Available keys: {list(embeddings.keys())}"
                embeddings = embeddings[key]

        assert isinstance(
            embeddings, torch.Tensor
        ), f"The negative embedding must be a tensor, found {type(embeddings)}."
        assert (
            embeddings.ndim == 2
        ), f"The negative embedding must be a 2D tensor, found {embeddings.ndim}D tensor."

        extender = ConceptExtender(self.sd.clip_text_encoder)
        negative_embedding_token = ", "
        for i, embedding in enumerate(embeddings):
            embedding = embedding.to(device=self.device, dtype=self.dtype)
            extender.add_concept(token=f"<{i}>", embedding=embedding)
            negative_embedding_token += f"<{i}> "
        extender.inject()

        return negative_embedding_token

    def pre_upscale(
        self,
        image: Image.Image,
        upscale_factor: float,
        use_esrgan: bool = True,
        use_esrgan_tiling: bool = True,
        **_: Any,
    ) -> Image.Image:
        if self.esrgan is None or not use_esrgan:
            return super().pre_upscale(image=image, upscale_factor=upscale_factor)

        width, height = image.size

        if use_esrgan_tiling:
            image = self.esrgan.upscale_with_tiling(image)
        else:
            image = self.esrgan.upscale_without_tiling(image)

        return image.resize(
            size=(
                int(width * upscale_factor),
                int(height * upscale_factor),
            ),
            resample=Image.LANCZOS,
        )
