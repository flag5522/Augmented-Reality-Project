from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: Optional[str]
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    seed: Optional[int]
    remove_background: bool


class GarmentGenerator:
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: str = "lllyasviel/sd-controlnet-openpose",
        device: str = "cpu",
    ) -> None:
        self.base_model_id = base_model_id
        self.controlnet_model_id = controlnet_model_id
        self.device = device

        # Lazy heavy imports
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
        import torch

        dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

        controlnet = ControlNetModel.from_pretrained(self.controlnet_model_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        self.torch = torch
        self.pipe = pipe.to(self.device)

    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0,
        remove_background: bool = True,
    ) -> Image.Image:
        control_image = control_image.convert("RGB").resize((width, height))

        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            image=control_image,
            width=width,
            height=height,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        image: Image.Image = result.images[0]

        if remove_background:
            from .bg import remove_background_rgba

            image = remove_background_rgba(image)
        else:
            if image.mode != "RGBA":
                image = image.convert("RGBA")

        return image

