import os
import re
import sys
from contextlib import contextmanager, nullcontext

import numpy as np
import PIL
import torch
from einops import rearrange
from nataili.util.cache import torch_gc
from nataili.util.check_prompt_length import check_prompt_length
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.image_grid import image_grid
from nataili.util.load_learned_embed_in_clip import load_learned_embed_in_clip
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int
from slugify import slugify
from diffusers import StableDiffusionInpaintPipeline
import random

class inpainting:
    def __init__(self, output_dir, device, save_extension='jpg', output_file_path=False, load_concepts=False, concepts_dir=None,
    verify_input=True, auto_cast=True):
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.device = device
        self.comments = []
        self.output_images = []
        self.info = ''
        self.stats = ''
        self.images = []

        model_path = "runwayml/stable-diffusion-inpainting"

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
           model_path,
           revision="fp16",
           torch_dtype=torch.float16,
           use_auth_token=True
        ).to(self.device)

    def generate(self, prompt: str, inpaint_img=None, inpaint_mask=None, mask_mode='mask', resize_mode='resize', noise_mode='seed', denoising_strength:float=0.8, ddim_steps=50, sampler_name='k_lms', n_iter=1, batch_size=1, cfg_scale=7.5, seed=None, height=512, width=512, save_individual_images: bool = True, save_grid: bool = True, ddim_eta:float = 0.0):
        guidance_scale = 7.5
        num_samples = 1

        seed = random.randint(0, 2**32)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        x_samples = self.pipe(
           prompt=prompt,
           image=inpaint_img,
           mask_image=inpaint_mask,
           guidance_scale=guidance_scale,
           generator=generator,
           num_images_per_prompt=num_samples,
        ).images

        return x_samples
