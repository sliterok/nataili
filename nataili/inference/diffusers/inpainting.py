import os
import re
import sys
import tqdm
from contextlib import contextmanager, nullcontext
import skimage
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
import PIL

from diffusers import StableDiffusionInpaintPipeline
import random
import uuid

class inpainting:
    def __init__(self, device, output_dir, save_extension='jpg',
    output_file_path=False, load_concepts=False, concepts_dir=None,
    verify_input=True, auto_cast=True):
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
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

    def resize_image(self, resize_mode, im, width, height):
        LANCZOS = (PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, 'Resampling') else PIL.Image.LANCZOS)
        if resize_mode == "resize":
            res = im.resize((width, height), resample=LANCZOS)
        elif resize_mode == "crop":
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        else:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio < src_ratio else im.width * height // im.height
            src_h = height if ratio >= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

        return res

    def generate(self, prompt: str, inpaint_img=None, inpaint_mask=None, mask_mode='mask', resize_mode='resize', noise_mode='seed',
      denoising_strength:float=0.8, ddim_steps=50, sampler_name='k_lms', n_iter=1, batch_size=1, cfg_scale=7.5, seed=None,
                height=512, width=512, save_individual_images: bool = True, save_grid: bool = True, ddim_eta:float = 0.0):

        seed = seed_to_int(seed)

        image_dict = {
            "seed": seed
        }

        inpaint_img = self.resize_image('resize', inpaint_img, width, height)
        inpaint_mask = self.resize_image('resize', inpaint_mask, width, height)

        # tbd: is this still needed?
        #torch_gc()

        # tbd: inpaint mask generation. for now we assume, that the original image and a mask image is passed to the function

        if self.load_concepts and self.concepts_dir is not None:
            prompt_tokens = re.findall('<([a-zA-Z0-9-]+)>', prompt)
            if prompt_tokens:
                self.process_prompt_tokens(prompt_tokens)

        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # tbd: model doesn't exist
        if self.verify_input and 1 == 0:
            try:
                check_prompt_length(self.model, prompt, self.comments)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        # tbd: these lines are currently only executed, if verify_input is True. Bug?
        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

        #precision_scope = torch.autocast if self.auto_cast else nullcontext
        #with torch.no_grad(), precision_scope("cuda"):

        for n in range(n_iter):
           print(f"Iteration: {n+1}/{n_iter}")
           prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
           seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

           if isinstance(prompts, tuple):
              prompts = list(prompts)

           # tbd: which is the right seed?
           mySeed = random.randint(0, 2**32)
           generator = torch.Generator(device=self.device).manual_seed(mySeed)

           x_samples = self.pipe(
              prompt=prompt,
              image=inpaint_img,
              mask_image=inpaint_mask,
              guidance_scale=cfg_scale,
              generator=generator,
              num_images_per_prompt=n_iter
           ).images

           for x_sample in x_samples:
              # tbd: which is the right filename?
              #sanitized_prompt = slugify(prompts[i])
              #full_path = os.path.join(os.getcwd(), sample_path)
              #filename = f"{base_count:05}-{ddim_steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[:200-len(full_path)]

              image_dict['image'] = x_sample
              self.images.append(image_dict)

              # tbd: which is the right path name?
              if save_individual_images and 1 == 0:
                 path = os.path.join(sample_path, filename + '.' + self.save_extension)
                 success = save_sample(image, filename, sample_path_i, self.save_extension)

                 if success:
                    if self.output_file_path:
                       self.output_images.append(path)
                    else:
                       self.output_images.append(image)
                 else:
                    return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = f'''
                '''

        for comment in self.comments:
            self.info += "\n\n" + comment

        # tbd: is this still needed?
        #torch_gc()

        del generator

        return
