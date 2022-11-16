import os
import uuid

import numpy as np
import PIL.Image

from nataili.util.save_sample import save_sample


class realesrgan:
    def __init__(self, model, device, output_dir, output_ext="jpg"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.output_images = []

    def __call__(self, input_image: PIL.Image = None, input_path: str = None):
        img = None
        if input_image is not None:
            img = input_image
            img_array = np.array(img)
        elif input_path is not None:
            img = PIL.Image.open(input_path)
            img_array = np.array(img)
        else:
            raise ValueError("No input image or path provided")
        output, _ = self.model.enhance(img_array)
        output_array = np.array(output)
        esrgan_image = PIL.Image.fromarray(output_array)
        if img.mode == "RGBA":
            self.output_ext = "png"
        filename = (
            os.path.basename(input_path).splitext(input_path)[0] if input_path is not None else str(uuid.uuid4())
        )
        filename = f"{filename}_esrgan"
        filename_with_ext = f"{filename}.{self.output_ext}"
        output_image = os.path.join(self.output_dir, filename_with_ext)
        save_sample(esrgan_image, filename, self.output_dir, self.output_ext)
        self.output_images.append(output_image)
        return
