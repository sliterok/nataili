import os
import uuid

import numpy as np
import PIL.Image

from nataili.util.save_sample import save_sample


class PostProcessor:
    def __init__(self, model, device, output_dir="./", output_ext="jpg", save_individual_images=True):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.output_images = []
        self.save_individual_images = save_individual_images
        self.set_filename_append()
    
    def set_filename_append(self):
        self.filename_append = 'post_processed'

    # Example __call__ function. Classes should override this with their own
    def __call__(self, input_image: PIL.Image = None, input_path: str = None, **kwargs):
        img, img_array = self.parse_image(input_image)
        output, _ = self.model.enhance(img_array)
        output_array = np.array(output)
        output_image = PIL.Image.fromarray(output_array)
        if self.save_individual_images:
            self.store_to_disk(input_path, output_image)
        self.output_images.append(output_image)

    def parse_image(self, input_image):
        img = None
        if input_image is not None:
            img = input_image
            img_array = np.array(img)
        elif input_path is not None:
            img = PIL.Image.open(input_path)
            img_array = np.array(img)
        else:
            raise ValueError("No input image or path provided")
        if img.mode == "RGBA":
            self.output_ext = "png"
        return (img, img_array)

    def store_to_disk(self, input_path, output_image):
        filename = (
            os.path.basename(input_path).splitext(input_path)[0] if input_path is not None else str(uuid.uuid4())
        )
        filename = f"{filename}_{self.filename_append}"
        # filename_with_ext = f"{filename}.{self.output_ext}"
        # output_image = os.path.join(self.output_dir, filename_with_ext)
        save_sample(output_image, filename, self.output_dir, self.output_ext)
