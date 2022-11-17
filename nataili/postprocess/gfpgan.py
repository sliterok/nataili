from nataili.postprocessor import *


class gfpgan(PostProcessor):
    def __call__(self, input_image: PIL.Image = None, input_path: str = None, **kwargs):
        strength = kwargs.get("strength", 1.0)
        img, img_array = self.parse_image(input_image)
        _, _, output = self.model.enhance(img_array, weight=strength)
        output_array = np.array(output)
        output_image = PIL.Image.fromarray(output_array)
        self.output_images.append(output_image)
        if self.save_individual_images:
            self.store_to_disk(input_path, output_image)
