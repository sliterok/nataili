from nataili.postprocessor import *

class realesrgan(PostProcessor):

    def __call__(self, input_image: PIL.Image = None, input_path: str = None, **kwargs):
        img = None
        img, img_array = self.parse_image(input_image)
        output, _ = self.model.enhance(img_array)
        output_array = np.array(output)
        output_image = PIL.Image.fromarray(output_array)
        self.output_images.append(output_image)
        if self.save_individual_images:
            self.store_to_disk(input_path, output_image)
