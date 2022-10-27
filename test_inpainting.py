from nataili.inference.diffusers.inpainting import inpainting
from PIL import Image

original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")

generator = inpainting("output_dir", "cuda")
images = generator.generate("a mecha robot sitting on a bench", original, mask)
image = images[0]
image.save("robot_sitting_on_a_bench.png", format="PNG")
