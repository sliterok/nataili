import time

from PIL import Image

from nataili.inference.diffusers.inpainting import inpainting
from nataili.model_manager import ModelManager
from nataili.util.cache import torch_gc
from nataili.util.logger import logger

original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")

mm = ModelManager()

mm.init()
logger.debug("Available dependencies:")
for dependency in mm.available_dependencies:
    logger.debug(dependency)

logger.debug("Available models:")
for model in mm.available_models:
    logger.debug(model)

model = "stable_diffusion_inpainting"

tic = time.time()
logger.init(f"Model: {model}", status="Loading")

success = mm.load_model(model)

toc = time.time()
logger.init_ok(f"Loading {model}: Took {toc-tic} seconds", status=success)
torch_gc()

generator = inpainting(mm.loaded_models[model]["model"], "cuda", "output_dir")
generator.generate("a mecha robot sitting on a bench", sampler="k_euler_a", inpaint_img=original, inpaint_mask=mask)
