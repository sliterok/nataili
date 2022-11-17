import PIL

from nataili.model_manager import ModelManager
from nataili.postprocess.gfpgan import gfpgan
from nataili.util.logger import logger

MODEL = "GFPGAN"

def fix_face(image, model_manager):
    if MODEL not in model_manager.available_models:
        logger.warning(f"{MODEL} not available")
        logger.init(f"{MODEL}", status="Downloading")
        model_manager.download_model(MODEL)
        logger.init_ok(f"{MODEL}", status="Downloaded")

    if not model_manager.is_model_loaded(MODEL):
        logger.init(f"{MODEL}", status="Loading")
        success = model_manager.load_model(MODEL)
        logger.init_ok(f"{MODEL}", status="Success")

    facefixer = gfpgan(
        model_manager.loaded_models[MODEL]["model"],
        model_manager.loaded_models[MODEL]["device"],
        save_individual_images = False,
    )

    results = facefixer(input_image=image, strength=1.0)
    return facefixer.output_images[0]
