This repository includes a modified version of the nataili stable-diffusion server to show an inpainting prototype.

## Installation
- ```git clone https://github.com/blueturtleai/nataili```
- ```cd nataili```

### Huggingface access token
- Register on Huggingface
- Go here https://huggingface.co/runwayml/stable-diffusion-inpainting
- Agree to the terms
- Create a token via Settings/Access Tokens
- ```export HUGGING_FACE_HUB_TOKEN=<your huggingface token>```

## Run inpainting test
```./runtime.sh python test_inpainting.py```

- "inpaint_original.png" and "inpaint_mask.png" together with the prompt "a mecha robot sitting on a bench" are used to create a new image, where the dog will be replaced by a robot. Outputfile is "robot_sitting_on_a_bench.png".

## Changes
It was necessary to make two changes in requirements.txt to make this working:

- diffusers 0.4.1 -> 0.6.0. 0.6.0 is needed for making the 1.5 inpainting model working.
- transformers 4.19.2 -> latest version. Most likely, this will lead to a conflict because the line for version 4.19.2 was commented with "don't change". The newest version is needed, because otherwise errors occurred during generation.

## Comments
- This is only a prototype which shows, how inpainting could be added to the nataili server in general. In "inpainting.py" I had to remove many code parts I copied over from "img2img.py" to make it working. Due to time restrictions, I were not able to study the existing code and modify it accordingly. Due to lack of a local GPU, I also had to rent an AWS server to develop this prototype.

- I would recommend, to integrate it in that way, that the client only sends one image which includes an Alpha channel. The Alpha channel holds the information, which part should be inpainted. The mask image file would be generated on the server side based on the Alpha channel. Imo that way more clients would be able to use inpainting, because they wouldn't need to create the mask image themselves.
