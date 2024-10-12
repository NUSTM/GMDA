import json
import os.path

import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline



torch.manual_seed(42)

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path
)

# breakpoint()
pipe = pipe.to(device)

generator = torch.Generator("cuda").manual_seed(42)

img_dir = "image_path"

# for dataset in ["10FMNERG", "50FMNERG", "FMNERG", "10GMNER", "50GMNER", "GMNER"]:
for dataset in ["T15"]: # , 
    json_path = f"filtered_InstructBLIP_out/{dataset}/aug_train_unique.json"
    output_image_dir = f"./diff_image/{dataset}"
    # for json_path, output_image_dir in zip(json_paths, output_image_dirs):

    os.makedirs(output_image_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        aug_text = json.load(f)

    for item in aug_text:
        img_id = item["img_id"]
        sentence = item["sentence"]
        sentence = sentence.split()
        for i in range(len(sentence)):
            if "http" in sentence[i]:
                sentence[i] = ""

        sentence = ' '.join(sentence)
        if os.path.exists(os.path.join(output_image_dir, img_id + "_aug.jpg")):
            continue

        img_path = os.path.join(img_dir, img_id + ".jpg")
        prompt = f"A photo of {sentence}"

        try:
            init_image = Image.open(img_path).convert("RGB")

            images = pipe( 
                prompt=prompt,  
                image=init_image,
                strength=0.8,
                guidance_scale=10,
                generator=generator,
            ).images
        except:
            images = pipe(prompt, guidance_scale=7.5, generator=generator).images

        images[0].save(os.path.join(output_image_dir, img_id + "_aug.jpg"))

