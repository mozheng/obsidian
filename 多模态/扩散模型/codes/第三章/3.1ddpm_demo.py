# -*- coding: utf-8 -*-
import os
# 添加国内加速镜像
# 或者在终端用 export HF_ENDPOINT=https://hf-mirror.com 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)

# diffusers 库与 transformers 库都适配CPU/GPU模型，如需GPU推理可打开下面注释
# pipeline.to("cuda") 

image = pipeline("An image of a squirrel in Picasso style").images[0]
image.save("squirrel_picasso.png") 