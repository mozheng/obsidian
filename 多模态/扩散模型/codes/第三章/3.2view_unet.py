import os
# 添加国内加速镜像
# 或者在终端用 export HF_ENDPOINT=https://hf-mirror.com
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import DiffusionPipeline
import torch

cache_dir = "./models"

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                             cache_dir=cache_dir,
                                             torch_dtype=torch.float16)

# 获取Unet结构
unet = pipeline.unet
print(unet)
