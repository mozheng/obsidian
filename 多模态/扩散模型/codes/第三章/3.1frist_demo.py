# -*- coding: utf-8 -*-
import os
# 添加国内加速镜像
# 或者在终端用 export HF_ENDPOINT=https://hf-mirror.com 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

scheduler = DDPMScheduler.from_config("google/ddpm-cat-256") # 这就是个配置不是模型不用 to("CUDA")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50) # 设置50轮生成扩散

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise 

for t in scheduler.timesteps:
	with torch.no_grad():
		noisy_residual = model(input, t).sample
		prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
		input = prev_noisy_sample

# input的结果在[-1.,1.]范围内以小数形式表示
# 下面要将其映射到[0,255]的离散整数域中
image = (input / 2 + 0.5).clamp(0, 1) 
image = image.cpu().permute(0, 2, 3, 1).numpy()[0] 
image = Image.fromarray((image * 255).round().astype("uint8"))