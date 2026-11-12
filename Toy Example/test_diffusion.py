import torch
from diffusers import DDPMPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-cifar10-32"
)

pipe = pipe.to(device)

print("Device:", device)
print("UNet parameters:",
      sum(p.numel() for p in pipe.unet.parameters()) / 1e6,
      "million")

image = pipe().images[0]

image.save("original_ddpm_sample.png")

print("Saved original_ddpm_sample.png")