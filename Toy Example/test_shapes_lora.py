import torch
import matplotlib.pyplot as plt

from diffusers import DDPMPipeline
from peft import PeftModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "google/ddpm-cifar10-32"
LORA_DIR = "models/ddpm-shapes-lora"


# 1. Load original pretrained CIFAR DDPM
pipe = DDPMPipeline.from_pretrained(BASE_MODEL)

base_unet = pipe.unet


# 2. Load your trained LoRA adapter
peft_unet = PeftModel.from_pretrained(
    base_unet,
    LORA_DIR
)

# Merge LoRA correction into the original UNet
unet = peft_unet.merge_and_unload()

unet = unet.to(DEVICE)

pipe.unet = unet
pipe = pipe.to(DEVICE)


# 3. Generate several images
images = []

for i in range(16):
    image = pipe(
        num_inference_steps=1000
    ).images[0]

    images.append(image)


# 4. Plot them
fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for ax, image in zip(axes.flat, images):
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "lora_shape_samples.png",
    dpi=200
)
plt.show()

print("Saved: lora_shape_samples.png")