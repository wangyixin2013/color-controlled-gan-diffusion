---
base_model: google/ddpm-cifar10-32
library_name: peft
tags:
  - lora
  - diffusion
  - image-generation
---

# Synthetic-shapes LoRA pilot

Saved LoRA adapter from the separate synthetic colored-shapes experiment in this CS680 project. It adapts the attention projections of the pretrained CIFAR-10 DDPM UNet.

The accompanying training script uses rank 8, LoRA alpha 8, no dropout, learning rate `1e-4`, and three epochs on 10,000 generated 32 × 32 images of colored circles, squares, and triangles. These describe the current script settings; complete logs for the saved adapter are not included.

From `Toy Example`, run `python test_shapes_lora.py` to load this adapter into `google/ddpm-cifar10-32`, merge its weights, and save a sample grid. The base model is downloaded separately.

This exploratory adapter is separate from the GAN color-control pilot and has not been quantitatively evaluated.
