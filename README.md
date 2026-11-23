# Color Control of the GAN-Diffusion Model

**CS680 final project · Yixin Wang · University of Waterloo · Work in progress**

Can a GAN introduce color control into a noisy intermediate image, and can a frozen diffusion model preserve that control while finishing image generation?

This repository contains the [project proposal (PDF)](docs/proposal.pdf), pilot code, saved pilot checkpoints, and exploratory sample images. The current implementation uses a fixed green-color penalty. Conditional control with requested chroma targets and full evaluation are planned work.

## Approach

The pilot trains a small convolutional GAN on noisy CIFAR-10 images at a chosen diffusion timestep. During sampling, the GAN generates an intermediate state, and the pretrained `google/ddpm-cifar10-32` denoiser completes the remaining reverse steps without additional guidance or updates to its weights.

`Random latent vector → GAN → noisy image at timestep t → frozen DDPM → final image`

The current green-control experiment uses `t = 300`. Its generator loss combines an adversarial loss with a penalty encouraging the mean green channel to exceed the mean red and blue channels by a margin of `0.2`, with color-loss weight `1.0`. This penalty is applied to the GAN output; preserving it through denoising is the research question.

The general GAN–diffusion hybrid is established prior work. This project's focus is color retention through unguided denoising.

## Current progress

| Component | Status |
| --- | --- |
| CIFAR-10 GAN training on noisy intermediate states | Pilot implemented |
| Fixed green-color penalty at `t = 300` | Pilot implemented |
| GAN output followed by frozen DDPM sampling | Pilot implemented |
| Pure DDPM comparison, sample grids, and timing output | Implemented; exploratory comparisons |
| LoRA adaptation to synthetic colored shapes | Separate exploratory experiment included |
| Conditional GAN with a requested average-chroma target | Planned |
| Controlled comparison across handoff timesteps and random seeds | Planned |
| Directly guided diffusion baseline | Planned |
| Chroma error, FID, and matched-control evaluation | Planned |

The saved images demonstrate pilot behavior. They do not establish a quantitative advantage in image quality, color accuracy, or speed. Existing scripts do not fix random seeds, and older checkpoints have incomplete training provenance. Timing comparisons also use different numbers of reverse steps.

## Pilot samples

| Pretrained DDPM | Green GAN output at `t = 300` | After unguided DDPM denoising |
| --- | --- | --- |
| ![Exploratory pretrained DDPM samples](Toy%20Example/diffusion_only.png) | ![Green GAN intermediate samples](Toy%20Example/green_gan_x300.png) | ![Green GAN samples after diffusion](Toy%20Example/green_gan_plus_diffusion.png) |

These are saved exploratory outputs, not a controlled benchmark.

## Setup

Use Python 3.12. The dependencies in [requirements.txt](requirements.txt) pin the environment recorded on the original computer; they are not a record of every earlier training run. Install a suitable PyTorch CUDA build separately if needed for your GPU.

```sh
git clone https://github.com/wangyixin2013/cs680-color-controlled-gan-diffusion.git
cd cs680-color-controlled-gan-diffusion
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux / macOS
source .venv/bin/activate
```

Then install dependencies and enter the script directory:

```sh
python -m pip install -r requirements.txt
cd "Toy Example"
```

Run all experiment commands from `Toy Example`, since script paths are relative to that directory. Scripts use CUDA when available and otherwise use the CPU. CPU sampling can be slow. The first run requires internet access to download the base DDPM from Hugging Face.

## Run the saved pilots

Compare full DDPM sampling with the green GAN followed by diffusion:

```sh
python test_gan_diffusion.py
```

This loads the included `gan_xt_green/generator_epoch_17.pt`, generates 16 images per method, prints timing and mean RGB values, and writes `diffusion_only.png`, `green_gan_x300.png`, and `green_gan_plus_diffusion.png`. Full DDPM uses 1,000 reverse steps; the hybrid runs from timestep 300 through 0.

Other sampling scripts:

| Command | Purpose |
| --- | --- |
| `python test_diffusion.py` | Generate one image from the pretrained DDPM |
| `python test_gan_300.py` | Run the saved GAN in `gan_xt/generator.pt` with a handoff at `t = 300` |
| `python test_shapes_lora.py` | Load the included LoRA adapter and generate a grid of shape samples |

Sampling overwrites images with matching output names. The files named `test_*.py` are standalone sampling or inspection scripts, not an automated test suite.

`test_gan.py` is an older inspection script configured at `t = 100`; it needs a local CIFAR-10 download and loads `gan_xt_green/generator.pt`. Its timestep differs from the current green-GAN training configuration. Check checkpoint provenance and timestep before using it for comparisons. The current training script does not reproduce every older included checkpoint.

## Training

Train the green GAN on noisy CIFAR-10 images:

```sh
python train_gan_xt.py
```

The current settings use 50 epochs, batch size 128, a 100-dimensional latent input, and handoff timestep 300. CIFAR-10 is downloaded to `data/`. Epoch checkpoints and preview images are saved to `gan_xt_green/`. A new run can overwrite the included epoch 17 checkpoint. Training does not update `generator.pt` automatically; change `GAN_PATH` in the sampling script to select a different checkpoint.

For the separate synthetic-shapes LoRA experiment:

```sh
python make_shapes.py
python view_shapes.py
python train_shapes_ddpm.py
python test_shapes_lora.py
```

This creates 10,000 colored shape images, trains rank-8 LoRA attention adapters for three epochs, and saves the final adapter to `models/ddpm-shapes-lora/`. Training overwrites adapters at the same output paths. This experiment does not implement the proposed conditional GAN.

## Repository contents

```text
docs/proposal.pdf                 Project proposal
requirements.txt                  Recorded dependency versions
Toy Example/
  train_gan_xt.py                  Fixed-green GAN training
  test_gan_diffusion.py            DDPM vs. green GAN + DDPM pilot
  test_gan_300.py                  Saved GAN + DDPM sampling
  test_gan.py                      Older intermediate-state inspection
  test_diffusion.py                Pretrained DDPM sampling
  make_shapes.py                   Synthetic shape dataset generation
  view_shapes.py                   Dataset preview
  train_shapes_ddpm.py             Exploratory LoRA training
  test_shapes_lora.py              LoRA sampling
  gan_xt/                         Earlier saved GAN weights
  gan_xt_green/                   Saved green-GAN weights
  models/ddpm-shapes-lora/         Saved LoRA adapter
  *.png                           Exploratory output images
```

Downloaded datasets, generated training data, repeated epoch checkpoints, environment files, and caches are excluded from Git. The base DDPM is downloaded separately.
