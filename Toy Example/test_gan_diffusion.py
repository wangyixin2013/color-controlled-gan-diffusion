import time
import torch
import torch.nn as nn
from torchvision.utils import save_image
from diffusers import DDPMPipeline
from torchvision.transforms.functional import pil_to_tensor


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 100

# Green GAN was trained at timestep 300
T_HANDOFF = 300

N_IMAGES = 16

# Green GAN checkpoint
GAN_PATH = "gan_xt_green/generator_epoch_17.pt"

print("Device:", DEVICE)


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()

        self.net = nn.Sequential(

            nn.ConvTranspose2d(
                latent_dim, 256,
                4, 1, 0,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                256, 128,
                4, 2, 1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128, 64,
                4, 2, 1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 3,
                4, 2, 1,
                bias=False
            )
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# LOAD GREEN GAN
# ============================================================

G = Generator(LATENT_DIM).to(DEVICE)

G.load_state_dict(
    torch.load(
        GAN_PATH,
        map_location=DEVICE
    )
)

G.eval()

print("Green GAN loaded.")


# ============================================================
# LOAD PRETRAINED DIFFUSION MODEL
# ============================================================

pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-cifar10-32"
)

pipe = pipe.to(DEVICE)

unet = pipe.unet
scheduler = pipe.scheduler

unet.eval()

print("Diffusion loaded.")


# ============================================================
# METHOD A:
# PURE DIFFUSION
# ============================================================

print("\nRunning pure diffusion...")

if DEVICE == "cuda":
    torch.cuda.synchronize()
start = time.time()

with torch.no_grad():

    diffusion_output = pipe(
        batch_size=N_IMAGES,
        num_inference_steps=1000
    ).images

if DEVICE == "cuda":
    torch.cuda.synchronize()

diffusion_time = time.time() - start

print(
    f"Pure diffusion time: "
    f"{diffusion_time:.2f} seconds"
)


# Convert PIL images to tensors
diffusion_tensor = torch.stack([
    pil_to_tensor(img).float() / 255.0
    for img in diffusion_output
])

save_image(
    diffusion_tensor,
    "diffusion_only.png",
    nrow=4
)


# ============================================================
# METHOD B:
# GREEN GAN -> x_300 -> DIFFUSION
# ============================================================

print("\nRunning GREEN GAN + diffusion...")

if DEVICE == "cuda":
    torch.cuda.synchronize()
start = time.time()


# --------------------------------------------
# Green GAN generates x_300
# --------------------------------------------

with torch.no_grad():

    z = torch.randn(
        N_IMAGES,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    sample = G(z)


# ============================================================
# SAVE RAW GREEN GAN x_300
# ============================================================

raw_gan = sample.clamp(-1, 1)
raw_gan = (raw_gan + 1) / 2

save_image(
    raw_gan,
    "green_gan_x300.png",
    nrow=4
)


# ============================================================
# PRINT RGB BEFORE DIFFUSION
# ============================================================

rgb_before = sample.mean(
    dim=(0, 2, 3)
)

print("\nAverage RGB at x300:")
print("Red:  ", rgb_before[0].item())
print("Green:", rgb_before[1].item())
print("Blue: ", rgb_before[2].item())


# --------------------------------------------
# diffusion x_300 -> x_0
# --------------------------------------------

scheduler.set_timesteps(1000)

with torch.no_grad():

    for t in range(
        T_HANDOFF,
        -1,
        -1
    ):

        timestep = torch.full(
            (N_IMAGES,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        noise_pred = unet(
            sample,
            timestep
        ).sample

        sample = scheduler.step(
            noise_pred,
            t,
            sample
        ).prev_sample


if DEVICE == "cuda":
    torch.cuda.synchronize()

hybrid_time = time.time() - start

print(
    f"\nGreen GAN + diffusion time: "
    f"{hybrid_time:.2f} seconds"
)


# ============================================================
# RGB AFTER DIFFUSION
# ============================================================

rgb_after = sample.mean(
    dim=(0, 2, 3)
)

print("\nAverage RGB after diffusion:")
print("Red:  ", rgb_after[0].item())
print("Green:", rgb_after[1].item())
print("Blue: ", rgb_after[2].item())


# ============================================================
# SAVE FINAL RESULT
# ============================================================

hybrid = sample.clamp(-1, 1)

hybrid = (
    hybrid + 1
) / 2

save_image(
    hybrid,
    "green_gan_plus_diffusion.png",
    nrow=4
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("RESULTS")
print("==============================")

print(
    f"Pure diffusion:          "
    f"{diffusion_time:.2f} sec"
)

print(
    f"Green GAN + diffusion:   "
    f"{hybrid_time:.2f} sec"
)

print(
    f"Speed-up:                "
    f"{diffusion_time / hybrid_time:.2f}x"
)

print("\nSaved:")
print("diffusion_only.png")
print("green_gan_x300.png")
print("green_gan_plus_diffusion.png")
