import torch
import torch.nn as nn
from torchvision.utils import save_image
from diffusers import DDPMPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 100
T_HANDOFF = 300
N_IMAGES = 16

GAN_PATH = "gan_xt/generator.pt"


# ============================================================
# GENERATOR -- must match training
# ============================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# LOAD GAN
# ============================================================

G = Generator(LATENT_DIM).to(DEVICE)

G.load_state_dict(
    torch.load(GAN_PATH, map_location=DEVICE)
)

G.eval()

print("GAN loaded.")


# ============================================================
# LOAD DDPM
# ============================================================

pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-cifar10-32"
).to(DEVICE)

unet = pipe.unet
scheduler = pipe.scheduler

unet.eval()

print("Diffusion loaded.")


# ============================================================
# GAN GENERATES x_300
# ============================================================

with torch.no_grad():

    z = torch.randn(
        N_IMAGES,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    sample = G(z)


print("GAN x300:")
print("mean:", sample.mean().item())
print("std :", sample.std().item())
print("min :", sample.min().item())
print("max :", sample.max().item())


# Save GAN output
gan_view = sample.clamp(-1, 1)
gan_view = (gan_view + 1) / 2

save_image(
    gan_view,
    "gan_x300.png",
    nrow=4
)


# ============================================================
# x_300 -> x_0
# ============================================================

scheduler.set_timesteps(1000)

with torch.no_grad():

    for t in range(300, -1, -1):

        t_batch = torch.full(
            (N_IMAGES,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        noise_pred = unet(
            sample,
            t_batch
        ).sample

        sample = scheduler.step(
            noise_pred,
            t,
            sample
        ).prev_sample


# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

final = sample.clamp(-1, 1)
final = (final + 1) / 2

save_image(
    final,
    "gan300_plus_diffusion.png",
    nrow=4
)

print("Saved:")
print("gan_x300.png")
print("gan300_plus_diffusion.png")