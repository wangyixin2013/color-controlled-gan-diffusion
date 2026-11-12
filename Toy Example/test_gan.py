import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from diffusers import DDPMPipeline


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 100
T_HANDOFF = 100
N_IMAGES = 64

GAN_PATH = "gan_xt_green/generator.pt"

print("Device:", DEVICE)


# ============================================================
# GENERATOR
# Must be identical to the one used during training
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
# LOAD TRAINED GAN
# ============================================================

G = Generator(LATENT_DIM).to(DEVICE)

G.load_state_dict(
    torch.load(
        GAN_PATH,
        map_location=DEVICE
    )
)

G.eval()

print("Generator loaded.")


# ============================================================
# GENERATE FAKE x_100
# ============================================================

with torch.no_grad():

    z = torch.randn(
        N_IMAGES,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    fake_xt = G(z)


print("\nFAKE x_100")
print("min:", fake_xt.min().item())
print("max:", fake_xt.max().item())
print("mean:", fake_xt.mean().item())
print("std:", fake_xt.std().item())


# Save fake samples for visualization
fake_visual = fake_xt.clamp(-1, 1)
fake_visual = (fake_visual + 1) / 2

save_image(
    fake_visual,
    "gan_fake_x100.png",
    nrow=8
)


# ============================================================
# CREATE REAL CIFAR x_100 FOR COMPARISON
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=N_IMAGES,
    shuffle=True
)

clean_images, _ = next(iter(loader))
clean_images = clean_images.to(DEVICE)


# Get DDPM scheduler
pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-cifar10-32"
)

scheduler = pipe.scheduler

noise = torch.randn_like(clean_images)

t = torch.full(
    (N_IMAGES,),
    T_HANDOFF,
    device=DEVICE,
    dtype=torch.long
)

real_xt = scheduler.add_noise(
    clean_images,
    noise,
    t
)


print("\nREAL x_100")
print("min:", real_xt.min().item())
print("max:", real_xt.max().item())
print("mean:", real_xt.mean().item())
print("std:", real_xt.std().item())


real_visual = real_xt.clamp(-1, 1)
real_visual = (real_visual + 1) / 2

save_image(
    real_visual,
    "real_x100.png",
    nrow=8
)


print("\nSaved:")
print("gan_fake_x100.png")
print("real_x100.png")
