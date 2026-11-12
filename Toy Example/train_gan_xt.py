import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from diffusers import DDPMPipeline


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 128
EPOCHS = 50
LR = 2e-4

LATENT_DIM = 100
T_HANDOFF = 300

# Green-control settings
MARGIN = 0.2
LAMBDA_GREEN = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "gan_xt_green"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# ============================================================
# CIFAR-10
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    # [0,1] -> [-1,1]
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("CIFAR images:", len(dataset))


# ============================================================
# LOAD DIFFUSION SCHEDULER
# ============================================================

pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-cifar10-32"
)

scheduler = pipe.scheduler

# Only scheduler is needed during GAN training
del pipe


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()

        self.net = nn.Sequential(

            # z: B x 100 x 1 x 1
            nn.ConvTranspose2d(
                latent_dim,
                256,
                4,
                1,
                0,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                64,
                3,
                4,
                2,
                1,
                bias=False
            )

            # No Tanh because x_300 can exceed [-1,1]
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            # 32x32 -> 16x16
            nn.Conv2d(
                3,
                64,
                4,
                2,
                1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(
                64,
                128,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(
                128,
                256,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1
            nn.Conv2d(
                256,
                1,
                4,
                1,
                0,
                bias=False
            )
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ============================================================
# CREATE MODELS
# ============================================================

G = Generator(LATENT_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_G = torch.optim.Adam(
    G.parameters(),
    lr=LR,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    D.parameters(),
    lr=LR,
    betas=(0.5, 0.999)
)


# ============================================================
# FIXED RANDOM INPUT FOR MONITORING
# ============================================================

fixed_z = torch.randn(
    64,
    LATENT_DIM,
    1,
    1,
    device=DEVICE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    total_D = 0.0
    total_G = 0.0
    total_realism = 0.0
    total_green = 0.0


    for batch_idx, (clean_images, _) in enumerate(loader):

        clean_images = clean_images.to(
            DEVICE,
            non_blocking=True
        )

        batch_size = clean_images.size(0)


        # ====================================================
        # CREATE REAL x_300
        # ====================================================

        noise = torch.randn_like(clean_images)

        t = torch.full(
            (batch_size,),
            T_HANDOFF,
            device=DEVICE,
            dtype=torch.long
        )

        real_xt = scheduler.add_noise(
            clean_images,
            noise,
            t
        )


        # ====================================================
        # TRAIN DISCRIMINATOR
        # ====================================================

        optimizer_D.zero_grad()


        # Real x_300
        real_logits = D(real_xt)

        real_labels = torch.ones_like(
            real_logits
        )

        loss_real = criterion(
            real_logits,
            real_labels
        )


        # Fake x_300
        z = torch.randn(
            batch_size,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        fake_xt = G(z)

        fake_logits = D(
            fake_xt.detach()
        )

        fake_labels = torch.zeros_like(
            fake_logits
        )

        loss_fake = criterion(
            fake_logits,
            fake_labels
        )


        # Total discriminator loss
        loss_D = (
            loss_real +
            loss_fake
        ) / 2

        loss_D.backward()

        optimizer_D.step()


        # ====================================================
        # TRAIN GENERATOR
        # ====================================================

        optimizer_G.zero_grad()


        # ----------------------------------------------------
        # REALISM LOSS
        # ----------------------------------------------------

        fake_logits = D(fake_xt)

        generator_targets = torch.ones_like(
            fake_logits
        )

        loss_realism = criterion(
            fake_logits,
            generator_targets
        )


        # ----------------------------------------------------
        # GREEN LOSS
        # ----------------------------------------------------

        rgb = fake_xt.mean(
            dim=(2, 3)
        )

        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]


        # Goal:
        #
        # green - red >= MARGIN
        # green - blue >= MARGIN

        loss_green = (

            torch.relu(
                MARGIN - (green - red)
            )

            +

            torch.relu(
                MARGIN - (green - blue)
            )

        ).mean()


        # ----------------------------------------------------
        # TOTAL GENERATOR LOSS
        # ----------------------------------------------------

        loss_G = (
            loss_realism
            +
            LAMBDA_GREEN * loss_green
        )


        loss_G.backward()

        optimizer_G.step()


        # ====================================================
        # RECORD LOSSES
        # ====================================================

        total_D += loss_D.item()
        total_G += loss_G.item()

        total_realism += loss_realism.item()
        total_green += loss_green.item()


        # ====================================================
        # PRINT EVERY 100 BATCHES
        # ====================================================

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"| Batch {batch_idx}/{len(loader)} "
                f"| D: {loss_D.item():.4f} "
                f"| Realism: {loss_realism.item():.4f} "
                f"| Green: {loss_green.item():.4f} "
                f"| G Total: {loss_G.item():.4f}"
            )


    # ========================================================
    # END OF EPOCH
    # ========================================================

    G.eval()

    with torch.no_grad():

        # Same latent inputs every epoch
        samples = G(fixed_z)


        # ====================================================
        # RGB CHECK
        # ====================================================

        rgb_check = samples.mean(
            dim=(0, 2, 3)
        )

        r_check, g_check, b_check = (
            rgb_check
            .detach()
            .cpu()
            .tolist()
        )


        # ====================================================
        # SAVE SAMPLE IMAGE
        # ====================================================

        visual = samples.clamp(
            -1,
            1
        )

        visual = (
            visual + 1
        ) / 2

        save_image(
            visual,
            os.path.join(
                OUTPUT_DIR,
                f"xt_epoch_{epoch+1}.png"
            ),
            nrow=8
        )


    G.train()


    # ========================================================
    # SAVE WEIGHTS AFTER EVERY EPOCH
    # ========================================================

    generator_path = os.path.join(
        OUTPUT_DIR,
        f"generator_epoch_{epoch+1}.pt"
    )

    discriminator_path = os.path.join(
        OUTPUT_DIR,
        f"discriminator_epoch_{epoch+1}.pt"
    )


    torch.save(
        G.state_dict(),
        generator_path
    )

    torch.save(
        D.state_dict(),
        discriminator_path
    )


    print(
        f"\nSaved generator weights: "
        f"generator_epoch_{epoch+1}.pt"
    )

    print(
        f"Saved discriminator weights: "
        f"discriminator_epoch_{epoch+1}.pt"
    )


    # ========================================================
    # PRINT RGB
    # ========================================================

    print(
        f"\nEpoch {epoch+1} RGB "
        f"| R: {r_check:.6f} "
        f"| G: {g_check:.6f} "
        f"| B: {b_check:.6f}"
    )

    print(
        f"Color differences "
        f"| G-R: {g_check-r_check:.6f} "
        f"| G-B: {g_check-b_check:.6f}"
    )


    # ========================================================
    # PRINT AVERAGE LOSSES
    # ========================================================

    print(
        f"Epoch {epoch+1} averages "
        f"| D: {total_D / len(loader):.4f} "
        f"| Realism: {total_realism / len(loader):.4f} "
        f"| Green: {total_green / len(loader):.4f} "
        f"| G Total: {total_G / len(loader):.4f}\n"
    )


print("GAN training complete.")