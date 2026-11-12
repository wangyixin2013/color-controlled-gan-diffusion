import os
import glob
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import DDPMPipeline
from peft import LoraConfig, get_peft_model


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/shapes"
OUTPUT_DIR = "models/ddpm-shapes-lora"

BATCH_SIZE = 32

# Start small for the proof-of-concept.
# You can increase later.
EPOCHS = 3

LEARNING_RATE = 1e-4
LORA_RANK = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

class ShapeDataset(Dataset):

    def __init__(self, folder):

        self.files = sorted(
            glob.glob(os.path.join(folder, "*.png"))
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No PNG images found in {folder}"
            )

        self.transform = transforms.Compose([
            transforms.ToTensor(),

            # Convert [0,1] -> [-1,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

        print("Images:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = Image.open(
            self.files[idx]
        ).convert("RGB")

        return self.transform(image)


dataset = ShapeDataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,       # safest on Windows
    pin_memory=True,
)


# ============================================================
# LOAD PRETRAINED CIFAR-10 DDPM
# ============================================================

print("\nLoading pretrained DDPM...")

pipe = DDPMPipeline.from_pretrained(
    "google/ddpm-cifar10-32"
)

base_model = pipe.unet.to(device)
noise_scheduler = pipe.scheduler

print("Base model loaded.")


# ============================================================
# FREEZE ORIGINAL DDPM
# ============================================================

for param in base_model.parameters():
    param.requires_grad = False


# ============================================================
# LoRA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=LORA_RANK,

    lora_alpha=LORA_RANK,

    lora_dropout=0.0,

    bias="none",

    # Attention projections inside this UNet
    target_modules=[
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    ],
)


# ============================================================
# WRAP THE UNET WITH LoRA
# ============================================================

model = get_peft_model(
    base_model,
    lora_config
)

model = model.to(device)


# ============================================================
# CHECK TRAINABLE PARAMETERS
# ============================================================

print("\nLoRA parameter information:")

model.print_trainable_parameters()

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: "
    f"{total_params / 1e6:.3f} M"
)

print(
    f"Trainable parameters: "
    f"{trainable_params / 1e6:.3f} M"
)

print(
    f"Trainable percentage: "
    f"{100 * trainable_params / total_params:.4f}%"
)


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters = [
    p
    for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting LoRA fine-tuning...\n")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch_idx, clean_images in enumerate(loader):

        clean_images = clean_images.to(
            device,
            non_blocking=True
        )

        batch_size = clean_images.shape[0]


        # ----------------------------------------------------
        # 1. Generate random Gaussian noise
        # ----------------------------------------------------

        noise = torch.randn_like(
            clean_images
        )


        # ----------------------------------------------------
        # 2. Pick random diffusion timestep for every image
        # ----------------------------------------------------

        timesteps = torch.randint(
            low=0,
            high=noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=device,
        ).long()


        # ----------------------------------------------------
        # 3. Forward diffusion:
        #
        #       x0 -> xt
        #
        # ----------------------------------------------------

        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps
        )


        # ----------------------------------------------------
        # 4. Predict the noise
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):

            output = model(
                noisy_images,
                timesteps
            )

            noise_pred = output.sample

            loss = F.mse_loss(
                noise_pred.float(),
                noise.float()
            )


        # ----------------------------------------------------
        # 5. Backprop ONLY through LoRA parameters
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()


        total_loss += loss.item()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if batch_idx % 50 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Batch {batch_idx}/{len(loader)} "
                f"| Loss: {loss.item():.6f}"
            )


    avg_loss = (
        total_loss / len(loader)
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} complete"
    )

    print(
        f"Average loss: {avg_loss:.6f}\n"
    )


    # --------------------------------------------------------
    # SAVE AFTER EACH EPOCH
    # --------------------------------------------------------

    epoch_dir = os.path.join(
        OUTPUT_DIR,
        f"epoch-{epoch + 1}"
    )

    os.makedirs(
        epoch_dir,
        exist_ok=True
    )

    model.save_pretrained(
        epoch_dir
    )

    print(
        f"Saved adapter: {epoch_dir}\n"
    )


# ============================================================
# FINAL SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

model.save_pretrained(
    OUTPUT_DIR
)

print("====================================")
print("LoRA fine-tuning complete.")
print("Adapter saved to:")
print(OUTPUT_DIR)
print("====================================")