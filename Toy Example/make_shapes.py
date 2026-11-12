from PIL import Image, ImageDraw
import random
import os

SIZE = 32
N_IMAGES = 10000

OUT_DIR = "data/shapes"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 200, 70),
    "blue": (40, 80, 220),
}

SHAPES = ["circle", "square", "triangle"]

for i in range(N_IMAGES):

    # Light background
    img = Image.new("RGB", (SIZE, SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    shape = random.choice(SHAPES)
    color_name = random.choice(list(COLORS))
    color = COLORS[color_name]

    # Random size
    s = random.randint(10, 20)

    # Random position
    x = random.randint(3, SIZE - s - 3)
    y = random.randint(3, SIZE - s - 3)

    if shape == "circle":
        draw.ellipse(
            [x, y, x + s, y + s],
            fill=color
        )

    elif shape == "square":
        draw.rectangle(
            [x, y, x + s, y + s],
            fill=color
        )

    elif shape == "triangle":
        draw.polygon(
            [
                (x + s // 2, y),
                (x, y + s),
                (x + s, y + s)
            ],
            fill=color
        )

    filename = f"{i:05d}_{shape}_{color_name}.png"
    img.save(os.path.join(OUT_DIR, filename))

print(f"Created {N_IMAGES} images in {OUT_DIR}")