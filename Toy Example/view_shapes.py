import glob
import matplotlib.pyplot as plt
from PIL import Image

files = glob.glob("data/shapes/*.png")[:25]

fig, axes = plt.subplots(5, 5, figsize=(6, 6))

for ax, file in zip(axes.flat, files):
    ax.imshow(Image.open(file))
    ax.axis("off")

plt.tight_layout()
plt.show()