from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def generate_dotblot(
    rows: int = 4,
    cols: int = 6,
    height: int = 420,
    width: int = 640,
    invert: bool = False,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    image = 20 + 0.015 * xx + 0.02 * yy
    anchor_x, anchor_y = 100, 80
    pitch_x, pitch_y = 80, 70
    radius = 16

    for row in range(rows):
        for col in range(cols):
            center_x = anchor_x + col * pitch_x + 0.05 * row * pitch_y
            center_y = anchor_y + row * pitch_y
            amplitude = 180 - row * 18 - col * 12
            blob = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * radius**2)))
            image += amplitude * blob

    image = np.clip(image, 0, 255)
    if invert:
        image = image.max() - image
    return image.astype(np.uint8)


def main():
    output_dir = Path(__file__).resolve().parent
    bright = generate_dotblot(invert=False)
    dark = generate_dotblot(invert=True)
    bright_16bit = (bright.astype(np.uint16) * 257)
    dark_16bit = (dark.astype(np.uint16) * 257)
    Image.fromarray(bright).save(output_dir / "synthetic_dotblot_bright.png")
    Image.fromarray(dark).save(output_dir / "synthetic_dotblot_dark.png")
    tifffile.imwrite(output_dir / "synthetic_dotblot_bright_16bit.tiff", bright_16bit)
    tifffile.imwrite(output_dir / "synthetic_dotblot_dark_16bit.tiff", dark_16bit)


if __name__ == "__main__":
    main()
