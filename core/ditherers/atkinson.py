from core import DitheringMethod
import numpy as np


class AtkinsonDitherer(DitheringMethod):
    """Implements the Atkinson dithering algorithm."""

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape

        for y in range(height):
            for x in range(width):
                old_pixel = pixels[y, x]
                new_pixel = 255 if old_pixel >= self.threshold else 0
                pixels[y, x] = new_pixel
                quant_error = old_pixel - new_pixel

                if x + 1 < width:
                    pixels[y, x + 1] += quant_error * 1 / 8
                if x + 2 < width:
                    pixels[y, x + 2] += quant_error * 1 / 8
                if y + 1 < height:
                    if x - 1 >= 0:
                        pixels[y + 1, x - 1] += quant_error * 1 / 8
                    pixels[y + 1, x] += quant_error * 1 / 8
                    if x + 1 < width:
                        pixels[y + 1, x + 1] += quant_error * 1 / 8
                if y + 2 < height:
                    pixels[y + 2, x] += quant_error * 1 / 8

        return np.clip(pixels, 0, 255)
