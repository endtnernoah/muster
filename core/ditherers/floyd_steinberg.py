from core import DitheringMethod
import numpy as np


class FloydSteinbergDitherer(DitheringMethod):
    """Floyd-Steinberg error diffusion dithering."""

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = pixels.copy()

        for y in range(height):
            for x in range(width):
                old_pixel = output[y, x]
                new_pixel = 255 if old_pixel > self.threshold else 0
                output[y, x] = new_pixel
                error = old_pixel - new_pixel

                # Distribute error to neighboring pixels
                if x + 1 < width:
                    output[y, x + 1] += error * 7 / 16
                if y + 1 < height:
                    if x > 0:
                        output[y + 1, x - 1] += error * 3 / 16
                    output[y + 1, x] += error * 5 / 16
                    if x + 1 < width:
                        output[y + 1, x + 1] += error * 1 / 16

        return output
