from core import DitheringMethod

import numpy as np


class BurkesDitherer(DitheringMethod):
    """Burkes dithering.

    A five-pixel-wide error diffusion method that produces very smooth
    gradients. It's essentially a simplified, faster version of Stucki
    that drops the two-rows-ahead component. Great for photographic
    images where you want minimal banding.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)

        # Burkes kernel (weights out of 32):
        #            [curr]  8  4
        #   2    4     8     4  2
        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                # Current row
                if x + 1 < width:
                    buf[y, x + 1] += error * 8 / 32
                if x + 2 < width:
                    buf[y, x + 2] += error * 4 / 32

                # Next row
                if y + 1 < height:
                    if x - 2 >= 0:
                        buf[y + 1, x - 2] += error * 2 / 32
                    if x - 1 >= 0:
                        buf[y + 1, x - 1] += error * 4 / 32
                    buf[y + 1, x] += error * 8 / 32
                    if x + 1 < width:
                        buf[y + 1, x + 1] += error * 4 / 32
                    if x + 2 < width:
                        buf[y + 1, x + 2] += error * 2 / 32

        return output
