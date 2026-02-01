from core import DitheringMethod

import numpy as np


class StevensonZalkamanDitherer(DitheringMethod):
    """Stevenson-Zalakman dithering.

    An asymmetric three-row error diffusion method. The uneven weight
    distribution creates a distinctive diagonal texture that differs
    from the more symmetric Stucki or Burkes patterns. Looks particularly
    interesting on areas with sharp edges.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)

        # Stevenson-Zalakman kernel (weights out of 32):
        #              [curr]  8
        #      2        5     1  1
        #      1        1     1
        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                # Row 0
                if x + 1 < width:
                    buf[y, x + 1] += error * 8 / 32

                # Row +1
                if y + 1 < height:
                    if x - 2 >= 0:
                        buf[y + 1, x - 2] += error * 2 / 32
                    buf[y + 1, x] += error * 5 / 32
                    if x + 1 < width:
                        buf[y + 1, x + 1] += error * 1 / 32
                    if x + 2 < width:
                        buf[y + 1, x + 2] += error * 1 / 32

                # Row +2
                if y + 2 < height:
                    if x - 2 >= 0:
                        buf[y + 2, x - 2] += error * 1 / 32
                    buf[y + 2, x] += error * 1 / 32
                    if x + 2 < width:
                        buf[y + 2, x + 2] += error * 1 / 32

        return output
