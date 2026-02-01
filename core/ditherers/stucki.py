from core import DitheringMethod

import numpy as np


class StuckiDitherer(DitheringMethod):
    """Stucki dithering.

    A wide three-row error diffusion method that produces exceptionally
    smooth results, especially in large flat gradient areas. Slower than
    Burkes due to the larger kernel, but noticeably higher quality on
    smooth images.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)

        # Stucki kernel (weights out of 42):
        #              [curr]  8  4
        #   1   2       4     8  4  2  1
        #   1   1       2     4  2  1  1
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
                    buf[y, x + 1] += error * 8 / 42
                if x + 2 < width:
                    buf[y, x + 2] += error * 4 / 42

                # Row +1
                if y + 1 < height:
                    for dx, w in [
                        (-2, 1),
                        (-1, 2),
                        (0, 4),
                        (1, 8),
                        (2, 4),
                        (3, 2),
                        (4, 1),
                    ]:
                        nx = x + dx
                        if 0 <= nx < width:
                            buf[y + 1, nx] += error * w / 42

                # Row +2
                if y + 2 < height:
                    for dx, w in [
                        (-2, 1),
                        (-1, 1),
                        (0, 2),
                        (1, 4),
                        (2, 2),
                        (3, 1),
                        (4, 1),
                    ]:
                        nx = x + dx
                        if 0 <= nx < width:
                            buf[y + 2, nx] += error * w / 42

        return output
