from core import DitheringMethod

import numpy as np


class SierraLiteDitherer(DitheringMethod):
    """Sierra Lite (two-row) dithering.

    A simplified two-row variant of Sierra dithering. Faster than
    Sierra 3 and produces a noticeably different texture — slightly
    grainier but still smoother than Floyd-Steinberg.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)

        # Sierra Lite kernel (weights out of 4):
        #   [curr]  +1
        #    +1     +1
        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                if x + 1 < width:
                    buf[y, x + 1] += error * 1 / 4
                if y + 1 < height:
                    buf[y + 1, x] += error * 1 / 4
                    if x + 1 < width:
                        buf[y + 1, x + 1] += error * 1 / 4
                    # Remaining 1/4 is intentionally lost (dampening)

        return output


class SierraTwoDitherer(DitheringMethod):
    """Sierra Two-Row dithering.

    The middle ground of the Sierra family. Uses two rows but a wider
    kernel than Sierra Lite, giving a balance of speed and quality.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)

        # Sierra Two-Row kernel (weights out of 8):
        #   [curr]  +2  +1
        #    +1     +2  +1
        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                if x + 1 < width:
                    buf[y, x + 1] += error * 2 / 8
                if x + 2 < width:
                    buf[y, x + 2] += error * 1 / 8

                if y + 1 < height:
                    if x - 1 >= 0:
                        buf[y + 1, x - 1] += error * 1 / 8
                    buf[y + 1, x] += error * 2 / 8
                    if x + 1 < width:
                        buf[y + 1, x + 1] += error * 1 / 8

        return output


class SierraThreeDitherer(DitheringMethod):
    """Sierra 3 (full) dithering.

    A three-row error diffusion method designed by Bill Sierra as a
    smoother alternative to Floyd-Steinberg. Spreads error across a
    wider neighborhood, resulting in softer, less directional artifacts.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        # Work on a float copy to accumulate error
        buf = pixels.astype(np.float64)

        # Sierra 3 kernel (offsets relative to current pixel, weight out of 32):
        #          [curr]  +2  +1
        #   +1      +3     +5  +3
        #           +1     +1
        # Total weights = 32
        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                # Row 0 (current row, pixels to the right)
                if x + 1 < width:
                    buf[y, x + 1] += error * 2 / 32
                if x + 2 < width:
                    buf[y, x + 2] += error * 1 / 32

                # Row +1
                if y + 1 < height:
                    if x - 1 >= 0:
                        buf[y + 1, x - 1] += error * 1 / 32
                    buf[y + 1, x] += error * 5 / 32
                    if x + 1 < width:
                        buf[y + 1, x + 1] += error * 3 / 32
                    if x + 2 < width:
                        buf[y + 1, x + 2] += (
                            error * 1 / 32
                        )  # Note: non-standard extension

                # Row +2
                if y + 2 < height:
                    buf[y + 2, x] += error * 1 / 32

        return output
