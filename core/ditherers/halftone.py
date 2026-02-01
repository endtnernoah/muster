from core import DitheringMethod

import numpy as np


class HalftoneDitherer(DitheringMethod):
    """Halftone dithering using circular dot patterns.

    Simulates the classic print halftone effect where dots of varying
    size represent different tonal values. Looks great at larger cell
    sizes — try 6–12 for the most visible effect.
    """

    def __init__(self, threshold: int = 128, cell_size: int = 8):
        super().__init__(threshold)
        self.cell_size = cell_size
        self.dot_mask = self._precompute_dot_masks()

    def _precompute_dot_masks(self) -> list[np.ndarray]:
        """Precompute circular masks for each intensity level."""
        n = self.cell_size
        center = (n - 1) / 2.0
        # Distance from center for each pixel in the cell
        yy, xx = np.ogrid[:n, :n]
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        max_dist = center  # radius of the largest dot

        # Build one mask per intensity level (0 = all white, n*n = all black)
        levels = n * n
        masks = []
        for level in range(levels + 1):
            # Map level to a radius: level 0 -> radius 0, level max -> radius max_dist
            radius = max_dist * np.sqrt(level / levels)
            masks.append((dist <= radius).astype(np.uint8) * 255)

        return masks

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        n = self.cell_size
        output = np.full((height, width), 255, dtype=np.uint8)
        levels = n * n

        for cy in range(0, height, n):
            for cx in range(0, width, n):
                # Sample the cell — use the mean intensity
                cell = pixels[cy : cy + n, cx : cx + n]
                avg = 255 - np.mean(cell)  # invert: dark pixels -> big dots
                # Map to a dot level
                level = int(np.clip(round(avg / 255.0 * levels), 0, levels))
                mask = self.dot_masks[level]

                # Paste the mask (handle edge cells that are smaller)
                h = min(n, height - cy)
                w = min(n, width - cx)
                # Invert: dot area is black (0), background is white (255)
                output[cy : cy + h, cx : cx + w] = 255 - mask[:h, :w]

        return output

    @property
    def dot_masks(self) -> list[np.ndarray]:
        return self.dot_mask
