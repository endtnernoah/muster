from core import DitheringMethod

import numpy as np


class DiffusionMapDitherer(DitheringMethod):
    """Diffusion map dithering using a precomputed anisotropic spread.

    Instead of a fixed error kernel, error is spread according to a
    per-pixel diffusion map derived from local image structure (gradient
    direction). This makes error flow along edges rather than across
    them, producing a texture that follows the contours of the image.
    Visually distinctive on anything with strong directional structure.
    """

    def __init__(self, threshold: int = 128):
        super().__init__(threshold)

    def _compute_structure_tensor(
        self, pixels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the local gradient direction and magnitude via structure tensor."""
        h, w = pixels.shape
        img = pixels.astype(np.float64)
        padded = np.pad(img, 1, mode="edge")

        # Sobel gradients
        gx = (
            padded[:-2, 2:]
            - padded[:-2, :-2]
            + 2 * (padded[1:-1, 2:] - padded[1:-1, :-2])
            + padded[2:, 2:]
            - padded[2:, :-2]
        )
        gy = (
            padded[2:, :-2]
            - padded[:-2, :-2]
            + 2 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
            + padded[2:, 2:]
            - padded[:-2, 2:]
        )

        # Gradient direction (perpendicular to edge = along-edge diffusion)
        direction = np.arctan2(gy, gx)
        magnitude = np.sqrt(gx**2 + gy**2)
        if magnitude.max() > 0:
            magnitude /= magnitude.max()

        return direction, magnitude

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)
        direction, magnitude = self._compute_structure_tensor(pixels)

        # Candidate diffusion neighbors and base weights (like Burkes)
        # We'll rotate/shift these based on local gradient direction
        base_offsets = [(0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)]  # FS as fallback

        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                mag = magnitude[y, x]

                if mag < 0.05:
                    # Flat region: standard Floyd-Steinberg
                    offsets = base_offsets
                    total_w = 16.0
                else:
                    # Strong gradient: spread error along the edge direction
                    # (perpendicular to the gradient)
                    angle = direction[y, x] + np.pi / 2  # perpendicular
                    dx_dir = np.cos(angle)
                    dy_dir = np.sin(angle)

                    # Build anisotropic kernel: stronger weight in the along-edge direction
                    offsets = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            # How aligned is this neighbor with the edge direction?
                            dot = dx * dx_dir + dy * dy_dir
                            # Weight: base weight biased by alignment
                            w = max(0.1, 1.0 + dot * mag * 2.0)
                            # Only spread forward (positive y, or same y positive x)
                            if dy > 0 or (dy == 0 and dx > 0):
                                offsets.append((dy, dx, w))

                    total_w = sum(w for _, _, w in offsets) if offsets else 1.0

                for dy, dx, w in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        buf[ny, nx] += error * (w / total_w)

        return output
