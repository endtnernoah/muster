from core import DitheringMethod

import numpy as np


class PencilSketchDitherer(DitheringMethod):
    """Pencil-sketch style dithering using edge-aware error diffusion.

    Detects edges first via a Sobel filter, then applies error diffusion
    that is dampened along edges and stronger in flat regions. The result
    keeps sharp outlines crisp while dithering smooth areas normally —
    giving a hand-drawn pencil sketch feel.
    """

    def __init__(self, threshold: int = 128, edge_strength: float = 0.6):
        super().__init__(threshold)
        self.edge_strength = edge_strength

    def _sobel_edges(self, pixels: np.ndarray) -> np.ndarray:
        """Compute edge magnitude using Sobel filters (no scipy dependency)."""
        # Sobel kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        h, w = pixels.shape
        padded = np.pad(pixels.astype(np.float64), 1, mode="edge")
        gx = np.zeros((h, w), dtype=np.float64)
        gy = np.zeros((h, w), dtype=np.float64)

        for dy in range(3):
            for dx in range(3):
                gx += padded[dy : dy + h, dx : dx + w] * kx[dy, dx]
                gy += padded[dy : dy + h, dx : dx + w] * ky[dy, dx]

        mag = np.sqrt(gx**2 + gy**2)
        # Normalize to [0, 1]
        if mag.max() > 0:
            mag /= mag.max()
        return mag

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        output = np.zeros_like(pixels)
        buf = pixels.astype(np.float64)
        edges = self._sobel_edges(pixels)

        # Floyd-Steinberg weights
        fs_weights = [(0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)]

        for y in range(height):
            for x in range(width):
                val = np.clip(buf[y, x], 0, 255)
                quantized = 255 if val > self.threshold else 0
                output[y, x] = quantized
                error = val - quantized

                if error == 0:
                    continue

                # Dampen diffusion along edges: strong edges keep error local
                edge_factor = 1.0 - (edges[y, x] * self.edge_strength)

                for dy, dx, w in fs_weights:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        buf[ny, nx] += error * (w / 16.0) * edge_factor

        return output
