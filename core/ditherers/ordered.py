from core import DitheringMethod

import numpy as np


class OrderedDitherer(DitheringMethod):
    """Ordered (Bayer matrix) dithering."""

    def __init__(self, threshold: int = 128, matrix_size: int = 4):
        super().__init__(threshold)
        self.bayer_matrix = self._generate_bayer_matrix(matrix_size)

    def _generate_bayer_matrix(self, n: int) -> np.ndarray:
        """Generate a Bayer matrix of size n x n, normalized to [0, 255]."""
        return self._bayer_recursive(n) / (n * n) * 255

    def _bayer_recursive(self, n: int) -> np.ndarray:
        """Recursively build the raw (unnormalized) Bayer matrix."""
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float64)
        smaller = self._bayer_recursive(n // 2)
        return np.block(
            [
                [4 * smaller, 4 * smaller + 2],
                [4 * smaller + 3, 4 * smaller + 1],
            ]
        )

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        n = self.bayer_matrix.shape[0]
        output = np.zeros_like(pixels)

        for y in range(height):
            for x in range(width):
                threshold = self.bayer_matrix[y % n, x % n]
                output[y, x] = 255 if pixels[y, x] > threshold else 0

        return output
