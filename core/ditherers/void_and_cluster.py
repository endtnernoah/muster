from core import DitheringMethod

import numpy as np


class VoidAndClusterDitherer(DitheringMethod):
    """Void-and-cluster dithering using a blue noise mask.

    Produces very even, non-repetitive dot distributions that look organic
    and avoid the grid artifacts of ordered dithering. The blue noise mask
    is generated iteratively: the darkest cluster gets a 1, the brightest
    void gets a 0, repeating until the mask is filled.
    """

    def __init__(self, threshold: int = 128, mask_size: int = 64):
        super().__init__(threshold)
        self.mask = self._generate_blue_noise_mask(mask_size)

    def _gaussian_filter(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply a toroidal Gaussian blur via FFT for periodic boundary conditions."""
        h, w = image.shape
        ky = np.fft.fftfreq(h).reshape(-1, 1)
        kx = np.fft.fftfreq(w).reshape(1, -1)
        kernel = np.exp(-2 * (np.pi**2) * (sigma**2) * (kx**2 + ky**2))
        return np.real(np.fft.ifft2(np.fft.fft2(image) * kernel))

    def _generate_blue_noise_mask(self, size: int) -> np.ndarray:
        """Generate a blue noise threshold mask using void-and-cluster."""
        # Seed with sparse random points
        mask = np.zeros((size, size), dtype=np.float64)
        seed_density = 0.1
        seed = (np.random.random((size, size)) < seed_density).astype(np.float64)
        mask[:] = seed

        result = np.full((size, size), -1, dtype=np.int32)
        total = size * size
        ones_count = int(np.sum(seed))

        # Phase 1: assign values 0..ones_count-1 by removing clusters (darkest blurred spots)
        for value in range(ones_count - 1, -1, -1):
            blurred = self._gaussian_filter(mask, sigma=1.5)
            # Find the tightest cluster (highest blurred value among the 1s)
            candidates = np.where(mask > 0.5)
            if len(candidates[0]) == 0:
                break
            idx = np.argmax(blurred[candidates])
            y, x = candidates[0][idx], candidates[1][idx]
            result[y, x] = value
            mask[y, x] = 0.0

        # Phase 2: assign values ones_count..total-1 by filling voids (lowest blurred spots)
        for value in range(ones_count, total):
            blurred = self._gaussian_filter(mask, sigma=1.5)
            # Find the largest void (lowest blurred value among the 0s)
            candidates = np.where(mask < 0.5)
            if len(candidates[0]) == 0:
                break
            idx = np.argmin(blurred[candidates])
            y, x = candidates[0][idx], candidates[1][idx]
            result[y, x] = value
            mask[y, x] = 1.0

        # Normalize to [0, 255]
        return (result.astype(np.float64) / (total - 1)) * 255.0

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape
        n = self.mask.shape[0]
        # Tile the mask to cover the full image
        tiled = np.tile(self.mask, (height // n + 1, width // n + 1))[:height, :width]
        return np.where(pixels > tiled, 255, 0).astype(np.uint8)
