from core import DitheringMethod
from .atkinson import AtkinsonDitherer
from .floyd_steinberg import FloydSteinbergDitherer
from .ordered import OrderedDitherer
from .halftone import HalftoneDitherer
from .pencil_sketch import PencilSketchDitherer
from .diffusion_map import DiffusionMapDitherer
from .sierra import SierraLiteDitherer, SierraTwoDitherer, SierraThreeDitherer
from .void_and_cluster import VoidAndClusterDitherer
from .burkes import BurkesDitherer
from .stucki import StuckiDitherer
from .stevenson_zalkaman import StevensonZalkamanDitherer

import numpy as np
import random


# Algorithm darkness characteristics
# Values closer to 100 = lighter output, values closer to 0 = darker output
ALGORITHM_BRIGHTNESS = {
    "ordered": 50,  # Medium darkness
    "floyd_steinberg": 48,  # Slightly darker (aggressive diffusion)
    "atkinson": 60,  # Lighter (less error propagation)
    "threshold": 45,  # Darkest (sharp cutoff)
    "sierra_lite": 52,  # Medium-dark
    "sierra_two": 50,  # Medium darkness
    "sierra_three": 55,  # Medium-light
    "void_and_cluster": 58,  # Lighter (preserves details)
    "burkes": 49,  # Darker (strong diffusion)
    "stucki": 47,  # Darker (more aggressive)
    "stevenson_zalkaman": 54,  # Medium-light
    "halftone": 65,  # Lighter (artistic, preserves brightness)
    "pencil_sketch": 70,  # Much lighter (artistic effect)
    "diffusion_map": 61,  # Lighter (spread diffusion)
}


class ThresholdDitherer(DitheringMethod):
    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        return np.where(pixels > self.threshold, 255, 0)


class NoiseThresholdDitherer(DitheringMethod):
    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        noise = np.random.randint(
            -self.threshold, self.threshold + 1, size=pixels.shape
        )
        noisy_pixels = pixels + noise
        return np.where(noisy_pixels > 128, 255, 0)


class RandomDitherer(DitheringMethod):
    """Random ditherer that selects from various algorithms.

    Supports auto-threshold mode that adjusts threshold based on the selected
    algorithm's brightness characteristics to maintain consistent output brightness.
    """

    def __init__(
        self,
        threshold: int = 128,
        include_artistic: bool = False,
        auto_threshold: bool = False,
    ):
        super().__init__(threshold)
        self.auto_threshold = auto_threshold
        self.selected_ditherer = None

        self.ditherers_with_names = [
            ("ordered", OrderedDitherer(threshold)),
            ("floyd_steinberg", FloydSteinbergDitherer(threshold)),
            ("atkinson", AtkinsonDitherer(threshold)),
            ("sierra_lite", SierraLiteDitherer(threshold)),
            ("sierra_two", SierraTwoDitherer(threshold)),
            ("sierra_three", SierraThreeDitherer(threshold)),
            ("void_and_cluster", VoidAndClusterDitherer(threshold)),
            ("burkes", BurkesDitherer(threshold)),
            ("stucki", StuckiDitherer(threshold)),
            ("stevenson_zalkaman", StevensonZalkamanDitherer(threshold)),
        ]

        if include_artistic:
            self.ditherers_with_names.extend(
                [
                    ("halftone", HalftoneDitherer(threshold)),
                    ("pencil_sketch", PencilSketchDitherer(threshold)),
                    ("diffusion_map", DiffusionMapDitherer(threshold)),
                ]
            )

        # Keep original list for compatibility
        self.ditherers = [ditherer for _, ditherer in self.ditherers_with_names]

    def _apply_auto_threshold(self, ditherer_name: str) -> int:
        """Calculate adjusted threshold based on algorithm brightness.

        Args:
            ditherer_name: Name of the selected dithering algorithm

        Returns:
            Adjusted threshold value (0-255)
        """
        if ditherer_name not in ALGORITHM_BRIGHTNESS:
            return self.threshold

        # Get the brightness factor (0-100)
        brightness_factor = ALGORITHM_BRIGHTNESS[ditherer_name]

        # Map brightness factor to threshold adjustment
        # If algorithm is "lighter" (higher brightness), increase threshold
        # If algorithm is "darker" (lower brightness), decrease threshold
        base_threshold = 128
        adjustment = (brightness_factor - 50) * 1.28  # Scale to ±64 range
        adjusted_threshold = max(0, min(255, int(base_threshold + adjustment)))

        return adjusted_threshold

    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        # Select a random ditherer
        ditherer_name, ditherer = random.choice(self.ditherers_with_names)
        self.selected_ditherer = ditherer_name

        # Apply auto-threshold if enabled
        if self.auto_threshold:
            adjusted_threshold = self._apply_auto_threshold(ditherer_name)
            ditherer.set_threshold(adjusted_threshold)
        else:
            ditherer.set_threshold(self.threshold)

        return ditherer._dither(pixels)

    def get_selected_algorithm(self) -> str:
        """Get the name of the last selected algorithm."""
        return self.selected_ditherer or "unknown"


ALGORITHMS = {
    "ordered": OrderedDitherer,
    "floyd_steinberg": FloydSteinbergDitherer,
    "atkinson": AtkinsonDitherer,
    "threshold": ThresholdDitherer,
    "halftone": HalftoneDitherer,
    "pencil_sketch": PencilSketchDitherer,
    "diffusion_map": DiffusionMapDitherer,
    "sierra_lite": SierraLiteDitherer,
    "sierra_two": SierraTwoDitherer,
    "sierra_three": SierraThreeDitherer,
    "void_and_cluster": VoidAndClusterDitherer,
    "burkes": BurkesDitherer,
    "stucki": StuckiDitherer,
    "stevenson_zalkaman": StevensonZalkamanDitherer,
    "noise_threshold": NoiseThresholdDitherer,
    "random": RandomDitherer,
}
