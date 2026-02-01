from abc import ABC, abstractmethod
from PIL import Image
from typing import Union
import numpy as np


def is_video_file(filepath: str) -> bool:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    return filepath.lower().endswith(tuple(video_extensions))


def is_image_file(filepath: str) -> bool:
    return filepath.lower().endswith(tuple(Image.EXTENSION.keys()))


class DitheringMethod(ABC):
    """Abstract base class for implementing different dithering algorithms."""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold
        self._current_image = None

    def load(self, image: Union[str, Image.Image, np.ndarray]) -> "DitheringMethod":
        # Convert input to numpy array
        if isinstance(image, str):
            img = Image.open(image).convert("L")
            self._current_image = np.array(img, dtype=np.float32)
        elif isinstance(image, Image.Image):
            img = image.convert("L")
            self._current_image = np.array(img, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            self._current_image = image.astype(np.float32)
        else:
            raise ValueError("Image must be a file path, PIL Image, or numpy array")

        return self

    def scale(self, scale_factor: float) -> "DitheringMethod":
        if self._current_image is None:
            raise ValueError("No image loaded. Call load() first.")

        # Early exit
        if scale_factor == 1.0:
            return self

        height, width = self._current_image.shape

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        img = Image.fromarray(self._current_image.astype(np.uint8), mode="L")
        img_scaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self._current_image = np.array(img_scaled, dtype=np.float32)

        return self

    def apply(self) -> "DitheringMethod":
        if self._current_image is None:
            raise ValueError("No image loaded. Call load() first.")

        self._current_image = self._dither(self._current_image.copy())
        return self

    def set_threshold(self, threshold: int) -> "DitheringMethod":
        self.threshold = threshold
        return self

    def save(self, filepath: str) -> "DitheringMethod":
        if self._current_image is None:
            raise ValueError("No image to save. Call load() and apply() first.")

        img = Image.fromarray(self._current_image.astype(np.uint8), mode="L")
        img.save(filepath)
        return self

    def get_image(self) -> Image.Image:
        if self._current_image is None:
            raise ValueError("No image loaded.")

        return Image.fromarray(self._current_image.astype(np.uint8), mode="L")

    def get_array(self) -> np.ndarray:
        if self._current_image is None:
            raise ValueError("No image loaded.")

        return self._current_image.copy()

    def process_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        return self.load(image).apply().get_image()

    @abstractmethod
    def _dither(self, pixels: np.ndarray) -> np.ndarray:
        pass
