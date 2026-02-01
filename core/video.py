import logging
import os
import cv2
import numpy as np

from core import DitheringMethod

logger = logging.getLogger(os.path.basename(os.path.dirname(__file__)))


class VideoProcessor:
    """Process video files by applying dithering to each frame.

    Supports the same fluent interface as DitheringMethod:
        VideoProcessor(ditherer).load("input.mp4").scale(0.5).apply().save("output.mp4")
    """

    def __init__(self, ditherer: DitheringMethod):
        self.ditherer = ditherer
        self.logger = logging.getLogger(__name__)

        # Source video state (set by load)
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0

        # Pipeline state
        self._scale_factor: float = 1.0
        self._frames: list[np.ndarray] = []  # holds processed frames after apply()
        self._applied: bool = False

    # Chainable API

    def load(self, input_path: str) -> "VideoProcessor":
        """Open a video file and read all frames into memory as grayscale."""
        self._reset_pipeline()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.debug(
            f"Loaded video: {self._width}x{self._height} @ {self._fps} fps, "
            f"{self._total_frames} frames"
        )

        # Read every frame as grayscale into _frames
        self._frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._frames.append(gray)

        cap.release()
        self._applied = False
        return self

    def scale(self, scale_factor: float) -> "VideoProcessor":
        """Resize every loaded frame by the given factor."""
        if not self._frames:
            raise ValueError("No video loaded. Call load() first.")

        if scale_factor == 1.0:
            return self

        self._scale_factor = scale_factor
        new_width = int(self._width * scale_factor)
        new_height = int(self._height * scale_factor)

        self._frames = [
            cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            for frame in self._frames
        ]

        # Update tracked dimensions so save() uses the correct size
        self._width = new_width
        self._height = new_height

        self.logger.debug(f"Scaled video to {self._width}x{self._height}")
        return self

    def apply(self) -> "VideoProcessor":
        """Run the dithering algorithm on every frame."""
        if not self._frames:
            raise ValueError("No video loaded. Call load() first.")

        self._frames = [
            self.ditherer.load(frame).apply().get_array().astype(np.uint8)
            for frame in self._frames
        ]

        self._applied = True

        self.logger.info(f"Dithering applied to {len(self._frames)} frames")
        return self

    def save(self, output_path: str) -> "VideoProcessor":
        """Write all processed frames to a video file."""
        if not self._frames:
            raise ValueError("No frames to save. Call load() and apply() first.")
        if not self._applied:
            raise ValueError("Dithering not applied. Call apply() before save().")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, self._fps, (self._width, self._height), isColor=False
        )

        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        try:
            for frame in self._frames:
                out.write(frame)
            self.logger.info(f"Video saved: {output_path} ({len(self._frames)} frames)")
        finally:
            out.release()

        return self

    # Accessors

    def get_frames(self) -> list[np.ndarray]:
        """Return a copy of the current frame list."""
        if not self._frames:
            raise ValueError("No video loaded.")
        return [f.copy() for f in self._frames]

    def get_frame(self, index: int) -> np.ndarray:
        """Return a single frame by index."""
        if not self._frames:
            raise ValueError("No video loaded.")
        if index < 0 or index >= len(self._frames):
            raise IndexError(
                f"Frame index {index} out of range (0–{len(self._frames) - 1})"
            )
        return self._frames[index].copy()

    # Helpers

    def _reset_pipeline(self) -> None:
        """Clear all in-memory state so load() can be called again cleanly."""
        self._frames = []
        self._scale_factor = 1.0
        self._applied = False
