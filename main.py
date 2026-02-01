#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field

from PIL import Image
import click

from core.ditherers import ALGORITHMS
from core.video import VideoProcessor
from core import is_video_file, is_image_file

__version__ = "0.1.0"
script_name = os.path.basename(sys.argv[0])


# Configuration dataclass
@dataclass
class Config:
    debug: bool = field(default=False)
    quiet: bool = field(default=False)

    # must-haves
    input_file: Path = field(default=None)
    output_file: Path = field(default="output.png")

    # control options
    threshold: int = field(default=128)
    scale: float = field(default=1.0)
    algorithm: str = field(default="ordered")
    auto_threshold: bool = field(default=False)


# Yes, I use real logging like a grown-up. fuck print()
def setup_logging(debug: bool, quiet: bool):
    """Configure logging based on debug and quiet flags."""
    if quiet:
        logger.setLevel(logging.ERROR)
    elif debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


class Formatter(logging.Formatter):
    """
    https://stackoverflow.com/questions/14844970/\
    modifying-logging-message-format-based-on-message-logging-level-in-python3
    """

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


# Set up logging to console
logger = logging.getLogger(script_name)
handler = logging.StreamHandler()
handler.setFormatter(Formatter())
logger.addHandler(handler)


# Default options
@click.command()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with verbose logging.",
)
@click.option(
    "--quiet", is_flag=True, default=False, help="Suppress all output except errors."
)

# Must-haves
@click.option("-i", "--input-file", required=True, type=click.Path(exists=True))
@click.option("-o", "--output-file", type=click.Path())
@click.option(
    "--algorithm",
    type=click.Choice(ALGORITHMS.keys(), case_sensitive=False),
    default="ordered",
    help="Dithering algorithm to use.",
)

# Control options
@click.option(
    "--threshold", type=int, default=128, help="Threshold value for dithering."
)
@click.option("--scale", type=float, default=1.0, help="Scale factor for the image.")
@click.option(
    "--auto-threshold",
    is_flag=True,
    default=False,
    help="Auto-adjust threshold based on algorithm darkness (works best with 'random' algorithm).",
)

# Auto version and help options
@click.version_option(version=__version__, prog_name=script_name)
@click.help_option("--help", "-h", "-?", help="Show this message and exit.")
def ctx(**kwargs):
    # Easier to fetch parameters this way
    config = Config(**kwargs)

    # Configure logging behavior
    setup_logging(config.debug, config.quiet)
    logger.debug(f"Parameters: \n{json.dumps(kwargs, indent=2)}")

    # Ensure PIL image plugins are loaded
    Image.init()

    #
    # Sanity checks
    #

    # Only allow certain file extensions (image or video)
    if not (is_image_file(config.input_file) or is_video_file(config.input_file)):
        Image.init()
        video_ext = ", ".join(
            {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
        )
        image_ext = ", ".join(Image.EXTENSION.keys())
        logger.error(
            f"Input file must be an image ({image_ext}) or video ({video_ext})"
        )
        sys.exit(1)

    # Default output file should always be out/dithered_<original_filename> if not specified
    if config.output_file is None:
        config.output_file = os.path.join(
            "out", f"dithered_{os.path.basename(config.input_file)}"
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We do not scale over 3x for now
    if config.scale > 3.0:
        logger.error("Scale factor cannot be greater than 3.0")
        sys.exit(1)

    # We do not scale below 0.01x for now
    if config.scale < 0.01:
        logger.error("Scale factor cannot be less than 0.01")
        sys.exit(1)

    # Threshold must be between 0 and 255
    if not (0 <= config.threshold <= 255):
        logger.error("Threshold must be between 0 and 255")
        sys.exit(1)

    # Validate algorithm choice
    valid_algorithms = ALGORITHMS.keys()
    if config.algorithm.lower() not in valid_algorithms:
        logger.error(f"Algorithm must be one of: {', '.join(valid_algorithms)}")
        sys.exit(1)

    # Execute main application logic
    main(config)


# Main application logic
def main(config: Config = None):
    if config is None:
        raise ValueError("Config must be provided to main().")

    try:
        logger.info("Starting dithering process...")

        # Select ditherer based on algorithm choice
        ditherer_class = ALGORITHMS[config.algorithm.lower()]
        
        # Handle RandomDitherer with auto_threshold option
        if config.algorithm.lower() == "random":
            ditherer = ditherer_class(threshold=config.threshold, auto_threshold=config.auto_threshold)
        else:
            ditherer = ditherer_class(threshold=config.threshold)

        # Process image or video
        if is_video_file(config.input_file):
            processor = VideoProcessor(ditherer)
            (
                processor.load(config.input_file)
                .scale(config.scale)
                .apply()
                .save(config.output_file)
            )
        else:
            # Original image processing pipeline
            (
                ditherer.load(config.input_file)
                .set_threshold(config.threshold)
                .scale(config.scale)
                .apply()
                .save(config.output_file)
            )

        logger.info(f"Dithering completed. Output saved to {config.output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if logger.level == logging.DEBUG:
            logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    ctx()
