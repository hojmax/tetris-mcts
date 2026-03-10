from __future__ import annotations

from PIL import ImageDraw, ImageFont


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[float, float]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def value_to_pixel(
    value: float,
    value_min: float,
    value_max: float,
    pixel_min: float,
    pixel_max: float,
) -> int:
    """Map a data value to a pixel coordinate."""
    if value_max == value_min:
        return int((pixel_min + pixel_max) / 2)
    ratio = (value - value_min) / (value_max - value_min)
    return int(pixel_min + ratio * (pixel_max - pixel_min))
