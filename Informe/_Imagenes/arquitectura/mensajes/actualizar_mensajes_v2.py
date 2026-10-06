#!/usr/bin/env python3
"""Actualiza el nombre del tópico de suelo en el diagrama fuente de v2."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE = Path(__file__).resolve().parent / "mensajes-v2.png"
FONT = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"


def main():
    image = Image.open(IMAGE).convert("RGBA")
    draw = ImageDraw.Draw(image)
    # La lámina original tiene fondo transparente. Se reemplaza solamente la
    # línea del tópico; las dos líneas que describen su tipo y marco se conservan.
    draw.rectangle((1930, 872, 3080, 950), fill=(255, 255, 255, 0))
    draw.text(
        (2505, 911),
        "/depth_obstacle_filter/ground_evidence",
        font=ImageFont.truetype(FONT, 52),
        fill=(0, 0, 0, 255),
        anchor="mm",
    )
    image.save(IMAGE)


if __name__ == "__main__":
    main()
