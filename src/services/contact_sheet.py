"""Contact sheets: candidate images in one numbered grid, for a vision model to choose from.

Each tile is cropped to the video's aspect ratio (as the frame will show it), so the model
judges what the viewer will actually see. One image per call = one request, few tokens.

    jpeg = grid([bytes, bytes, ...], aspect=9/16)    # tiles numbered 1..n
"""
import io
import math

from PIL import Image, ImageDraw, ImageFont, ImageOps

TILE_LONG = 400               # px of a tile's long side


def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def grid(images, aspect, columns=None):
    """images: JPEG/PNG bytes. aspect: width/height of the video frame. -> JPEG bytes."""
    tile_w, tile_h = (TILE_LONG, round(TILE_LONG / aspect)) if aspect >= 1 else (round(TILE_LONG * aspect), TILE_LONG)
    columns = columns or (3 if aspect >= 1 else 4)
    rows = math.ceil(len(images) / columns)
    gap = 8
    sheet = Image.new("RGB", (columns * tile_w + (columns + 1) * gap, rows * tile_h + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(round(tile_h * 0.12))
    for n, data in enumerate(images):
        tile = ImageOps.fit(Image.open(io.BytesIO(data)).convert("RGB"), (tile_w, tile_h), Image.LANCZOS)
        x, y = gap + (n % columns) * (tile_w + gap), gap + (n // columns) * (tile_h + gap)
        sheet.paste(tile, (x, y))
        label = str(n + 1)
        box = draw.textbbox((0, 0), label, font=font)
        pad = round(tile_h * 0.02)
        draw.rectangle((x, y, x + box[2] + 3 * pad, y + box[3] + 3 * pad), fill="black")
        draw.text((x + pad, y + pad), label, font=font, fill="yellow")
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=88)
    return out.getvalue()
