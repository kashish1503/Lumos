"""Apply edit parameters to an image with PIL."""

from PIL import Image, ImageEnhance


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def apply_edits(img: Image.Image, edits: dict) -> Image.Image:
    img = img.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(clamp(edits.get("brightness", 1), 0.2, 2))
    img = ImageEnhance.Contrast(img).enhance(clamp(edits.get("contrast", 1), 0.2, 2))
    img = ImageEnhance.Color(img).enhance(clamp(edits.get("saturation", 1), 0, 2))
    img = ImageEnhance.Sharpness(img).enhance(clamp(edits.get("sharpness", 1), 0, 3))

    # Warmth: boost red / cut blue (or the reverse for cooler). -50..50 -> -0.25..0.25
    w = clamp(edits.get("warmth", 0), -50, 50) / 200
    r, g, b = img.split()
    r = r.point(lambda x: min(255, int(x * (1 + w))))
    b = b.point(lambda x: min(255, int(x * (1 - w))))
    return Image.merge("RGB", (r, g, b))
