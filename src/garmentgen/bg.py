from __future__ import annotations

from PIL import Image


def remove_background_rgba(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    try:
        from rembg import remove

        rgba = remove(image)
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        return rgba
    except Exception:
        # Fallback: simple alpha matting by treating near-white as background
        # Not ideal, but keeps pipeline functional without rembg runtime.
        image = image.copy().convert("RGBA")
        datas = image.getdata()
        new_data = []
        for r, g, b, a in datas:
            if r > 240 and g > 240 and b > 240:
                new_data.append((r, g, b, 0))
            else:
                new_data.append((r, g, b, a))
        image.putdata(new_data)
        return image

